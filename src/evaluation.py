"""Evaluate trained Siamese models against labeled datasets."""

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tqdm import tqdm

from siamese.config import load_config
from siamese.datasets import LabeledCombinedDataset
from siamese.model import SharedSiamese
from siamese.streaming import IMAGENET_MEAN, IMAGENET_STD, AdaptiveNormalisation


@torch.no_grad()
def validate(
    shoeprint_model: SharedSiamese,
    shoemark_model: SharedSiamese,
    shoeprint_adaptive_norm: AdaptiveNormalisation,
    shoemark_adaptive_norm: AdaptiveNormalisation,
    *,
    dataset: LabeledCombinedDataset,
    device,
    p: int = 5,
    p_val: int = 2,
    checkpoint: str | Path | None = None,
    failure_dir: Path | str | None = None,
):
    """Evaluate model using all shoemarks in a dataset.

    Returns the proportion of shoemarks ranked within the top p percent of
    shoeprints. If failure_dir is given, shoemarks ranked outside the top p
    percent are copied to it from the dataset's shoemark directory.
    """
    shoeprint_model.eval()
    shoemark_model.eval()

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        shoeprint_model.load_state_dict(checkpoint["shoeprint_model_state_dict"])  # pyright: ignore
        shoemark_model.load_state_dict(checkpoint["shoemark_model_state_dict"])  # pyright: ignore
        shoeprint_adaptive_norm.load_state_dict(
            checkpoint["shoeprint_adaptive_norm_state_dict"]  # pyright: ignore
        )
        shoemark_adaptive_norm.load_state_dict(
            checkpoint["shoemark_adaptive_norm_state_dict"]  # pyright: ignore
        )

    shoeprint_embeddings = defaultdict(lambda: torch.zeros(1))
    shoemark_embeddings = defaultdict(lambda: torch.zeros(1))

    for shoeprint_class, (shoeprint, shoemarks) in dataset:
        normalised_shoeprint = shoeprint_adaptive_norm(
            shoeprint.to(shoeprint_adaptive_norm.device), update=False
        )
        shoeprint_embedding = shoeprint_model(normalised_shoeprint.to(device)).cpu()
        shoeprint_embeddings[shoeprint_class] = shoeprint_embedding.squeeze()

        # Not as fast as batching all shoemarks but works for very large numbers
        if len(shoemarks) > 0:
            shoemark_embeddings[shoeprint_class] = torch.cat(
                [
                    shoemark_model(
                        shoemark_adaptive_norm(
                            shoemark.to(shoemark_adaptive_norm.device), update=False
                        ).to(device)
                    ).cpu()
                    for shoemark in shoemarks
                ]
            )

    shoeprint_class_idxs = list(shoeprint_embeddings.keys())
    shoeprint_embeddings = torch.stack(list(shoeprint_embeddings.values()))

    shoeprint_model.train()
    shoemark_model.train()

    # ranks are 0-based, so a mark scores iff rank < k (the top k prints).
    # p=0 is S1: the correct print must rank first
    k = 1 if p == 0 else math.ceil(max(1, len(shoeprint_embeddings) * p / 100))

    ranks = []
    for shoe_id, class_shoemark_embeddings in shoemark_embeddings.items():
        # Compare distance between shoemark embedding and _all_ shoeprint embeddings

        dists = torch.cdist(
            class_shoemark_embeddings,
            shoeprint_embeddings,
            p=p_val,
        )

        # Get indices of distances sorted small->large
        sorted_dists = torch.argsort(dists)

        correct_idx = shoeprint_class_idxs.index(shoe_id)
        shoemark_ranks = (sorted_dists == correct_idx).nonzero()

        for shoemark_id, rank in enumerate([t.item() for t in shoemark_ranks[:, 1]]):
            ranks.append(rank)
            if failure_dir and rank >= k:
                shutil.copy(
                    dataset.shoemark_path / f"{shoe_id}_{shoemark_id}.png",
                    failure_dir,
                )

    ranks = np.array(ranks)

    return np.mean(ranks < k)


def validate_all_checkpoints(
    *,
    dataset: LabeledCombinedDataset,
    checkpoint_dir: Path | str,
    device: torch.device,
    p: int = 5,
    p_val: int = 2,
    log_name: str = "siamese_test.log",
):
    """Validate all checkpoints in a directory.

    The model architecture is inferred from the checkpoints themselves, so
    this works for runs whose config file no longer exists.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.tar"))
    if not checkpoints:
        msg = f"No checkpoints found in {checkpoint_dir}"
        raise FileNotFoundError(msg)

    first = torch.load(checkpoints[0], map_location="cpu")
    embedding_size = first["shoeprint_model_state_dict"]["model.fc.weight"].shape[0]

    shoeprint_model = SharedSiamese(embedding_size=embedding_size).to(device)
    shoemark_model = SharedSiamese(embedding_size=embedding_size).to(device)

    # Initial statistics are replaced by each checkpoint's saved state
    shoeprint_adaptive_norm = AdaptiveNormalisation(
        IMAGENET_MEAN, IMAGENET_STD, device=device
    )
    shoemark_adaptive_norm = AdaptiveNormalisation(
        IMAGENET_MEAN, IMAGENET_STD, device=device
    )

    scores = defaultdict(float)

    for checkpoint in tqdm(checkpoints):
        epoch = int(checkpoint.stem.split("_")[1])
        score = validate(
            shoeprint_model,
            shoemark_model,
            shoeprint_adaptive_norm,
            shoemark_adaptive_norm,
            dataset=dataset,
            device=device,
            p=p,
            p_val=p_val,
            checkpoint=checkpoint,
        )
        scores[epoch] = cast(float, score)

    with (checkpoint_dir / log_name).open("a") as f:
        for epoch, score in sorted(scores.items()):
            f.write(f"Epoch {epoch} p{p} validation: = {score}\n")


# Usage: python src/evaluation.py <checkpoint_dir> [config] [--dataset wvu|test|val]
# The config is only used for data paths, the GPU number, and p_val — the
# model architecture is read from the checkpoints, so the run's own config
# file is not needed.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint in a directory on a labeled dataset."
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("config", nargs="?", default="config.toml")
    parser.add_argument("--dataset", choices=["wvu", "test", "val"], default="wvu")
    parser.add_argument("-p", type=int, default=5, help="Top percentage rank cutoff")
    parser.add_argument(
        "--log-name",
        default=None,
        help="results file written inside checkpoint_dir "
        "(default: siamese_<dataset>.log)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(
        f"cuda:{config.training.gpu_number}" if torch.cuda.is_available() else "cpu"
    )

    data_dir = {
        "wvu": config.data.wvu_data_dir,
        "test": config.data.test_dir,
        "val": config.data.val_dir,
    }[args.dataset]

    dataset = LabeledCombinedDataset(
        data_dir / "Shoeprints",
        data_dir / "Shoemarks",
    )

    validate_all_checkpoints(
        dataset=dataset,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        p=args.p,
        p_val=config.hyperparameters.p_val,
        log_name=args.log_name or f"siamese_{args.dataset}.log",
    )
