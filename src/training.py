"""Train a Siamese model on streamed real and pooled synthetic shoe impressions."""

import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluation import validate
from siamese.config import parse_config
from siamese.datasets import LabeledCombinedDataset
from siamese.model import SharedSiamese
from siamese.streaming import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AdaptiveNormalisation,
    DifficultyScheduler,
    StreamingDataset,
    StreamingTransforms,
    augment_batch,
)

# * Config


config = parse_config()

# * Seeding


def seed_worker(worker_id):
    """Seed DataLoader workers with random seed."""
    worker_seed = (
        config.training.seed + worker_id
    ) % 2**32  # Ensure we don't overflow 32 bit
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)


# Passed to dataloaders
dataloader_g = torch.Generator()
dataloader_g.manual_seed(config.training.seed)


torch.manual_seed(config.training.seed)
np.random.default_rng(config.training.seed)
random.seed(config.training.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)

# * PyTorch

torch.backends.fp32_precision = "tf32"  # pyright: ignore [reportAttributeAccessIssue]
torch.backends.cuda.fp32_precision = "tf32"  # pyright: ignore [reportAttributeAccessIssue]
torch.backends.cudnn.fp32_precision = "tf32"  # pyright: ignore [reportAttributeAccessIssue]


device = torch.device(
    f"cuda:{config.training.gpu_number}" if torch.cuda.is_available() else "cpu"
)

# * Curriculum

# The dataset draws each batch's synthetic marks from the difficulty-stratified
# pool tree nearest the scheduled level (see StreamingDataset)
difficulty_scheduler = (
    DifficultyScheduler(
        initial_difficulty=config.curriculum.initial_difficulty,
        max_difficulty=config.curriculum.max_difficulty,
        peak_steps=config.curriculum.peak_steps,
    )
    if config.curriculum.enabled
    else None
)

# * Model

shoeprint_model = SharedSiamese(
    embedding_size=config.hyperparameters.embedding_size,
    pre_trained=config.training.pre_training.pre_trained,
    refreeze=config.training.pre_training.refreeze,
    permafrost=config.training.pre_training.permafrost,
    gradient_checkpointing=config.training.pre_training.gradient_checkpointing,
).to(device)

shoemark_model = SharedSiamese(
    embedding_size=config.hyperparameters.embedding_size,
    pre_trained=config.training.pre_training.pre_trained,
    refreeze=config.training.pre_training.refreeze,
    permafrost=config.training.pre_training.permafrost,
    gradient_checkpointing=config.training.pre_training.gradient_checkpointing,
).to(device)

shoeprint_optimizer = torch.optim.AdamW(
    shoeprint_model.parameters(), lr=0.001, weight_decay=1e-4
)
shoemark_optimizer = torch.optim.AdamW(
    shoemark_model.parameters(), lr=0.001, weight_decay=1e-4
)

shoeprint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    shoeprint_optimizer, T_max=config.training.epochs
)  # T_max is the maximum
shoemark_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    shoemark_optimizer, T_max=config.training.epochs
)  # T_max is the maximum


criterion = torch.nn.TripletMarginLoss(
    margin=config.hyperparameters.margin,
    p=config.hyperparameters.p_val,
    swap=config.hyperparameters.triplet_swapping,
)

# * Data

# ** Transforms

streaming_transform = StreamingTransforms(
    asdict(config.augmentations), size=config.data.image_size
)

shoeprint_adaptive_norm = AdaptiveNormalisation(
    IMAGENET_MEAN, IMAGENET_STD, device=device, momentum=0.9
)
shoemark_adaptive_norm = AdaptiveNormalisation(
    IMAGENET_MEAN, IMAGENET_STD, device=device, momentum=0.9
)


# ** Training

dataset = StreamingDataset(
    config.data.streaming.shoeprint_data_dir,
    config.data.streaming.shoemark_data_dir,
    config.data.streaming.floor_image_data_dir,
    config.data.image_size,
    config.data.streaming.min_floor_roi_height,
    config.data.streaming.synthetic_ratio,
    synthetic_shoemark_path=config.data.streaming.synthetic_shoemark_data_dir,
    real_pairs_only=config.data.streaming.real_pairs_only,
)
# When a pre-generated pool is configured, the dataset streams synthetic marks
# from disk; otherwise grayscale shoeprints stand in for them
synthetic_pool = config.data.streaming.synthetic_shoemark_data_dir is not None

class _EpochStreamBatchSampler:
    """Endless stream of per-epoch permutation batches.

    Same shuffling and batch composition as shuffle=True/drop_last=False, but
    never exhausts, so one DataLoader iterator serves the whole run and worker
    prefetch flows across epoch boundaries. Per-epoch iterators spend most of
    their life filling or draining at ~17 batches per epoch, which serialised
    the loader and the GPU (square-wave utilisation). The training loop draws
    ceil(len(dataset) / batch_size) batches per epoch.
    """

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        self.n = n
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        while True:
            perm = torch.randperm(self.n, generator=self.generator).tolist()
            for i in range(0, self.n, self.batch_size):
                yield perm[i : i + self.batch_size]


batches_per_epoch = math.ceil(len(dataset) / config.hyperparameters.batch_size)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=_EpochStreamBatchSampler(
        len(dataset), config.hyperparameters.batch_size, dataloader_g
    ),
    # Pooled runs decode synthetic marks from disk in __getitem__, so workers
    # overlap that with the GPU step; dataset.difficulty reaches them through
    # shared memory. In-loop runs keep 0: everything is preloaded in RAM
    num_workers=16 if synthetic_pool else 0,
    pin_memory=True,
    worker_init_fn=seed_worker,
)

# ** Validation

val_dataset = LabeledCombinedDataset(
    config.data.val_dir / "Shoeprints",
    config.data.val_dir / "Shoemarks",
)

# * Main loop

# TODO extract into a separate function
# TODO start at 1
start_epoch = 0
if config.training.resume_checkpoint:
    checkpoint = torch.load(config.training.resume_checkpoint, map_location=device)
    shoeprint_model.load_state_dict(checkpoint["shoeprint_model_state_dict"])
    shoemark_model.load_state_dict(checkpoint["shoemark_model_state_dict"])
    shoeprint_adaptive_norm.load_state_dict(
        checkpoint["shoeprint_adaptive_norm_state_dict"]
    )
    shoemark_adaptive_norm.load_state_dict(
        checkpoint["shoemark_adaptive_norm_state_dict"]
    )
    shoeprint_optimizer.load_state_dict(checkpoint["shoeprint_optim_state_dict"])
    shoemark_optimizer.load_state_dict(checkpoint["shoemark_optim_state_dict"])
    if (
        difficulty_scheduler is not None
        and "difficulty_scheduler_state_dict" in checkpoint
    ):
        difficulty_scheduler.load_state_dict(
            checkpoint["difficulty_scheduler_state_dict"]
        )
    start_epoch = int(config.training.resume_checkpoint.stem.split("_")[-1])

    if "shoeprint_scheduler_state_dict" in checkpoint:
        shoeprint_scheduler.load_state_dict(checkpoint["shoeprint_scheduler_state_dict"])
        shoemark_scheduler.load_state_dict(checkpoint["shoemark_scheduler_state_dict"])
    else:
        # Checkpoints predating scheduler state: replay the per-epoch steps the
        # original run had taken by the start of the resume epoch
        for _ in range(start_epoch):
            shoeprint_scheduler.step()
            shoemark_scheduler.step()

    # FIXME this wont currently work if epochs start unfreezing later than 0
    unfrozen_layers = start_epoch // config.training.pre_training.epoch_unfreeze
    shoeprint_model.unfreeze_to(unfrozen_layers)
    shoemark_model.unfreeze_to(unfrozen_layers)


def _write_line(line: str, pbar: tqdm, checkpoint_dir: Path):
    pbar.write(line, end="")
    with (checkpoint_dir / "siamese.log").open("a") as f:
        f.write(line)


def _select_negatives(dists: torch.Tensor, current_batch_size: int):
    """Select the hardest semi-hard negative shoemark for each shoeprint.

    Finds negatives closer to the anchor than positives, violating
    d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin.
    Falls back to a random negative when there are no violations.
    """
    # Positive distances
    pos_dists = dists.diag().view(-1, 1)

    # Mask to exclude the positive pairs (0s apart from the diagonal)
    idt_mask = torch.eye(pos_dists.size(0), dtype=torch.bool, device=device)

    # Identify semi-hard violations
    semi_hard_mask = (dists > pos_dists) & (
        dists < pos_dists + config.hyperparameters.margin
    )
    semi_hard_mask[idt_mask] = False

    # Store indices of selected negatives
    neg_idxs = []
    for i in range(current_batch_size):
        violation_inds = torch.where(semi_hard_mask[i])[0]

        if len(violation_inds) > 0:
            # Get hardest violation
            hardest_violation_idx = violation_inds[
                torch.argmin(dists[i, violation_inds])
            ]
            neg_idxs.append(hardest_violation_idx.item())
        else:
            # Ensure not to select the positive
            candidates = [j for j in range(current_batch_size) if j != i]
            neg_idxs.append(random.choice(candidates))

    # Convert to tensor indices
    return torch.tensor(neg_idxs, device=device)


def training_loop():
    """Run training loop for siamese model."""
    checkpoint_dir = Path("checkpoints") / config.training.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=config.training.epochs, dynamic_ncols=True, initial=start_epoch
    ) as pbar:
        losses = 0
        stream = iter(loader)
        for epoch in range(start_epoch, config.training.epochs):
            pbar.set_description(f"Epoch: {epoch}")

            if difficulty_scheduler is not None:
                # pooled runs read this in the dataset (stratified pool trees);
                # in-loop runs pass the scheduler value to the generator instead
                dataset.difficulty = difficulty_scheduler.get_difficulty()

            for _ in range(batches_per_epoch):
                (
                    shoeprint_batch,
                    shoeprint_gen_batch,
                    floor_image_batch,
                    shoemark_batch,
                    shoemark_type_mask_batch,
                ) = next(stream)
                # All shoeprints will be used
                with torch.no_grad():
                    shoeprints, shoemarks = augment_batch(
                        shoeprint_batch.to(device),
                        shoeprint_gen_batch.to(device),
                        floor_image_batch.to(device),
                        shoemark_batch.to(device),
                        shoemark_type_mask_batch.to(device),
                        streaming_transform=streaming_transform,
                        device=device,
                        synthetic_pool=synthetic_pool,
                    )

                    # Use EMA for normalisations
                    shoeprints = shoeprint_adaptive_norm(shoeprints, update=True)
                    shoemarks = shoemark_adaptive_norm(shoemarks, update=True)

                # Get embeddings
                shoeprint_embeddings = shoeprint_model(shoeprints)  # [b, d]
                shoemark_embeddings = shoemark_model(shoemarks)  # [b, d]

                # Pairwise distances matrix [N, N]
                dists = torch.cdist(
                    shoeprint_embeddings,
                    shoemark_embeddings,
                    p=config.hyperparameters.p_val,
                )

                # As we don't drop the last batch,
                # this may be less than overall batch size
                neg_idxs = _select_negatives(dists, shoeprint_batch.shape[0])

                # Extract negative embeddings
                negatives = shoemark_embeddings[neg_idxs]

                # Calculate triplet loss
                loss = criterion(shoeprint_embeddings, shoemark_embeddings, negatives)

                shoeprint_optimizer.zero_grad()
                shoemark_optimizer.zero_grad()
                loss.backward()
                shoeprint_optimizer.step()
                shoemark_optimizer.step()

                losses += loss.item()

            shoeprint_scheduler.step()
            shoemark_scheduler.step()

            if difficulty_scheduler is not None:
                difficulty_scheduler.step()

            if epoch % config.training.print_iter == 0 and epoch != 0:
                mean_loss = losses / config.training.print_iter
                line = f"Epoch {epoch} loss: {mean_loss}"
                if difficulty_scheduler is not None:
                    line += f" difficulty: {difficulty_scheduler.get_difficulty()}"
                line += "\n"
                _write_line(line, pbar, checkpoint_dir)
                losses = 0

            if (
                epoch % config.training.val_iter == 0
                or epoch == config.training.epochs - 1
            ) and epoch != 0:
                val = validate(
                    shoeprint_model,
                    shoemark_model,
                    shoeprint_adaptive_norm,
                    shoemark_adaptive_norm,
                    dataset=val_dataset,
                    device=device,
                    p=0,
                    p_val=config.hyperparameters.p_val,
                )
                line = f"Epoch {epoch} S1 validation: = {val}\n"
                _write_line(line, pbar, checkpoint_dir)

                checkpoint = {
                    "shoeprint_model_state_dict": shoeprint_model.state_dict(),
                    "shoemark_model_state_dict": shoemark_model.state_dict(),
                    "shoeprint_optim_state_dict": shoeprint_optimizer.state_dict(),
                    "shoemark_optim_state_dict": shoemark_optimizer.state_dict(),
                    "shoeprint_scheduler_state_dict": shoeprint_scheduler.state_dict(),
                    "shoemark_scheduler_state_dict": shoemark_scheduler.state_dict(),
                    "shoeprint_adaptive_norm_state_dict": shoeprint_adaptive_norm.state_dict(),  # noqa: E501
                    "shoemark_adaptive_norm_state_dict": shoemark_adaptive_norm.state_dict(),  # noqa: E501
                }
                if difficulty_scheduler is not None:
                    checkpoint["difficulty_scheduler_state_dict"] = (
                        difficulty_scheduler.state_dict()
                    )
                torch.save(checkpoint, checkpoint_dir / f"siamese_{epoch}.tar")

            # TODO tidy this up a bit
            if (
                config.training.pre_training.pre_trained
                and config.training.pre_training.frozen
                and epoch != 0
                and (epoch - config.training.pre_training.defrost)
                % config.training.pre_training.epoch_unfreeze
                == 0
            ):
                shoeprint_model.unfreeze_next()
                shoemark_model.unfreeze_next()

            pbar.update()


# * Entry Point

if __name__ == "__main__":
    training_loop()

# Local Variables:
# jinx-local-words: "noqa"
# End:
