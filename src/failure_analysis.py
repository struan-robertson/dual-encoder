"""Failure-case analysis of a trained dual-encoder checkpoint.

Mirrors the template-matching chapter's analysis (sec:template|failures) for
the learned retrieval model: every query shoemark whose correct gallery
shoeprint ranks outside the top p percent is a failure case. For each one the
query, the correct shoeprint, and the incorrect top-1 shoeprint are copied
into the output directory (the thesis figure triptych), and failures.csv
records the rank and the distance gap between the correct print and the
top-1 print — the distance-space analogue of the chapter's similarity gap
(positive gap: the model actively preferred the wrong print).

    uv run python src/failure_analysis.py \
        checkpoints/unsb_no_noise_1/best/siamese_4000.tar --dataset test

Passing --mark <file> instead inspects a single query in depth, writing every
gallery print the model ranked above the correct one.

Writes to failure_analysis/<run>_<dataset>/ in the repo root.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import torch

from siamese.config import load_config
from siamese.datasets import LabeledCombinedDataset, get_id
from siamese.model import ImpressionEncoder
from siamese.streaming import IMAGENET_MEAN, IMAGENET_STD, AdaptiveNormalisation


def write_chain(mark_name, dataset, mark_embeddings, gallery, class_idxs,
                print_files, distance_norm, out_dir):
    """Dump the ranked gallery chain for one query mark.

    Everything the model preferred over the correct print is written out, not
    just the top-1: `rank<r>_<class>` images from rank 0 down to the correct
    print, and chain.csv with every gallery print's distance and its margin
    over the correct print. This is what shows whether a failure is one
    confusable neighbour or a whole family of them, and how the band the
    failure lives in compares with the spread of the gallery as a whole.
    """
    shoe_id = get_id(Path(mark_name))
    if shoe_id not in mark_embeddings:
        raise SystemExit("no query mark of class %s in this dataset" % shoe_id)
    marks = dataset.shoemark_classes[shoe_id]
    matches = [i for i, m in enumerate(marks) if m.name == mark_name]
    if not matches:
        raise SystemExit("%s not among the marks of %s: %s"
                         % (mark_name, shoe_id, [m.name for m in marks]))
    mark_idx = matches[0]

    dists = torch.cdist(mark_embeddings[shoe_id][mark_idx : mark_idx + 1],
                        gallery, p=distance_norm)[0]
    order = torch.argsort(dists)
    correct_idx = class_idxs.index(shoe_id)
    rank = int((order == correct_idx).nonzero()[0, 0])
    dist_correct = float(dists[correct_idx])

    mark = marks[mark_idx]
    shutil.copy(mark, out_dir / ("shoemark%s" % mark.suffix))
    with (out_dir / "chain.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rank", "class", "distance", "margin_over_correct", "correct",
        ])
        writer.writeheader()
        for r in range(len(gallery)):
            idx = int(order[r])
            print_file = print_files[class_idxs[idx]]
            if r <= rank:  # only the prints preferred over the correct one
                shutil.copy(print_file, out_dir / ("rank%d_%s%s" % (
                    r, class_idxs[idx], print_file.suffix)))
            writer.writerow({
                "rank": r,
                "class": class_idxs[idx],
                "distance": float(dists[idx]),
                "margin_over_correct": dist_correct - float(dists[idx]),
                "correct": idx == correct_idx,
            })

    print("%s (%s): correct print at rank %d of %d, distance %.4f; "
          "%d intervening prints written to %s"
          % (mark_name, shoe_id, rank, len(gallery), dist_correct, rank, out_dir))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", nargs="?", default="config.toml")
    parser.add_argument("--dataset", choices=["wvu", "test", "val"], default="test")
    parser.add_argument("-p", type=int, default=5, help="Top percentage rank cutoff")
    parser.add_argument("--mark", default=None,
                        help="file name of a single query mark; writes the whole "
                             "ranked chain up to the correct print instead of the sweep")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: failure_analysis/<run>_<dataset>/")
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
    dataset = LabeledCombinedDataset(data_dir / "Shoeprints", data_dir / "Shoemarks")

    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parent.name
    if run == "best":  # checkpoints/<run>/best/<tar>
        run = checkpoint.parent.parent.name
    suffix = "" if args.mark is None else "_" + Path(args.mark).stem
    out_dir = args.out or Path(__file__).resolve().parent.parent / "failure_analysis" / (
        "%s_%s%s" % (run, args.dataset, suffix)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(checkpoint, map_location=device)
    embedding_size = state["shoeprint_model_state_dict"]["model.fc.weight"].shape[0]
    shoeprint_model = ImpressionEncoder(embedding_size=embedding_size).to(device).eval()
    shoemark_model = ImpressionEncoder(embedding_size=embedding_size).to(device).eval()
    shoeprint_model.load_state_dict(state["shoeprint_model_state_dict"])
    shoemark_model.load_state_dict(state["shoemark_model_state_dict"])
    shoeprint_norm = AdaptiveNormalisation(IMAGENET_MEAN, IMAGENET_STD, device=device)
    shoemark_norm = AdaptiveNormalisation(IMAGENET_MEAN, IMAGENET_STD, device=device)
    shoeprint_norm.load_state_dict(state["shoeprint_adaptive_norm_state_dict"])
    shoemark_norm.load_state_dict(state["shoemark_adaptive_norm_state_dict"])

    # embed the whole gallery and every query, keeping file paths for the copies
    print_embeddings, print_files, mark_embeddings = {}, {}, {}
    for shoeprint_class, (shoeprint, shoemarks) in dataset:
        x = shoeprint_norm(shoeprint.to(device), update=False)
        print_embeddings[shoeprint_class] = shoeprint_model(x).squeeze(0).cpu()
        if len(shoemarks) > 0:
            mark_embeddings[shoeprint_class] = torch.cat([
                shoemark_model(shoemark_norm(m.to(device), update=False))
                for m in shoemarks
            ]).cpu()
    for f in dataset.shoeprint_files:
        print_files[get_id(f)] = f

    class_idxs = list(print_embeddings.keys())
    gallery = torch.stack(list(print_embeddings.values()))
    distance_norm = config.hyperparameters.distance_norm
    k = math.ceil(len(gallery) * args.p / 100)

    if args.mark is not None:
        write_chain(args.mark, dataset, mark_embeddings, gallery, class_idxs,
                    print_files, distance_norm, out_dir)
        return

    print("%d gallery prints, %d query classes, failure = rank >= %d (top %d%%)"
          % (len(gallery), len(mark_embeddings), k, args.p))

    failures, n_marks = [], 0
    for shoe_id, embeddings in mark_embeddings.items():
        dists = torch.cdist(embeddings, gallery, p=distance_norm)
        order = torch.argsort(dists)
        correct_idx = class_idxs.index(shoe_id)
        ranks = (order == correct_idx).nonzero()[:, 1]
        n_marks += len(ranks)
        for mark_idx, rank in enumerate(ranks.tolist()):
            if rank < k:
                continue
            top1_idx = int(order[mark_idx, 0])
            failures.append({
                "class": shoe_id,
                "mark_file": dataset.shoemark_classes[shoe_id][mark_idx].name,
                "rank": rank,
                "gallery": len(gallery),
                "dist_correct": float(dists[mark_idx, correct_idx]),
                "dist_top1": float(dists[mark_idx, top1_idx]),
                "gap": float(dists[mark_idx, correct_idx] - dists[mark_idx, top1_idx]),
                "top1_class": class_idxs[top1_idx],
                "_mark_path": dataset.shoemark_classes[shoe_id][mark_idx],
            })

    failures.sort(key=lambda f: -f["rank"])
    with (out_dir / "failures.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "n", "class", "mark_file", "rank", "gallery",
            "dist_correct", "dist_top1", "gap", "top1_class",
        ])
        writer.writeheader()
        for n, f in enumerate(failures, 1):
            mark = f.pop("_mark_path")
            shutil.copy(mark, out_dir / ("%d_shoemark%s" % (n, mark.suffix)))
            true = print_files[f["class"]]
            shutil.copy(true, out_dir / ("%d_true_shoeprint%s" % (n, true.suffix)))
            false = print_files[f["top1_class"]]
            shutil.copy(false, out_dir / ("%d_false_shoeprint%s" % (n, false.suffix)))
            writer.writerow({"n": n, **f})

    print("%d/%d marks failed; images and failures.csv in %s"
          % (len(failures), n_marks, out_dir))


if __name__ == "__main__":
    main()
