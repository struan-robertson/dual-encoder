"""Per-GPU campaign runner: pool generation -> siamese training -> evaluation.

One instance per GPU, driven by a queue file (see campaign_example.toml).
Jobs run sequentially and every stage is idempotent: pool generation resumes
by file existence, training resumes from the newest checkpoint, finished
stages are skipped. Re-running the same queue therefore continues wherever
the previous invocation stopped (or was killed).

    cd ~/Development/Doctorate/siamese && uv run python campaign_runner.py campaign_gpu0.toml

Which generative model each job uses is set manually in the queue file (the
backend, its config, and the exact checkpoint); everything downstream —
the per-run pool size from style_coverage.py (fixed coverage criterion, N
wherever this run's style distribution meets it), pool generation, the
derived siamese config under the locked protocol template, training, and
the test/WVU evaluation of the val-best checkpoint — runs unattended.
Failures abort the job but not the queue, so one broken row cannot idle the
GPU for the rest of an unattended week.

Pools are TRANSIENT: each runner owns one working tree
(<pool_root>/active_gpu<N>) that the next job overwrites — generation is
seeded, so wiping a pool loses nothing that a re-run would not reproduce
bit-identically. Only the per-run pool sizes (<pool_root>/n_styles/) and the
style_coverage analyses persist.

Results accumulate in campaign/results.csv (one row per job: val-best epoch
and S1, test S1/S5/S10, WVU S5/S10).
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent


def sh(cmd, cwd=REPO):
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def set_key(text, section, key, value):
    """Replace `key = ...` inside [section] of a toml document.

    Line-based on purpose: it preserves the template's comments, which carry
    the protocol rationale into every derived config.
    """
    out, current, hits = [], None, 0
    for line in text.splitlines():
        header = re.match(r"\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if header:
            current = header.group(1)
        elif current == section and re.match(r"\s*%s\s*=" % re.escape(key), line):
            line = "%s = %s" % (key, value)
            hits += 1
        out.append(line)
    if hits != 1:
        raise RuntimeError("expected one [%s] %s, found %d" % (section, key, hits))
    return "\n".join(out) + "\n"


def patched_generator_config(job, camp):
    """Copy the generator config with this job's checkpoint pinned."""
    src = Path(job["generator_config"]).expanduser()
    text = src.read_text()
    if job["backend"] in ("gan", "munit"):
        ckpt = Path(job["checkpoint"]).expanduser()
        if not ckpt.exists():
            raise RuntimeError("checkpoint missing: %s" % ckpt)
        if job["backend"] == "gan":
            text = set_key(text, "inference", "checkpoint", '"%s"' % ckpt)
        else:  # munit configs are yaml: pin via a flat checkpoint: key
            text = re.sub(r"^checkpoint:.*\n", "", text, flags=re.MULTILINE)
            text += "checkpoint: %s\n" % ckpt
    else:
        text = set_key(text, "experiment", "name", '"%s"' % job["experiment"])
        text = set_key(text, "test", "epoch", '"%s"' % job["epoch"])
    # written next to the original so its relative paths (checkpoints_dir
    # etc.) resolve against the same directory
    patched = src.with_name("campaign_%s%s" % (job["name"], src.suffix))
    patched.write_text(text)
    return patched


# the coverage criterion is the fixed part of the protocol; N is where each
# run's own style distribution meets it (trajectory-noise floor for the
# stochastic bridge, 95% of achievable reduction for the deterministic GAN)
N_PATTERNS = {
    "unsb": r"N at <=1\.1x floor: (\d+)\s*$",
    "gan": r"N at 95% coverage: (\d+)\s*$",
    "munit": r"N at 95% coverage: (\d+)\s*$",  # deterministic given a style, like the gan
}


def stage_n_styles(job, camp, patched):
    """Per-run pool size from style_coverage.py (cached under pool_root).

    The cache lives outside the working pool tree — pools are overwritten
    job-to-job, but the per-run Ns are kept (the methods section reports them).
    """
    if "n_styles" in job:
        return job["n_styles"]
    n_file = Path(camp["pool_root"]).expanduser() / "n_styles" / ("%s.txt" % job["name"])
    if n_file.exists():
        return int(n_file.read_text().split()[0])

    cmd = [
        "uv", "run", "python",
        Path(camp["unsb_repo"]).expanduser() / "style_coverage.py",
        "--backend", job["backend"],
        "--generator-config", patched,
        "--tag", job["name"],
    ]
    if camp.get("print_dir"):
        cmd += ["--print-dir", Path(camp["print_dir"]).expanduser()]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run(
        [str(c) for c in cmd], cwd=REPO, capture_output=True, text=True
    )
    if result.returncode:
        # capture_output hides the traceback, which is the only useful part of a
        # failure here; surface it before raising
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print(result.stdout, flush=True)
    m = re.search(N_PATTERNS[job["backend"]], result.stdout, re.MULTILINE)
    if not m:
        raise RuntimeError(
            "style_coverage gave no usable recommendation for %s "
            "(criterion not reached? raise M/POOL_SIZES)" % job["name"]
        )
    n = int(m.group(1))
    n_file.parent.mkdir(parents=True, exist_ok=True)
    n_file.write_text("%d\n" % n)
    return n


def pool_workdir(camp):
    """One reusable pool tree per runner instance, overwritten job-to-job.

    Pools are transient (regeneration is seeded and cheap next to keeping
    ~15 GB per job around); per-GPU naming keeps two runners sharing a
    pool_root (the A40 box) out of each other's tree.
    """
    return Path(camp["pool_root"]).expanduser() / ("active_gpu%s" % camp.get("gpu", 0))


def stage_pool(job, camp, patched, n_styles):
    """Generate the job's pool into the working tree (resumes by existence).

    A marker file records the full pool identity (job, pinned checkpoint, N).
    Identical marker -> resume the partial pool; anything else -> DELETE the
    whole tree first and regenerate. Never overwrite in place: pools differ in
    size and per-file resume would both keep stale images beyond the new N and
    skip indices that belong to a different generator.
    """
    if job["backend"] in ("gan", "munit"):
        ident = str(Path(job["checkpoint"]).expanduser())
    else:
        ident = "%s:%s" % (job["experiment"], job["epoch"])
    # pool_as lets several jobs (e.g. the augmentation ablations of one
    # generator row) share a pool: their stamp matches the original job's, so
    # the tree is resumed by existence rather than wiped and regenerated
    stamp = "%s %s N=%d\n" % (job.get("pool_as", job["name"]), ident, n_styles)

    work = pool_workdir(camp)
    pool = work / "train"
    marker = work / "job.txt"
    if pool.exists() and (not marker.exists() or marker.read_text() != stamp):
        print("[%s] deleting pool left by: %s" % (
            job["name"], marker.read_text().strip() if marker.exists() else "unknown"
        ), flush=True)
        shutil.rmtree(pool)
    work.mkdir(parents=True, exist_ok=True)
    marker.write_text(stamp)
    cmd = [
        "uv", "run", "python",
        Path(camp["unsb_repo"]).expanduser() / "generate_pool.py",
        "--backend", job["backend"],
        "--n-styles", n_styles,
        "--out-dir", pool,
        "--generator-config", patched,
    ]
    if job.get("frozen_style"):
        cmd += ["--frozen-style"]
    if camp.get("print_dir"):
        cmd += ["--print-dir", Path(camp["print_dir"]).expanduser()]
    sh(cmd)
    return pool


def stage_replica(job, camp, pool):
    """Copy the pool to fast storage when the queue asks for it (HDD hosts)."""
    if not camp.get("train_root"):
        return pool
    replica = Path(camp["train_root"]).expanduser() / ("active_gpu%s" % camp.get("gpu", 0)) / "train"
    replica.parent.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-a", "--delete", str(pool) + "/", str(replica) + "/"])
    return replica


def toml_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"%s"' % value


def derived_config(job, camp, train_dir):
    """Instantiate the locked-protocol template for this job."""
    text = (REPO / camp["template"]).read_text()
    text = set_key(text, "training", "name", '"%s"' % job["name"])
    text = set_key(text, "training", "gpu_number", "0")  # runner masks the GPU
    text = set_key(
        text, "data.streaming", "synthetic_shoemark_data_dir", '"%s"' % train_dir
    )
    # per-job deviations from the template ([job.set]: "section.key" = value) —
    # the ablation campaigns flip augmentation stages and seeds through this
    for dotted, value in job.get("set", {}).items():
        section, key = dotted.rsplit(".", 1)
        text = set_key(text, section, key, toml_literal(value))
    ckpt_dir = REPO / "checkpoints" / job["name"]
    tars = sorted(
        ckpt_dir.glob("siamese_*.tar"), key=lambda p: int(p.stem.split("_")[1])
    )
    if tars:
        text = set_key(text, "training", "resume_checkpoint", '"%s"' % tars[-1])
    cfg = REPO / "campaign" / ("%s.toml" % job["name"])
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text)
    return cfg


def training_complete(job, camp):
    epochs = tomllib.loads((REPO / camp["template"]).read_text())["training"]["epochs"]
    log = REPO / "checkpoints" / job["name"] / "siamese.log"
    return log.exists() and ("Epoch %d S1" % (epochs - 1)) in log.read_text()


def stage_train(job, camp, cfg):
    if training_complete(job, camp):
        print("[%s] training already complete" % job["name"], flush=True)
        return
    sh(["uv", "run", "python", "src/training.py", cfg.relative_to(REPO)])


def stage_eval(job, cfg):
    """Evaluate the val-best checkpoint on test and WVU; return the results row."""
    ckpt_dir = REPO / "checkpoints" / job["name"]
    scores = re.findall(
        r"Epoch (\d+) S1 validation: = ([0-9.]+)", (ckpt_dir / "siamese.log").read_text()
    )
    best_epoch, best_s1 = max(
        ((int(e), float(s)) for e, s in scores), key=lambda t: t[1]
    )
    # evaluation.py scores every .tar in a directory, so the protocol's
    # "selected checkpoint only" reporting is enforced by linking just one
    best_dir = ckpt_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for stale in best_dir.glob("siamese_*.tar"):
        stale.unlink()
    link = best_dir / ("siamese_%d.tar" % best_epoch)
    link.symlink_to(ckpt_dir / link.name)

    row = {"job": job["name"], "best_epoch": best_epoch, "val_S1": best_s1}
    for ds, ps in (("test", (1, 5, 10)), ("wvu", (5, 10))):
        for p in ps:
            log = best_dir / ("eval_%s_p%d.log" % (ds, p))
            if not log.exists():
                sh([
                    "uv", "run", "python", "src/evaluation.py", best_dir,
                    cfg.relative_to(REPO), "--dataset", ds, "-p", p,
                    "--log-name", log.name,
                ])
            row["%s_S%d" % (ds, p)] = float(
                re.search(r"= ([0-9.]+)", log.read_text()).group(1)
            )
    return row


def record(row):
    out = REPO / "campaign" / "results.csv"
    fields = ["job", "best_epoch", "val_S1", "test_S1", "test_S5", "test_S10",
              "wvu_S5", "wvu_S10"]
    rows = []
    if out.exists():
        with out.open(newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["job"] != row["job"]]
    rows.append(row)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("[%s] %s" % (row["job"], row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path, help="campaign queue toml")
    args = parser.parse_args()

    with args.queue.open("rb") as fh:
        queue = tomllib.load(fh)
    camp = queue["campaign"]

    # one runner instance per GPU: mask the device here so every stage —
    # generate_pool's cuda:0, the derived config's gpu_number 0 — lands on it
    gpu = str(camp.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HIP_VISIBLE_DEVICES"] = gpu

    failed = []
    for job in queue["job"]:
        print("\n=== %s (%s) ===" % (job["name"], job["backend"]), flush=True)
        try:
            if job["backend"] == "none":
                # pool-less baseline (real-only or augmentation-only rows):
                # no synthetic tree, straight to training
                cfg = derived_config(job, camp, "")
                stage_train(job, camp, cfg)
            elif training_complete(job, camp):
                # done jobs need no pool — don't overwrite the current one
                print("[%s] training already complete; skipping pool" % job["name"],
                      flush=True)
                cfg = derived_config(job, camp, pool_workdir(camp) / "train")
            else:
                patched = patched_generator_config(job, camp)
                n_styles = stage_n_styles(job, camp, patched)
                print("[%s] pool size N = %d" % (job["name"], n_styles), flush=True)
                pool = stage_pool(job, camp, patched, n_styles)
                train_dir = stage_replica(job, camp, pool)
                cfg = derived_config(job, camp, train_dir)
                stage_train(job, camp, cfg)
            record(stage_eval(job, cfg))
        except Exception:
            traceback.print_exc()
            failed.append(job["name"])

    if failed:
        raise SystemExit("failed jobs: %s" % ", ".join(failed))
    print("\nqueue complete", flush=True)


if __name__ == "__main__":
    main()
