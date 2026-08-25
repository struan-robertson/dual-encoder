"""Check everything a campaign queue needs before launching it.

    uv run python campaign_preflight.py campaign_santa_kl.toml

Verifies the queue's paths, the template's data directories, the pinned
checkpoints, the helper scripts in the UNSB repo, and that torch actually sees
a GPU of the expected vendor. Exits non-zero if anything is missing.
"""

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent
ok = True


def check(label, path, want_files=False):
    global ok
    p = Path(path).expanduser()
    if not p.exists():
        print("  MISSING  %-28s %s" % (label, p))
        ok = False
        return
    if want_files:
        n = sum(1 for _ in p.rglob("*.png")) + sum(1 for _ in p.rglob("*.jpg"))
        if n == 0:
            print("  EMPTY    %-28s %s" % (label, p))
            ok = False
            return
        print("  ok       %-28s %s (%d images)" % (label, p, n))
    else:
        print("  ok       %-28s %s" % (label, p))


def main():
    queue = tomllib.loads(Path(sys.argv[1]).read_text())
    camp, jobs = queue["campaign"], queue["job"]

    print("campaign")
    tmpl = REPO / camp["template"]
    check("template", tmpl)
    check("unsb_repo", camp["unsb_repo"])
    for script in ("generate_pool.py", "style_coverage.py"):
        check(script, Path(camp["unsb_repo"]).expanduser() / script)
    check("print_dir", camp["print_dir"], want_files=True)
    check("pool_root parent", Path(camp["pool_root"]).expanduser().parent)

    if tmpl.exists():
        t = tomllib.loads(tmpl.read_text())
        print("\ntemplate data")
        for k in ("val_dir", "test_dir", "wvu_data_dir"):
            check(k, t["data"][k], want_files=True)
        for k in ("floor_image_data_dir", "shoeprint_data_dir", "shoemark_data_dir"):
            check(k, t["data"]["streaming"][k], want_files=True)
        print("  note     epochs=%d batch=%d permafrost=%d"
              % (t["training"]["epochs"], t["hyperparameters"]["batch_size"],
                 t["training"]["pre_training"]["permafrost"]))

    print("\njobs")
    for j in jobs:
        print(" %s (%s)" % (j["name"], j["backend"]))
        check("generator_config", j["generator_config"])
        if j["backend"] == "gan":
            check("checkpoint", j["checkpoint"])
        else:
            check("experiment", Path(camp["unsb_repo"]).expanduser()
                  / "checkpoints" / j["experiment"] / ("%s_net_G.pth" % j["epoch"]))

    print("\ncpu")
    import os
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    quota = None
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            quota = int(raw[0]) / int(raw[1])
    except OSError:
        pass
    print("  os.cpu_count()=%s  sched_getaffinity=%s  cgroup quota=%s"
          % (os.cpu_count(), aff, ("%.1f" % quota) if quota else "none"))
    budget = int(quota) if quota else (aff or os.cpu_count() or 1)
    workers = 16  # training.py:226, pool mode
    threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    print("  DataLoader workers=%d (training.py), OMP_NUM_THREADS=%s"
          % (workers, os.environ.get("OMP_NUM_THREADS", "unset")))
    if threads == 0:
        print("  WARNING  OMP_NUM_THREADS unset: each of the %d workers may start"
              " one OpenMP thread per visible core, oversubscribing %d usable cores"
              % (workers, budget))
    elif workers * threads > budget:
        print("  WARNING  workers x OMP_NUM_THREADS = %d exceeds %d usable cores"
              % (workers * threads, budget))

    print("\nruntime")
    try:
        import torch
        print("  torch %s  cuda=%s  hip=%s  devices=%d"
              % (torch.__version__, torch.version.cuda, torch.version.hip,
                 torch.cuda.device_count()))
        if not torch.cuda.is_available():
            print("  NO GPU VISIBLE")
            globals()["ok"] = False
    except Exception as e:  # noqa: BLE001
        print("  torch import failed: %s" % e)
        globals()["ok"] = False

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE -- fix before launching"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
