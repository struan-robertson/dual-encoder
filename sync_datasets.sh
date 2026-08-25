#!/usr/bin/env bash
# Push the datasets the campaign needs to the A40 box (~1.75 GB total).
# Local root is ~/Vault/.../Data; remote root is ~/Datasets (the A40 layout the
# campaign_a40_* queues and gan_pool_permafrost2_b96_a40.toml expect).
# Extra args pass through to rsync (-n for a dry run).
set -euo pipefail

REMOTE="cvip-a40:Datasets/"
ROOT="$HOME/Vault/University/Doctorate/Data"

# /./ anchors rsync --relative: everything right of it is recreated remotely
rsync -avR --progress "$@" \
    "$ROOT/./Siamese/Compiled/Shoeprints/train/" \
    "$ROOT/./Siamese/Compiled/Shoemarks/train/" \
    "$ROOT/./Siamese/Evaluation/no_synth/val/" \
    "$ROOT/./Siamese/Evaluation/no_synth/test/" \
    "$ROOT/./Siamese/Evaluation/wvu/cropped/" \
    "$ROOT/./Flooring/train/" \
    "$REMOTE"
