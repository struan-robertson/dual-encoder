#!/usr/bin/env bash
# Push the GAN checkpoints the A40 campaign queues need (UNSB models already
# exist on both machines). The file list is parsed from campaign_a40_*.toml so
# re-pinning a checkpoint just needs a re-run; commented-out jobs (only_style,
# which lives on the A40 anyway) are ignored. ~6 GB for the 12 pinned tars.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="cvip-a40:one-to-many-gan/checkpoints/"
ROOT="$HOME/Extra/Doctorate/Checkpoints/one_to_many_gan"

srcs=()
while read -r rel; do
    if [[ -f "$ROOT/$rel" ]]; then
        srcs+=("$ROOT/./$rel")   # /./ anchors rsync --relative at <run>/models/<step>.tar
    else
        echo "WARNING: skipping missing $ROOT/$rel" >&2
    fi
done < <(grep -h '^checkpoint = ' campaign_a40_*.toml \
         | sed -E 's/^checkpoint = "(.*)"$/\1/; s|.*/checkpoints/||' | sort -u)

echo "syncing ${#srcs[@]} checkpoints -> $REMOTE"
rsync -avR --progress "$@" "${srcs[@]}" "$REMOTE"
