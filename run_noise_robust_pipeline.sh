#!/usr/bin/env bash
# Chained noise-robust feature-extractor pipeline:
#   1. Wait for an already-running phone_extractor training to finish.
#   2. Export the resulting phone_extractor_en.pt.
#   3. Resume pitch_estimator with --augment for +200k more steps.
#   4. Export the resulting pitch_estimator_v2.pt.
# Safe to run on its own; idempotent if pitch is already exported (it will just
# overwrite with the latest checkpoint).

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

LOG="$REPO/run_noise_robust_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds): noise-robust pipeline start ==="

# 1) Wait for phone trainer ----------------------------------------------------
PHONE_PID="${1:-}"
if [[ -z "$PHONE_PID" ]]; then
    # Auto-discover the running phone trainer
    PHONE_PID="$(pgrep -f phone_extractor_trainer.train | head -n1 || true)"
fi

if [[ -n "$PHONE_PID" ]] && kill -0 "$PHONE_PID" 2>/dev/null; then
    echo "Waiting for phone trainer PID=$PHONE_PID to exit ..."
    # tail --pid waits without polling; falls through when the process exits.
    tail --pid="$PHONE_PID" -f /dev/null
    echo "Phone trainer PID=$PHONE_PID exited."
else
    echo "No running phone trainer found; assuming it already finished."
fi

# 2) Export phone ckpt ---------------------------------------------------------
echo "=== exporting phone_extractor ==="
uv run python -m phone_extractor_trainer.export \
    outputs/phone_extractor_en/checkpoint_latest.pt \
    assets/pretrained/phone_extractor/en_latest.pt

# 3) Resume pitch trainer with augmentation ------------------------------------
echo "=== resuming pitch_estimator with --augment to 500k steps ==="
uv run python -m pitch_estimator_trainer.train \
    --data-dir datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed \
    --out-dir outputs/pitch_estimator_v2 \
    --steps 500000 \
    --batch-size 256 \
    --num-workers 8 \
    --resume \
    --augment

# 4) Export pitch ckpt ---------------------------------------------------------
echo "=== exporting pitch_estimator ==="
uv run python -m pitch_estimator_trainer.export \
    outputs/pitch_estimator_v2/checkpoint_latest.pt \
    assets/pretrained/pitch_estimator/vctk_latest.pt

echo "=== $(date -Iseconds): noise-robust pipeline DONE ==="
echo "Next step: retrain Beatrice on preprocessed/new_lol_data_df with the new feature extractors."
