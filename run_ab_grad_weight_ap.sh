#!/usr/bin/env bash
# A/B test: grad_weight_ap=0 (upstream default after PR #1) vs grad_weight_ap=100 (legacy).
# Both runs use the v2 noise-robust extractors pinned in assets/configs/ab_ap*_v2.json:
#   phone_extractor/en_nr_targetmix02_300k.pt, pitch_estimator/vctk_nr_300k.pt
# and the DeepFilterNet-denoised dataset (preprocessed/new_lol_data_df).
#
# Sequential because the 12 GB GPU cannot fit two Beatrice trainers simultaneously.
# Each run: 60k steps @ ~2.9 it/s ≈ 5.75 h. Total wall-clock ≈ 11.5 h.
#
# Outputs:
#   outputs/ab_ap0_v2/    (training logs, checkpoints, samples)
#   outputs/ab_ap100_v2/
#   logs/ab_ap0_v2.log
#   logs/ab_ap100_v2.log
#
# After both finish, listen to the generated samples in
# outputs/ab_ap<N>_v2/test/ to judge the buzz/clarity tradeoff.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=.venv/bin/python
DATA=preprocessed/new_lol_data_df

run_one() {
    local name=$1
    local out=outputs/${name}
    local log=logs/${name}.log
    local cfg=${out}/config.json

    if [ -f "${out}/checkpoint_latest.pt" ]; then
        echo "[$(date +%H:%M:%S)] ${name}: resuming from existing checkpoint"
        ${PY} -m beatrice_trainer -d ${DATA} -o ${out} -c ${cfg} -r 2>&1 | tee -a ${log}
    else
        echo "[$(date +%H:%M:%S)] ${name}: starting fresh"
        ${PY} -m beatrice_trainer -d ${DATA} -o ${out} -c ${cfg} 2>&1 | tee ${log}
    fi
    echo "[$(date +%H:%M:%S)] ${name}: done"
}

echo "=== A/B starting at $(date) ==="
run_one ab_ap0_v2
run_one ab_ap100_v2
echo "=== A/B done at $(date) ==="
