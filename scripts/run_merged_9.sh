#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-data}"
OUT_DIR="${2:-results/merged_minimal}"
CFG="${3:-configs/config_merged_minimal.yaml}"

python scripts/train_eval_all.py \
  --base_dir "${BASE_DIR}" \
  --base_out_dir "${OUT_DIR}" \
  --extra_args "--config_path ${CFG} --enable_reboot"

