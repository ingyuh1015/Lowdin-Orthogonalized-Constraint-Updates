#!/usr/bin/env bash
set -euo pipefail

TASK_CFG="${1:-tasks/cassie_goal.yaml}"
ALGO_CFG="${2:-algos/locu/legged.yaml}"
SEED="${3:-1}"
GPU_IDX="${4:-0}"

python main.py \
  --task_cfg_path "$TASK_CFG" \
  --algo_cfg_path "$ALGO_CFG" \
  --seed "$SEED" \
  --gpu_idx "$GPU_IDX"
