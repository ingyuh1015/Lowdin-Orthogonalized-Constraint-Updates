#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_locu_redun1.sh [TASK] [SHORT_TASK]
# Defaults match locu_cfg.py
TASK=${1:-SafetyBallCircle-v0}
SHORT_TASK=${2:-BC-v3}

python3 train_locu_agent.py \
  --task "$TASK" \
  --short_task "$SHORT_TASK" \
  --cost_0_redundant_num 1 \
  --cost_1_redundant_num 1 \
  --cost_2_redundant_num 1 \
  --device cuda
