#!/usr/bin/env bash
set -euo pipefail

ALGO="${1:-LOCU}"
REDUN="${2:-2}"
SEED="${3:-0}"

python -m train \
  --algo "$ALGO" \
  --task SafetyPointCircle2 \
  --task-preset vel_hi08_lo04 \
  --redun "$REDUN" \
  --seed "$SEED"
