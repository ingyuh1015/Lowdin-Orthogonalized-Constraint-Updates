

## Setup

Use Python 3.10 or newer. The package depends on JAX and MuJoCo/MJX.

```bash
cd "LOCU_Supplementary Material /LOCU-mjx-circle"
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Supported Task

Use this task name with the training entry point:

```text
SafetyPointCircle2
```

Optional task preset:

```text
vel_hi08_lo04
```

The preset adds high/low velocity constraints to the base circle task.

## Supported Algorithms

Use one of these algorithm names:

```text
LOCU
OnSDAC
TRPOLag
PPOLag
CRPO
```

Algorithm YAML files live under `config/yaml/algorithms/`.

## Train

Run directly:

```bash
python -m train \
  --algo LOCU \
  --task SafetyPointCircle2 \
  --task-preset vel_hi08_lo04 \
  --redun 2 \
  --seed 0 \
  --backend gpu
```

