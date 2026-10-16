## Setup

Use Python 3.10 or newer. The package depends on JAX and MuJoCo/MJX.

```bash
cd CircleRun-Mjx
python -m venv .venv
source .venv/bin/activate
pip install -e .
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
