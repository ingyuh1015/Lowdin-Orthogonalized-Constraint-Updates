# LOCU MJX Circle Supplementary Code

This folder is a standalone MJX/JAX implementation of the LOCU circle-run experiments. It contains a point-agent circle task, configurable safety constraints, and the algorithm implementations needed to reproduce the MJX circle experiments.

## Contents

- `train/`: command-line training entry point and on-policy training pipeline.
- `algos/`: algorithm implementations.
  - `locu`
  - `on_sdac`
  - `trpo_lag`
  - `ppo_lag`
  - `crpo`
- `config/`: YAML-backed configuration registry.
  - `config/yaml/algorithms/`: algorithm hyperparameters.
  - `config/yaml/tasks/SafetyPointCircle2.yaml`: point-agent circle task.
  - `config/yaml/task_presets/vel_hi08_lo04.yaml`: velocity-limit constraint preset.
- `envs/`: MJX environment, point-agent model, task objects, and wrappers.
- `estimation.py`, `networks.py`: value estimation and neural network modules.
- `sweep/`: lightweight batch launcher example.
- `pyproject.toml`: editable-install package metadata.

Generated outputs, caches, bytecode, local run directories, and W&B run data are intentionally excluded by `.gitignore`.

## License and Third-Party Notice

This supplementary folder is project-specific MJX circle experiment code. Source-origin and redistribution notes for the complete supplementary package are summarized in `../NOTICE.md`.

The MJX circle environment and experiment setup were implemented for this supplementary package, with design inspiration from existing safe-RL environment and algorithm resources:

- Safety Gym: https://github.com/openai/safety-gym
- Safety-Gymnasium: https://github.com/PKU-Alignment/safety-gymnasium
- Bullet-Safety-Gym: https://github.com/SvenGronauer/Bullet-Safety-Gym
- Gradient Shaping for Multi-Constraint Safe Reinforcement Learning (GradS): https://proceedings.mlr.press/v242/yao24a.html

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

