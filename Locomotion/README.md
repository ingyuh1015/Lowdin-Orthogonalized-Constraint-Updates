# LOCU Locomotion Supplementary Code

This folder is a standalone copy of the locomotion experiments from the Spectral-Risk-Constrained-RL repository. It contains the requested Cassie/Laikago locomotion tasks, Cassie goal-hazard task variants, and the full algorithm implementation used by those tasks.

## Contents

- `main.py`: train, evaluate, and render entry point.
- `algos/`: full algorithm package copied from the repository.
  - `locu`
  - `sdac`
  - `trpol`
  - `crpo`
  - `common`
- `tasks/`: selected locomotion environments and configs.
  - `cassie.yaml`, `tasks/cassie/`
  - `laikago.yaml`, `tasks/laikago/`
  - `cassie_goal.yaml`, `tasks/cassie_goal/`
  - `cassie_goal_redun.yaml`, `tasks/cassie_goal_redun/`
  - `goal_hazard_base.py`, `goal_hazard_utils.py`, `register.py`
- `utils/`: vectorized environment, logging, and seed/color helpers.
- `scripts/`: convenience launch scripts.
- `requirements.txt`: Python package summary.

Unrelated point/car/Safety-Gym tasks, old outputs, debug scripts, and Python cache files are intentionally excluded.

## License and Third-Party Notice

This supplementary folder includes code adapted from the Spectral-Risk-Constrained-RL repository. The copied source is distributed under the MIT License notice preserved in `LICENSE`. Source-origin details for the complete supplementary package are summarized in `../NOTICE.md`.

Some locomotion model assets under `tasks/cassie/models/` and `tasks/laikago/models/` may originate from their corresponding robot simulation models. Keep those files with the supplementary package when submitting or redistributing it.

## Setup

Use Python 3.8 or newer. The locomotion environments use `mujoco_py`, so MuJoCo and its system dependencies must be installed before running rollouts.

```bash
cd "LOCU_Supplementary Material /LOCU-Locomotion"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `mujoco_py` cannot find MuJoCo, configure the standard MuJoCo environment variables for your machine before running training.

## Supported Tasks

Use one of these task configuration files:

```text
tasks/cassie.yaml
tasks/laikago.yaml
tasks/cassie_goal.yaml
tasks/cassie_goal_redun.yaml
```

Use one of these algorithm configuration files:

```text
algos/locu/legged.yaml
algos/sdac/legged.yaml
algos/trpol/legged.yaml
algos/crpo/legged.yaml
```

The Cassie goal-hazard task YAML files override constraint thresholds for each algorithm when needed.

## Train

Run directly:

```bash
python main.py \
  --task_cfg_path tasks/cassie_goal.yaml \
  --algo_cfg_path algos/locu/legged.yaml \
  --seed 1 \
  --gpu_idx 0
```

Or use the helper script:

```bash
bash scripts/train_locomotion.sh tasks/cassie_goal.yaml algos/locu/legged.yaml 1 0
```

Add `--wandb` to the direct command if you want Weights & Biases logging. Without `--wandb`, logs and checkpoints are written locally under `results/`.

## Evaluate

Evaluate a saved checkpoint:

```bash
python main.py \
  --task_cfg_path tasks/cassie_goal.yaml \
  --algo_cfg_path algos/locu/legged.yaml \
  --seed 1 \
  --gpu_idx 0 \
  --model_num 5000000 \
  --eval
```

Or use:

```bash
bash scripts/eval_locomotion.sh tasks/cassie_goal.yaml algos/locu/legged.yaml 1 0 5000000
```

Evaluation writes `result.pkl` under the matching `results/.../eval/` directory.

## Render / Test

Render rollouts from a saved checkpoint:

```bash
python main.py \
  --task_cfg_path tasks/cassie.yaml \
  --algo_cfg_path algos/locu/legged.yaml \
  --seed 1 \
  --gpu_idx 0 \
  --model_num 5000000 \
  --test
```

## Notes

- Use `--device cpu` if CUDA is unavailable.
- The default checkpoint directory is `results/{task}_{algo}/seed_{seed}`.
- The C++ helper modules in `algos/*/cpp_modules/` are included. If a shared object is incompatible with your platform, rebuild it from its directory with `bash compile.sh`.
