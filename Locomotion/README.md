## Setup

Use Python 3.8 or newer. Install MuJoCo and the system dependencies required by `mujoco_py`.

```bash
cd Locomotion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task YAML files live under `tasks/`; LOCU configuration is in `algos/locu/legged.yaml`.

## Train

```bash
python main.py \
  --task_cfg_path tasks/cassie_goal.yaml \
  --algo_cfg_path algos/locu/legged.yaml \
  --seed 1 \
  --gpu_idx 0
```

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
