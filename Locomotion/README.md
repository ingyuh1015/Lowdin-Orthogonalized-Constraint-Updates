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
