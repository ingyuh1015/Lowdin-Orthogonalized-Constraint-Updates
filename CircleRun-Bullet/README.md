## Setup

Install the dependencies from this folder.

```bash
cd CircleRun-Bullet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Run training:

```bash
cd train
python train_locu_agent.py \
  --task SafetyBallCircle-v0 \
  --short_task BC-v3 \
  --seed 1 \
  --device cuda:0
```

## Evaluate

From `CircleRun-Bullet/train`, evaluate a saved checkpoint using its training configuration. Replace the paths below with your run's files.

```bash
python eval_locu_agent.py \
  --config /path/to/run/config.yaml \
  --checkpoint /path/to/run/checkpoint/model.pt \
  --episodes 10 \
  --device cuda:0
```

Prints evaluation reward, episode length, and constraint costs.
