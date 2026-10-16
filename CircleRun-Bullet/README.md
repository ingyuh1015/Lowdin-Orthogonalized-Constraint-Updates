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
