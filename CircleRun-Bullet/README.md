## Setup

Install the dependencies from this folder.

```bash
cd CircleRun-Bullet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Run LOCU with redundancy 1 or 3:

```bash
cd train
bash run_locu_redun1.sh
# or
bash run_locu_redun3.sh
```
