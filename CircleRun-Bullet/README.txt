LOCU training quick run

This folder has simple scripts to run LOCU with different redundancy settings.

Prerequisites
- Python environment with project dependencies installed. (with requirements.txt)
- Run from the train/ directory.


Usage

Redundancy = 1:
  ./run_locu_redun1.sh

Redundancy = 3:
  ./run_locu_redun3.sh

Optional arguments
You can pass SHORT_TASK as positional args:

  ./run_locu_redun1.sh SafetyBallCircle-v0 BC-v3
  ./run_locu_redun3.sh SafetyDroneCircle-v0 DC-v3



Notes
- The scripts call train_locu_agent.py in this directory.
- If you want to run from another directory, cd train first.
- License/source-origin notices are summarized in ../NOTICE.md. Keep that file and the package-local LICENSE files when submitting or redistributing this folder.
