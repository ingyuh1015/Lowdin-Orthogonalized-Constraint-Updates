import uuid
from typing import Iterable

import wandb

from fsrl.utils.logger.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    A typical usage example: ::

        config = {...} project = "test_cvpo" group = "SafetyCarCircle-v0" name =
        "default_param" log_dir = "logs"

        logger = WandbLogger(config, project, group, name, log_dir)
        logger.save_config(config)

        agent = CVPOAgent(env, logger=logger) agent.learn(train_envs)

    :param str config: experiment configurations. Default to an empty dict.
    :param str project: W&B project name. Default to "fsrl".
    :param str group: W&B group name. Default to "test".
    :param str name: W&B experiment run name. If None, it will use the current time as
        the name. Default to None.
    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    """

    def __init__(
        self,
        config: dict = {},
        project: str = "fsrl",
        group: str = "test",
        name: str = None,
        log_dir: str = "log",
        log_txt: bool = True
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            id=str(uuid.uuid4()),
            resume="allow",
            config=config,  # type: ignore
        ) if not wandb.run else wandb.run
        # wandb.run.save()

    def write(
        self,
        step: int,
        display: bool = True,
        display_keys: Iterable[str] = None
    ) -> None:
        """Writing data to somewhere and reset the stored data.

        :param int step: the current training step or epochs
        :param bool display: whether print the logged data in terminal, default to False
        :param Iterable[str] display_keys: a list of keys to be printed. If None, print
            all stored keys, default to None.
        """
        self.store(tab="update", env_step=step)
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step: int) -> None:
        """Sending data to wandb without resetting the current stored stats."""
        wandb.log(self.stats_mean, step=step)

    def restore_data(self) -> None:
        """Not implemented yet"""
'''
import uuid
from typing import Iterable, Dict, Any

import wandb
import weave

from fsrl.utils.logger.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """
    WandB + Weave logger for fsrl / GradS.

    - wandb: scalar / step-based logging (loss, reward mean, cost mean, etc.)
    - weave: structured logging (episode, evaluation, constraint, artifacts)
    """

    def __init__(
        self,
        config: dict = {},
        project: str = "fsrl",
        group: str = "test",
        name: str = None,
        log_dir: str = "log",
        log_txt: bool = True,
    ) -> None:
        super().__init__(log_dir, log_txt, name)

        # -------------------------
        # WandB init
        # -------------------------
        self.wandb_run = (
            wandb.init(
                project=project,
                group=group,
                name=name,
                id=str(uuid.uuid4()),
                resume="allow",
                config=config,  # type: ignore
            )
            if not wandb.run
            else wandb.run
        )

        # -------------------------
        # Weave init (auto-linked to wandb run)
        # -------------------------
        self.weave_client = weave.init(project_name=project)

        # optional: store static experiment config in weave
        weave.publish(
            {
                "project": project,
                "group": group,
                "name": name,
                "config": config,
            },
            name="experiment_config",
        )

    # ======================================================================
    # Standard fsrl logging (unchanged behavior)
    # ======================================================================

    def write(
        self,
        step: int,
        display: bool = True,
        display_keys: Iterable[str] = None,
    ) -> None:
        """
        Write data to wandb + txt logger, then reset stats.
        """
        self.store(tab="update", env_step=step)
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step: int) -> None:
        """
        Send scalar statistics to wandb without resetting.
        """
        if len(self.stats_mean) > 0:
            wandb.log(self.stats_mean, step=step)

    def restore_data(self) -> None:
        """Not implemented (kept for interface compatibility)."""
        pass

    # ======================================================================
    # Weave structured logging (NEW)
    # ======================================================================

    def log_episode(
        self,
        episode_id: int,
        reward: float,
        cost: float,
        length: int,
        info: Dict[str, Any] = None,
    ) -> None:
        """
        Log a single episode summary to Weave.

        Call this at:
        - evaluation end
        - training episode end (optional, but do NOT call every step)
        """
        weave.publish(
            {
                "episode_id": episode_id,
                "reward": reward,
                "cost": cost,
                "length": length,
                "info": info or {},
            },
            name="episode",
        )

    def log_evaluation(
        self,
        eval_id: int,
        rewards: Iterable[float],
        costs: Iterable[float],
        lengths: Iterable[int],
    ) -> None:
        """
        Log evaluation results as a structured object.
        """
        weave.publish(
            {
                "eval_id": eval_id,
                "reward_mean": float(sum(rewards) / len(rewards)),
                "cost_mean": float(sum(costs) / len(costs)),
                "reward_dist": list(rewards),
                "cost_dist": list(costs),
                "length_dist": list(lengths),
            },
            name="evaluation",
        )

    def log_constraint_state(
        self,
        step: int,
        constraints: Dict[str, float],
        lagrangian: Dict[str, float],
    ) -> None:
        """
        Log constraint + lagrangian state (Safety RL specific).
        """
        weave.publish(
            {
                "step": step,
                "constraints": constraints,
                "lagrangian": lagrangian,
            },
            name="constraint_state",
        )

    def log_policy_artifact(
        self,
        state_dict: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Log policy checkpoint as a Weave artifact.
        """
        weave.publish(
            state_dict,
            name="policy_checkpoint",
            metadata=metadata,
        )


'''