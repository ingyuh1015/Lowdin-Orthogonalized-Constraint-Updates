from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.env import BaseVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import SDACACTOR
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils import WandbLogger

class SDACACTORAgent(OnpolicyAgent):

    name = "SDACACTORAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,
        seed: int = 10,
        lr: float = 1e-3,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        unbounded: bool = False,
        last_layer_scale: bool = False,

        target_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        damping_coeff: float = 0.1,
        max_backtracks: int = 10,
        optim_critic_iters: int = 10,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,

        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        constraint_num : int = 2

    ) -> None:
        super().__init__()

        self.logger = logger
        self.cost_limit = cost_limit
        self.constraint_num = constraint_num




        seed_all(seed)
        torch.set_num_threads(thread)


        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]

        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = ActorProb(
            net, action_shape, max_action=max_action, unbounded=unbounded, device=device
        ).to(device)

        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device)
            for _ in range(constraint_num + 1)
        ]
        torch.nn.init.constant_(actor.sigma_param, -0.5)
        actor_critic = ActorCritic(actor, critic)

        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        if last_layer_scale:



            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        optim = [
            torch.optim.Adam(c.parameters(), lr=lr)
            for c in critic
        ]


        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = SDACACTOR(
            actor,
            critic,
            optim,
            dist,
            logger=logger,

            target_kl=target_kl,
            backtrack_coeff=backtrack_coeff,
            damping_coeff=damping_coeff,
            max_backtracks=max_backtracks,
            optim_critic_iters=optim_critic_iters,
            l2_reg=l2_reg,
            gae_lambda=gae_lambda,
            advantage_normalization=advantage_normalization,
            cost_limit=self.cost_limit,

            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )


    def process_fn(self, batch, buffer, indices):

        return super().process_fn(batch, buffer, indices)



