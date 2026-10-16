from argparse import Action
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from qpsolvers import solve_qp
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn
from torch.distributions import kl_divergence
from itertools import combinations
from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic
from scipy import optimize

class PCRPO(BasePolicy):
    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: List[torch.optim.Optimizer],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),

        target_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        damping_coeff: float = 0.1,
        cg_iters: int = 10,
        max_backtracks: int = 10,
        optim_critic_iters: int = 40,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        cost_limit: Union[List, float] = np.inf,
        total_epochs: Optional[int] = None,
        step_per_epoch: Optional[int] = None,

        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        super().__init__(
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.optim = optim

        self.cost_limit = cost_limit
        self._lambda = gae_lambda
        self._norm_adv = advantage_normalization
        self._max_backtracks = max_backtracks
        self._optim_critic_iters = optim_critic_iters
        self._l2_reg = l2_reg
        self._kl_limit = target_kl
        self._backtrack_coeff = backtrack_coeff
        self._damping_coeff = damping_coeff
        self._cg_iters = cg_iters
        self.max_grad_norm = 40
        self.cost_limit = cost_limit
        self.critics_num = 1 + len(cost_limit)
        self._total_epochs = total_epochs
        self._step_per_epoch = step_per_epoch



    def pre_update_fn(self, stats_train: Dict, **kwarg) -> Any:
        m_cost = stats_train.get("m_cost", [])
        if isinstance(m_cost, torch.Tensor):
            m_cost = m_cost.detach().cpu().numpy()
        m_cost = list(m_cost)
        target_len = self.critics_num - 1
        if len(m_cost) >= target_len:
            m_cost = m_cost[:target_len]
        else:
            m_cost = m_cost + [0.0] * (target_len - len(m_cost))
        self._ave_cost_return = m_cost

    def update_cost_limit(self, cost_limit: float) -> None:
        """Update the cost limit threshold.

        :param float cost_limit: new cost threshold
        """
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

    def _compute_slack_bounds(self, limit: float) -> Tuple[float, float]:
        upper = 1.25 * limit
        lower = 0.75 * limit
        return upper, lower

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        if self._norm_adv:
            adv = batch.advs[..., 0]
            batch.advs[..., 0] = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob, old_mean, old_std = [], [], []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                res = self.forward(minibatch)
                old_log_prob.append(res.dist.log_prob(minibatch.act))
                old_mean.append(res.logits[0, ...])
                old_std.append(res.logits[1, ...])
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.mean_old = torch.cat(old_mean, dim=0)
        batch.std_old = torch.cat(old_std, dim=0)
        return batch


    def critics_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        stats = {}
        total_loss = 0.0

        for i in range(self.critics_num):
            critic = self.critics[i]

            optim = self.optim[i]

            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]

            vf_loss = (ret - value).pow(2).mean()
            for p in critic.parameters():
                vf_loss += self._l2_reg * p.pow(2).sum()
            optim.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=self.max_grad_norm)
            optim.step()

            stats[f"loss/vf{i}"] = vf_loss.item()
            total_loss += vf_loss.item()

        stats["loss/vf_total"] = total_loss
        return torch.tensor(total_loss), stats


    def _get_objective(
        self, logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.exp(logp - logp_old) * adv)

    def _get_cost_surrogate(
        self, logp: torch.Tensor, logp_old: torch.Tensor, cadv: torch.Tensor,i
    ) -> torch.Tensor:
        cost_surrogate = self._ave_cost_return[i-1] + torch.mean(
            torch.exp(logp - logp_old) * cadv
        )
        return cost_surrogate

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """
        Fisher-vector product H v with fixed signature.
        Assumes flat_kl_grad = grad(KL, theta) with create_graph=True.
        """

        kl_v = torch.dot(flat_kl_grad, v.detach())

        hvp = torch.autograd.grad(
            kl_v,
            self.actor.parameters(),
            retain_graph=True
        )

        flat_hvp = torch.cat([g.view(-1) for g in hvp])

        return flat_hvp + self._damping_coeff * v


    def _conjugate_gradients(
        self,
        g: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-8
    ) -> torch.Tensor:
        x = torch.zeros_like(g)
        r, p = g.clone(), g.clone()
        rs_old = torch.sum(r * r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rs_old / torch.sum(p * z)
            x += alpha * p
            r -= alpha * z
            rs_new = torch.sum(r * r)
            if rs_new < residual_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def _get_flat_grad(
        self,
        y: torch.Tensor,
        model: nn.Module,
        retain_graph: bool = False,
        create_graph: bool = False
    ) -> torch.Tensor:
        retain_graph = True if create_graph else retain_graph
        grads = torch.autograd.grad(
            y,
            model.parameters(),
            retain_graph=retain_graph,
            create_graph=create_graph
        )
        return torch.cat([grad.view(-1) for grad in grads])

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        return flat_params

    def _set_from_flat_params(self, model: nn.Module, new_params: torch.Tensor) -> None:
        n = 0
        for param in model.parameters():
            numel = param.numel()
            new_param = new_params[n:n + numel].view(param.size())
            param.data = new_param
            n += numel

    def _project(self, v: torch.Tensor, onto: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = torch.dot(onto, onto)
        if denom.abs() < eps:
            return torch.zeros_like(v)
        return torch.dot(v, onto) / (denom + eps) * onto

    def policy_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        self.actor.train()
        eps = 1e-8
        max_kl = self._kl_limit
        R = float((2 * max_kl) ** 0.5)

        dist = self.forward(minibatch).dist
        logp = dist.log_prob(minibatch.act)
        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))
        kl = kl_divergence(dist_old, dist).mean()
        flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)

        con_vals = []
        cost_surrogates = []
        high_ids: List[int] = []
        mid_ids: List[int] = []
        low_ids: List[int] = []

        for i in range(1, self.critics_num):
            cs = self._get_cost_surrogate(
                logp, minibatch.logp_old, minibatch.advs[..., i], i
            )
            con_val = cs.item()
            con_vals.append(con_val)
            cost_surrogates.append(cs)
            limit = float(self.cost_limit[i - 1])
            upper, lower = self._compute_slack_bounds(limit)
            if con_val > upper:
                high_ids.append(i)
            elif con_val < lower:
                low_ids.append(i)
            else:
                mid_ids.append(i)

        active_constraint = None
        mode = "low"
        reward_obj = None
        H_inv_b = None

        def _select_constraint(c_ids: List[int]) -> int:
            if len(c_ids) == 1:
                return c_ids[0]
            alpha = 5.0
            c_scalars = [
                con_vals[idx - 1] - float(self.cost_limit[idx - 1])
                for idx in c_ids
            ]
            logits = alpha * np.array(c_scalars, dtype=np.float64)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            return c_ids[int(np.random.choice(len(c_ids), p=probs))]

        def _select_max_violation(c_ids: List[int]) -> int:
            if len(c_ids) == 1:
                return c_ids[0]
            best_id = c_ids[0]
            best_val = con_vals[best_id - 1] / float(self.cost_limit[best_id - 1])
            for idx in c_ids[1:]:
                val = con_vals[idx - 1] / float(self.cost_limit[idx - 1])
                if val > best_val:
                    best_val = val
                    best_id = idx
            return best_id


        if high_ids:
            mode = "high"
            active_constraint = _select_max_violation(high_ids)
            cost_obj = cost_surrogates[active_constraint - 1]
            b_raw = self._get_flat_grad(-cost_obj, self.actor, retain_graph=True)
        elif mid_ids:
            mode = "mid"
            active_constraint = _select_max_violation(mid_ids)
            reward_obj = self._get_objective(
                logp, minibatch.logp_old, minibatch.advs[..., 0]
            )
            cost_obj = cost_surrogates[active_constraint - 1]
            g_r = self._get_flat_grad(reward_obj, self.actor, retain_graph=True)

            g_c = self._get_flat_grad(-cost_obj, self.actor, retain_graph=True)


            nat_g_r = self._conjugate_gradients(g_r, flat_kl_grad, nsteps=self._cg_iters)
            nat_g_c = self._conjugate_gradients(g_c, flat_kl_grad, nsteps=self._cg_iters)
            cos_sim = torch.dot(nat_g_r, nat_g_c) / (
                torch.norm(nat_g_r) * torch.norm(nat_g_c) + eps
            )
            if cos_sim <= 0:
                g_r_proj = self._project(nat_g_r, nat_g_c, eps)
                g_c_proj = self._project(nat_g_c, nat_g_r, eps)
                nat_g_r = nat_g_r - g_r_proj
                nat_g_c = nat_g_c - g_c_proj

            H_inv_b = 0.5 * (nat_g_r + nat_g_c)
        else:
            reward_obj = self._get_objective(
                logp, minibatch.logp_old, minibatch.advs[..., 0]
            )
            b_raw = self._get_flat_grad(reward_obj, self.actor, retain_graph=True)
        if H_inv_b is None:
            H_inv_b = self._conjugate_gradients(b_raw, flat_kl_grad)
        approx_b = self._MVP(H_inv_b, flat_kl_grad)
        H_inv_b_dot_b = torch.dot(approx_b, H_inv_b)

        if H_inv_b_dot_b <= 0:
            step_dir = torch.zeros_like(H_inv_b)
            step_size = 0.0
        else:
            step_size = R / H_inv_b_dot_b.sqrt() + eps
            step_dir = step_size * H_inv_b

        init_theta = self._get_flat_params(self.actor).detach().clone()
        self._set_from_flat_params(self.actor, init_theta + step_dir)
        dist = self.forward(minibatch).dist
        logp = dist.log_prob(minibatch.act)
        new_kl = kl_divergence(dist_old, dist).mean().item()

        mode_id = {"low": 0, "mid": 1, "high": 2}.get(mode, -1)
        stats = {
            "kl": new_kl,
            "beta": 1,
            "mode": mode_id,
            "active_constraint": -1 if active_constraint is None else active_constraint - 1,
            "con_vals": con_vals,
            "high_cnt": len(high_ids),
            "mid_cnt": len(mid_ids),
            "low_cnt": len(low_ids),
        }
        return torch.tensor(0.0, device=kl.device), stats


    def learn(
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:

        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                for _ in range(self._optim_critic_iters):
                    loss_vf, stats_critic = self.critics_loss(minibatch)
                loss_actor, stats_actor = self.policy_loss(batch)
                self.logger.store(**stats_actor)
                self.gradient_steps += 1
                self.logger.store(**stats_critic)

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")

    def post_update_fn(self, stats_train: Dict, **kwarg: Any) -> Any:
        return
