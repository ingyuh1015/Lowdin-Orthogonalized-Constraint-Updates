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

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic
from scipy import optimize

EPS = 1e-8

class SDACACTOR(BasePolicy):


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
        max_backtracks: int = 10,
        optim_critic_iters: int = 40,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        cost_limit: Union[List, float] = np.inf,

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


        self.cost_limit = cost_limit
        self.critics_num = 1 + len(cost_limit)
        self.con_zeta = np.min(cost_limit)



    def pre_update_fn(self, stats_train: Dict, **kwarg) -> Any:
        m_cost = stats_train.get("m_cost", [])
        if isinstance(m_cost, torch.Tensor):
            m_cost = m_cost.detach().cpu().numpy()
        m_cost = list(m_cost)
        target_len = max(0, self.critics_num - 1)
        if len(m_cost) >= target_len:
            m_cost = m_cost[:target_len]
        else:
            m_cost = m_cost + [0.0] * (target_len - len(m_cost))
        self._ave_cost_return = m_cost

    def update_cost_limit(self, cost_limit: float) -> None:
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

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
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
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

    def policy_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        self.actor.train()
        eps = 1e-8
        max_kl = self._kl_limit


        dist = self.forward(minibatch).dist
        ent = dist.entropy().mean()
        logp = dist.log_prob(minibatch.act)
        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))
        kl = kl_divergence(dist_old, dist).mean()

        flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)


        objective = self._get_objective(logp, minibatch.logp_old, minibatch.advs[..., 0])
        g_tensor = self._get_flat_grad(objective, self.actor, retain_graph=True)
        H_inv_g_tensor = self._conjugate_gradients(g_tensor, flat_kl_grad)
        approx_g_tensor = self._MVP(H_inv_g_tensor, flat_kl_grad)
        g_H_inv_g_tensor = torch.dot(approx_g_tensor, H_inv_g_tensor)

        con_vals, b_tensors, H_inv_b_tensors, c_scalars, max_c_scalars = [], [], [], [], []
        safety_mode = False

        for i in range(1, self.critics_num):
            cs = self._get_cost_surrogate(logp, minibatch.logp_old, minibatch.advs[..., i], i)
            con_val = cs.item()
            b_raw = self._get_flat_grad(cs, self.actor, retain_graph=True)
            H_inv_b_tensor = self._conjugate_gradients(b_raw, flat_kl_grad)
            approx_b_tensor = self._MVP(H_inv_b_tensor, flat_kl_grad)

            b_H_inv_b_tensor = torch.dot(approx_b_tensor, H_inv_b_tensor)
            max_c_scalar = np.sqrt(np.clip(2.0 * max_kl * b_H_inv_b_tensor.item(), 0.0, np.inf))

            c_scalar = min(max_c_scalar, con_val - self.cost_limit[i - 1])

            if con_val > self.cost_limit[i - 1]:
                safety_mode = True

            con_vals.append(con_val)
            b_tensors.append(approx_b_tensor)
            H_inv_b_tensors.append(H_inv_b_tensor)
            c_scalars.append(c_scalar)
            max_c_scalars.append(max_c_scalar)

        with torch.no_grad():

            B_tensor = torch.stack(b_tensors).T
            H_inv_B_tensor = torch.stack(H_inv_b_tensors).T
            S_tensor = B_tensor.T @ H_inv_B_tensor
            r_tensor = approx_g_tensor @ H_inv_B_tensor


            S_mat = S_tensor.detach().cpu().numpy().astype(np.float64)
            S_mat = 0.5 * (S_mat + S_mat.T)

            min_eig_val = min(0.0, np.min(np.linalg.eigvals(S_mat)))
            S_mat += np.eye(S_mat.shape[0]) * (-min_eig_val + EPS)

            r_vector = r_tensor.detach().cpu().numpy()
            g_H_inv_g_scalar = g_H_inv_g_tensor.detach().cpu().numpy()
            c_vector = np.array(c_scalars)


            const_lam_vector = solve_qp(P=S_mat, q=-c_vector, lb=np.zeros_like(c_vector), solver="osqp")
            if const_lam_vector is None: const_lam_vector = np.zeros_like(c_vector)
            approx_kl = 0.5 * np.dot(const_lam_vector, S_mat @ const_lam_vector)
            scaling = 1.0 if approx_kl <= max_kl else np.sqrt(max_kl / approx_kl)

            if approx_kl / max_kl - 1.0 > -0.001:
                print(f"[QCQP] --- Recovery Mode ---")
                print(f"[QCQP] initial const_lam: {const_lam_vector}")
                for c_idx in range(len(c_vector)):
                    c_vector[c_idx] = min(max_c_scalars[c_idx], con_vals[c_idx] - self.cost_limit[c_idx] + self.con_zeta)
                const_lam_vector = solve_qp(P=S_mat, q=-c_vector, lb=np.zeros_like(c_vector), solver="osqp")
                if const_lam_vector is None: const_lam_vector = np.zeros_like(c_vector)

                approx_kl = 0.5 * np.dot(const_lam_vector, S_mat @ const_lam_vector)
                scaling = np.sqrt(max_kl / approx_kl)
                print(f"[QCQP] recovery const_lam: {const_lam_vector}")
                print(f"[QCQP] approx_kl: {approx_kl:.6f}, scaling: {scaling:.6f}")

                lam_tensor = torch.tensor(const_lam_vector, device=g_tensor.device, dtype=torch.float32)
                delta_theta = -scaling * H_inv_B_tensor @ lam_tensor
            else:
                print(f"[QCQP] --- Optimization Mode ---")
                def dual_func(x, gHg, r, S, c, kl_limit):
                    lam, nu = x[:-1], x[-1]

                    obj = (gHg - 2.0*np.dot(r, lam) + np.dot(lam, S@lam))/(2.0*nu + eps) - np.dot(lam, c) + nu*kl_limit
                    return obj
                def dualJac(x, g_H_inv_g, r_vector, S_mat, c_vector, max_kl):
                    lam_vector = x[:-1]
                    nu_scalar = x[-1]
                    jacobian = np.zeros_like(x)
                    jacobian[:-1] = (S_mat@lam_vector - r_vector)/(nu_scalar + EPS) - c_vector
                    jacobian[-1] = max_kl - (g_H_inv_g - 2.0*np.dot(r_vector, lam_vector) + np.dot(lam_vector, S_mat@lam_vector))/(2.0*(nu_scalar**2) + EPS)
                    return jacobian
                x0 = np.ones(len(c_vector) + 1)
                bounds = optimize.Bounds(np.zeros(len(c_vector) + 1), np.ones(len(c_vector) + 1)*np.inf)
                res = optimize.minimize(dual_func, x0, method='trust-constr', jac=dualJac,
                    args=(g_H_inv_g_scalar, r_vector, S_mat, c_vector, max_kl),
                    bounds=bounds, options={'disp': True, 'maxiter': 200})

                if res.success:
                    lam_vector, nu_scalar = res.x[:-1], res.x[-1]
                    print(f"[QCQP] Optimization Success! lam: {lam_vector}, nu: {nu_scalar:.6f}")
                    lam_tensor = torch.tensor(lam_vector, device=g_tensor.device, dtype=torch.float32)
                    delta_theta = (H_inv_g_tensor - H_inv_B_tensor @ lam_tensor) / (nu_scalar + eps)
                else:
                    print(f"[QCQP] Optimization Failed. fallback to const_lam: {const_lam_vector}")
                    lam_tensor = torch.tensor(const_lam_vector, device=g_tensor.device, dtype=torch.float32)
                    delta_theta = -scaling * H_inv_B_tensor @ lam_tensor


            init_theta = self._get_flat_params(self.actor)
            self._set_from_flat_params(self.actor, init_theta + delta_theta)

        return objective, {"loss/kl": kl.item(), "Update/safety_mode": int(safety_mode)}


    def learn(
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:

        loss_actor, stats_actor = self.policy_loss(batch)
        self.logger.store(**stats_actor)
        for minibatch in batch.split(batch_size, merge_last=True):
            for _ in range(self._optim_critic_iters):
                loss_vf, stats_critic = self.critics_loss(minibatch)

            self.gradient_steps += 1
            self.logger.store(**stats_critic)

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")
