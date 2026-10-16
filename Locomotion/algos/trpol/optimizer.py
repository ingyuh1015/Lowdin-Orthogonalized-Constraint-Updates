from algos.common.optimizer_base import TROptimizer, flatGrad

from typing import List, Tuple, Callable
import torch

EPS = 1e-8

class TRPOLOptimizer(TROptimizer):
    def __init__(
        self, device,
        actor, 
        damping_coeff, 
        num_conjugate, 
        line_decay, 
        max_kl, 
        con_thresholds, 
        con_zeta,
        con_entropy) -> None:

        super().__init__(device, actor, damping_coeff, num_conjugate, line_decay, max_kl)
        self.con_thresholds = con_thresholds
        self.con_zeta = con_zeta
        self.con_entropy = con_entropy
        self.n_cons = len(self.con_thresholds)
        self.lag_pid = (0.05, 0.0005, 0.1)
        self.rescaling = True
        self.gradient_shaping = True
        self.sim_threshold = 0.8
        self.conflicting_threshold = 0.999
        self.activated_index = -1

        self._lagrangian = torch.zeros(
            (self.n_cons,), device=self.device, dtype=torch.float32)
        self._lag_int = torch.zeros(
            (self.n_cons,), device=self.device, dtype=torch.float32)
        self._lag_prev_err = torch.zeros(
            (self.n_cons,), device=self.device, dtype=torch.float32)

        # Pre-allocate tensors on GPU for efficiency (like SRCPO)
        self.B_tensor = torch.zeros(
            (1 + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)
        self.H_inv_B_tensor = torch.zeros(
            (1 + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)

    def _compute_slack_bounds(self, limit: float) -> Tuple[float, float]:
        """Compute upper and lower slack bounds for a constraint."""
        upper = limit * 1
        lower = limit * 0.75
        return upper, lower

    def _project(self, a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        """Project vector a onto vector b."""
        b_norm_sq = torch.dot(b, b) + eps
        proj_scalar = torch.dot(a, b) / b_norm_sq
        return proj_scalar * b

    def _select_max_violation(self, c_ids: List[int], con_vals: List[float]) -> int:
        """Select the constraint with the highest relative violation."""
        if len(c_ids) == 1:
            return c_ids[0]
        best_id = c_ids[0]
        best_val = con_vals[best_id] / (abs(self.con_thresholds[best_id]) + EPS)
        for idx in c_ids[1:]:
            val = con_vals[idx] / (abs(self.con_thresholds[idx]) + EPS)
            if val > best_val:
                best_val = val
                best_id = idx
        return best_id

    def _update_lagrangian(self, constraints: torch.Tensor) -> torch.Tensor:
        n_cons = min(self.n_cons, constraints.numel())
        if n_cons == 0:
            return self._lagrangian[:0]
        thresholds = torch.as_tensor(
            self.con_thresholds, device=constraints.device, dtype=constraints.dtype
        )[:n_cons]
        errors = constraints[:n_cons] - thresholds
        with torch.no_grad():
            kp, ki, kd = self.lag_pid
            self._lag_int[:n_cons] = self._lag_int[:n_cons] + errors
            delta = errors - self._lag_prev_err[:n_cons]
            updated = self._lagrangian[:n_cons] + kp * errors + ki * self._lag_int[:n_cons] + kd * delta
            self._lagrangian[:n_cons] = torch.clamp(updated, min=0.0)
            self._lag_prev_err[:n_cons] = errors
        return self._lagrangian[:n_cons]

    def _select_active_constraint_gs(
        self, grad_list: List[torch.Tensor], lagrangian: torch.Tensor
    ) -> int:
        if not grad_list:
            return -1
        if len(grad_list) == 1:
            return 0

        norms = torch.as_tensor(
            self.con_thresholds, device=lagrangian.device, dtype=lagrangian.dtype
        )
        n_cons = min(norms.numel(), lagrangian.numel(), len(grad_list))
        norms = norms[:n_cons]
        lag = lagrangian[:n_cons]

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        index_list = [0]
        for i in range(1, n_cons):
            sims = []
            for j in index_list:
                sims.append(float(cos(grad_list[i], grad_list[j]).item()))
            if max(sims) < self.sim_threshold and min(sims) > -self.conflicting_threshold:
                index_list.append(i)

        if len(index_list) == 1:
            return index_list[0]

        norm_lag = lag / (norms + EPS)
        probs = torch.softmax(norm_lag[index_list], dim=0)
        choice = torch.multinomial(probs, 1).item()
        return index_list[choice]

    def step(self, get_obj_con_kl: Callable, mu_kl: float = 0.0):
        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)

        # calculate objective, constraints, kl
        objective, constraints, kl = get_obj_con_kl()
        if isinstance(constraints, (list, tuple)):
            constraints = torch.stack(list(constraints))
        if constraints.dim() == 0:
            constraints = constraints.unsqueeze(0)
        self._computeKLGrad(kl)

        n_cons = min(self.n_cons, constraints.numel())
        if n_cons == 0:
            return objective, constraints, kl, False

        lagrangian = self._update_lagrangian(constraints)

        # Compute gradients for constraints (indices 1 to n_cons)
        con_vals = []
        b_tensors = []
        n_cons = min(self.n_cons, constraints.numel())
        for con_idx in range(n_cons):
            con_val = constraints[con_idx].item()
            con_vals.append(con_val)

            b_tensor = flatGrad(constraints[con_idx], self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            approx_b_tensor = self._Hx(H_inv_b_tensor)
            self.B_tensor[1 + con_idx].data.copy_(approx_b_tensor)
            self.H_inv_B_tensor[1 + con_idx].data.copy_(H_inv_b_tensor)
            b_tensors.append(b_tensor)

        safety_mode = any(
            con_vals[i] > self.con_thresholds[i] for i in range(n_cons)
        )

        if self.gradient_shaping:
            active_idx = self._select_active_constraint_gs(b_tensors, lagrangian)
            self.activated_index = active_idx
            lag_vec = torch.zeros_like(lagrangian)
            if active_idx >= 0:
                lag_vec[active_idx] = lagrangian[active_idx]
            rescaling = 1.0 / (lagrangian[active_idx] + 1.0) if self.rescaling and active_idx >= 0 else 1.0
        else:
            active_idx = -1
            lag_vec = lagrangian
            rescaling = 1.0 / (lagrangian.sum() + 1.0) if self.rescaling else 1.0

        lagrangian_objective = rescaling * (objective - torch.dot(lag_vec, constraints[:n_cons]))
        loss_actor = -lagrangian_objective

        g_tensor = flatGrad(lagrangian_objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)
        self.B_tensor[0].data.copy_(approx_g_tensor)
        self.H_inv_B_tensor[0].data.copy_(H_inv_g_tensor)
        g_H_inv_g = torch.dot(approx_g_tensor, H_inv_g_tensor)

        if float(g_H_inv_g.item()) <= 0:
            step_dir = torch.zeros_like(H_inv_g_tensor)
        else:
            step_size = torch.sqrt(2.0 * max_kl / (g_H_inv_g + EPS))
            step_dir = step_size * H_inv_g_tensor

        with torch.no_grad():
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            self._applyParams(init_theta + step_dir)
            new_objective, new_constraints, new_kl = get_obj_con_kl()
            if isinstance(new_constraints, (list, tuple)):
                new_constraints = torch.stack(list(new_constraints))
            if new_constraints.dim() == 0:
                new_constraints = new_constraints.unsqueeze(0)

        return new_objective, new_constraints, new_kl, safety_mode
