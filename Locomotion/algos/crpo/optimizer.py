from algos.common.optimizer_base import TROptimizer, flatGrad

from typing import List, Tuple, Callable
import numpy as np
import torch

EPS = 1e-8

class CRPOOptimizer(TROptimizer):
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

        # Pre-allocate tensors on GPU for efficiency (like SRCPO)
        self.B_tensor = torch.zeros(
            (1 + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)
        self.H_inv_B_tensor = torch.zeros(
            (1 + self.n_cons, self.n_params), 
            device=self.device, dtype=torch.float32)




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

    def step(self, get_obj_con_kl: Callable, mu_kl: float = 0.0):
        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)
        R = float((2 * max_kl) ** 0.5)

        # calculate objective, constraints, kl
        objective, constraints, kl = get_obj_con_kl()
        self._computeKLGrad(kl)

        # Compute gradients for objective (index 0)
        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)
        self.B_tensor[0].data.copy_(approx_g_tensor)
        self.H_inv_B_tensor[0].data.copy_(H_inv_g_tensor)

        # Compute gradients for constraints (indices 1 to n_cons)
        con_vals = []
        violated_ids: List[int] = []

        for con_idx in range(self.n_cons):
            con_val = constraints[con_idx].item()
            con_vals.append(con_val)
            
            b_tensor = flatGrad(constraints[con_idx], self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            approx_b_tensor = self._Hx(H_inv_b_tensor)
            self.B_tensor[1 + con_idx].data.copy_(approx_b_tensor)
            self.H_inv_B_tensor[1 + con_idx].data.copy_(H_inv_b_tensor)

            # Track violated constraints (strictly above threshold)
            limit = float(self.con_thresholds[con_idx])
            if con_val > limit:
                violated_ids.append(con_idx)

        # Determine mode and compute update direction
        safety_mode = False
        H_inv_b = None

        if violated_ids:
            # Violation mode: reduce only the most violated constraint
            safety_mode = True
            active_idx = self._select_max_violation(violated_ids, con_vals)
            H_inv_b = -self.H_inv_B_tensor[1 + active_idx].clone()
            # Need to compute approx_b with gradients
            approx_b = self._Hx(H_inv_b)
            H_inv_b_dot_b = torch.dot(H_inv_b, approx_b)

        else:
            # Low mode: purely maximize reward (use pre-computed values)
            H_inv_b = self.H_inv_B_tensor[0]
            approx_b = self.B_tensor[0]
            H_inv_b_dot_b = torch.dot(H_inv_b, approx_b)

        with torch.no_grad():
            if H_inv_b_dot_b <= 0:
                step_dir = torch.zeros_like(H_inv_b)
            else:
                step_size = R / (torch.sqrt(H_inv_b_dot_b) + EPS)
                step_dir = step_size * H_inv_b

            # Apply update
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            self._applyParams(init_theta + step_dir)

        # Re-evaluate after update
        objective, constraints, kl = get_obj_con_kl()
        return objective, constraints, kl, safety_mode
