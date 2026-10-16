from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from tasks.cassie.env_v0 import Env as CassieTrackingEnv
from tasks.goal_hazard_base import GoalHazardMixin
from tasks.goal_hazard_utils import hazard_contact_cost


class Env(GoalHazardMixin, CassieTrackingEnv):
    def __init__(
        self,
        *,
        map_extents=(-3.5, -3.5, 3.5, 3.5),
        goal_radius=0.50,
        hazard_radius=0.35,
        hazard_count=8,
        agent_keepout=0.70,
        placement_clearance=0.25,
        lidar_num_bins=16,
        lidar_max_dist=6.0,
        goal_progress_weight=1.0,
        goal_bonus=0.2,
        speed_limit=1.0,
        energy_limit=13.0,
        **kwargs,
    ) -> None:
        self.speed_limit = float(speed_limit)
        self.energy_limit = float(energy_limit)
        self._init_goal_hazard(
            base_state_keys=[
                'gravity_vector',
                'base_lin_vel',
                'base_ang_vel',
                'joint_pos_list',
                'joint_vel_list',
                'phase_list',
                'joint_pos_history',
                'joint_vel_history',
                'joint_target_history',
                'contact_list',
                'base_height',
            ],
            map_extents=map_extents,
            goal_radius=goal_radius,
            hazard_radius=hazard_radius,
            hazard_count=hazard_count,
            agent_keepout=agent_keepout,
            placement_clearance=placement_clearance,
            lidar_num_bins=lidar_num_bins,
            lidar_max_dist=lidar_max_dist,
            goal_progress_weight=goal_progress_weight,
            goal_bonus=goal_bonus,
        )
        super().__init__(**kwargs)
        self.state_keys = self._goal_hazard_state_keys()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._sample_goal_hazard_layout()
        raw_state = self._getRawState()
        self.last_goal_distance = float(raw_state['goal_distance'][0])
        return self._convertState(raw_state), {}

    def _step(self, action):
        joint_targets = self.generator.getJointTargets(self.cur_step * self.env_dt)
        joint_targets = np.clip(action + joint_targets, self.lower_limits, self.upper_limits)
        self.action = self.action * self.action_weight + joint_targets * (1.0 - self.action_weight)

        self.power_consumption = 0.0
        for _ in range(self.n_substeps):
            d_error = self._getJointVelList()
            p_error = self.action - self._getJointPosList()
            torque = self.Kps * p_error - self.Kds * d_error
            torque[5:7] = -torque[5:7]
            self.sim.data.ctrl[:] = torque
            self.sim.step()
            self.power_consumption += np.mean(np.abs(self.sim.data.actuator_force * d_error))
        self.power_consumption /= self.n_substeps

        self.joint_pos_history.append(p_error)
        self.joint_vel_history.append(d_error)
        self.joint_target_history.append(self.action)

        raw_state = self._getRawState()
        current_distance = float(raw_state['goal_distance'][0])
        reached_goal = self._goal_hazard_reached(current_distance)
        rewards = self._getRewards(raw_state)
        costs = self._getCosts(raw_state)

        if reached_goal:
            self.goals_reached += 1
            agent_xy, _ = self._goal_hazard_agent_pose()
            self._respawn_goal(agent_xy)
            raw_state = self._getRawState()
        else:
            self.last_goal_distance = current_distance

        state = self._convertState(raw_state)
        reward = np.concatenate([rewards, costs])

        body_angle = raw_state['gravity_vector'][2] / np.linalg.norm(raw_state['gravity_vector'])
        truncate = self.cur_step >= self.max_episode_length
        terminate = body_angle >= 0
        info = {
            'goal_achieved': reached_goal,
            'goals_reached': self.goals_reached,
        }
        return state, reward, terminate, truncate, info

    def _getRawState(self):
        return self._augment_goal_hazard_state(super()._getRawState())

    def _convertState(self, state):
        return self._convert_goal_hazard_state(state)

    def _getRewards(self, state):
        current_distance = float(state['goal_distance'][0])
        reached_goal = current_distance <= self.goal_radius
        return self._goal_hazard_reward(current_distance, reached_goal, 0.0)

    def _getCosts(self, state):
        gv = state['gravity_vector']
        x_angle = gv[2] / np.linalg.norm(gv)
        h = state['base_height'][0]
        speed = abs(state['base_lin_vel'][0])
        contact_pts = self._floor_contact_points_xy()

        # foot_timing (no loosening possible, binary)
        ft_cost = 0.0
        for leg_idx in range(self.num_legs):
            cos_phase, sin_phase = state['phase_list'][2*leg_idx:2*(leg_idx+1)]
            if sin_phase < 0.0:
                ft_cost += 1.0 if state['contact_list'][leg_idx] else 0.0
            else:
                ft_cost += 0.0 if state['contact_list'][leg_idx] else 1.0
        ft_cost /= self.num_legs

        costs = np.array([
            # body_angle: 15°, 20°, 25°
            1.0 if x_angle > -np.cos(np.radians(15.0)) else 0.0,
            1.0 if x_angle > -np.cos(np.radians(20.0)) else 0.0,
            1.0 if x_angle > -np.cos(np.radians(25.0)) else 0.0,
            # com_height: 0.7, 0.6, 0.5
            1.0 if h < 0.7 else 0.0,
            1.0 if h < 0.6 else 0.0,
            1.0 if h < 0.5 else 0.0,
            # foot_timing: same x3
            ft_cost, ft_cost, ft_cost,
            # speed: 1.0, 1.2, 1.5
            1.0 if speed > 1.0 else 0.0,
            1.0 if speed > 1.2 else 0.0,
            1.0 if speed > 1.5 else 0.0,
            # hazard_contact: margin 0.0, -0.05, -0.1 (negative = looser)
            hazard_contact_cost(contact_pts, self.hazard_positions, self.hazard_radius, contact_margin=0.0),
            hazard_contact_cost(contact_pts, self.hazard_positions, self.hazard_radius, contact_margin=-0.05),
            hazard_contact_cost(contact_pts, self.hazard_positions, self.hazard_radius, contact_margin=-0.1),
            # energy: 13.0, 15.0, 18.0
            1.0 if self.power_consumption > 13.0 else 0.0,
            1.0 if self.power_consumption > 15.0 else 0.0,
            1.0 if self.power_consumption > 18.0 else 0.0,
        ], dtype=np.float32)
        return costs

    def _goal_hazard_agent_pose(self):
        base_pos = self.sim.data.get_site_xpos('imu')
        base_mat = self.sim.data.get_site_xmat('imu')
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        return np.asarray(base_pos[:2], dtype=np.float64), float(yaw_angle)
