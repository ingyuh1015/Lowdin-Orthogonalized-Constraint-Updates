from __future__ import annotations

import numpy as np

from tasks.goal_hazard_utils import (
    goal_progress_reward,
    hazard_contact_cost,
    pseudo_lidar,
    sample_entities,
)


class GoalHazardMixin:
    def _init_goal_hazard(
        self,
        *,
        base_state_keys: list[str],
        map_extents: tuple[float, float, float, float],
        goal_radius: float,
        hazard_radius: float,
        hazard_count: int,
        agent_keepout: float,
        placement_clearance: float,
        lidar_num_bins: int,
        lidar_max_dist: float,
        goal_progress_weight: float,
        goal_bonus: float,
    ) -> None:
        self._goal_hazard_base_state_keys = list(base_state_keys)
        self.map_extents = tuple(map_extents)
        self.goal_radius = float(goal_radius)
        self.hazard_radius = float(hazard_radius)
        self.hazard_count = int(hazard_count)
        self.agent_keepout = float(agent_keepout)
        self.placement_clearance = float(placement_clearance)
        self.lidar_num_bins = int(lidar_num_bins)
        self.lidar_max_dist = float(lidar_max_dist)
        self.goal_progress_weight = float(goal_progress_weight)
        self.goal_bonus = float(goal_bonus)
        self.goal_xy = np.array([self.map_extents[2] - self.goal_radius, 0.0], dtype=np.float64)
        self.hazard_positions = np.zeros((self.hazard_count, 2), dtype=np.float64)
        self.last_goal_distance = 0.0
        self.goals_reached = 0

    def _goal_hazard_state_keys(self) -> list[str]:
        return self._goal_hazard_base_state_keys + ['goal_lidar', 'hazards_lidar']

    def _sample_goal_hazard_layout(self) -> None:
        agent_xy, _ = self._goal_hazard_agent_pose()
        self.hazard_positions = sample_entities(
            self.np_random,
            count=self.hazard_count,
            extents=self.map_extents,
            entity_radius=self.hazard_radius,
            clearance=self.placement_clearance,
            avoid_positions=np.array([agent_xy]),
            avoid_radii=np.array([self.agent_keepout]),
        )
        self._respawn_goal(agent_xy)
        self.goals_reached = 0

    def _respawn_goal(self, agent_xy: np.ndarray | None = None) -> None:
        if agent_xy is None:
            agent_xy, _ = self._goal_hazard_agent_pose()
        avoid_positions = np.concatenate([np.array([agent_xy]), self.hazard_positions], axis=0)
        avoid_radii = np.concatenate(
            [
                np.array([self.agent_keepout], dtype=np.float64),
                np.full(self.hazard_count, self.hazard_radius, dtype=np.float64),
            ],
            axis=0,
        )
        self.goal_xy = sample_entities(
            self.np_random,
            count=1,
            extents=self.map_extents,
            entity_radius=self.goal_radius,
            clearance=self.placement_clearance,
            avoid_positions=avoid_positions,
            avoid_radii=avoid_radii,
        )[0]
        self.last_goal_distance = float(np.linalg.norm(agent_xy - self.goal_xy))

    def _augment_goal_hazard_state(self, state: dict) -> dict:
        if '_agent_xy' in state and '_agent_yaw' in state:
            agent_xy = np.asarray(state['_agent_xy'], dtype=np.float64)
            agent_yaw = float(np.asarray(state['_agent_yaw']).reshape(-1)[0])
        else:
            agent_xy, agent_yaw = self._goal_hazard_agent_pose()
        state['goal_lidar'] = pseudo_lidar(
            agent_xy=agent_xy,
            agent_yaw=agent_yaw,
            positions=np.array([self.goal_xy]),
            num_bins=self.lidar_num_bins,
            max_dist=self.lidar_max_dist,
            alias=True,
        )
        state['hazards_lidar'] = pseudo_lidar(
            agent_xy=agent_xy,
            agent_yaw=agent_yaw,
            positions=self.hazard_positions,
            num_bins=self.lidar_num_bins,
            max_dist=self.lidar_max_dist,
            alias=True,
        )
        state['goal_distance'] = np.array([np.linalg.norm(agent_xy - self.goal_xy)], dtype=np.float32)
        return state

    def _convert_goal_hazard_state(self, state: dict) -> np.ndarray:
        flatten_state = [np.asarray(state[key]).reshape(-1) for key in self._goal_hazard_state_keys()]
        return np.asarray(np.concatenate(flatten_state), dtype=np.float32)

    def _goal_hazard_reward(self, current_distance: float, reached_goal: bool, energy_penalty: float) -> np.ndarray:
        reward = goal_progress_reward(
            previous_distance=self.last_goal_distance,
            current_distance=current_distance,
            reached_goal=reached_goal,
            progress_weight=self.goal_progress_weight,
            goal_bonus=self.goal_bonus,
        )
        return np.array([reward], dtype=np.float32)

    def _goal_hazard_cost(self) -> float:
        return hazard_contact_cost(
            self._floor_contact_points_xy(),
            self.hazard_positions,
            self.hazard_radius,
        )

    def _goal_hazard_reached(self, current_distance: float) -> bool:
        return current_distance <= self.goal_radius

    def _floor_contact_points_xy(self) -> np.ndarray:
        points = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.geom1 == contact.geom2:
                continue
            if contact.geom1 == self.geom_floor_id or contact.geom2 == self.geom_floor_id:
                points.append(np.array(contact.pos[:2], dtype=np.float64))
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    def _goal_hazard_agent_pose(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError
