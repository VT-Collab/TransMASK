import os
from math import dist
from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import pybullet as p


class Opening(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        obj_range: float = 0.3,
        domain_randomize=False,
        random_init: bool = False,
        num_distractors: int = 1
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.domain_randomize = domain_randomize
        self.random_init = random_init
        self.num_distractors = num_distractors
        self.object_size = 0.04,
        self.get_ee_position = get_ee_position
        self.obj_range_low = np.array([0.7, -obj_range / 2, 0.5])
        self.obj_range_high = np.array([0.7, obj_range / 2, 0.5])
        self.ood_obj_range_low = np.array([-0.7, -obj_range, 0.5])
        self.ood_obj_range_high = np.array([0.7, obj_range, 0.5])
        self.object_name = 'ikeasmall'
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.cabinet = self.sim.create_object(
            body_name=self.object_name,
            scale=0.8,
            position=np.random.uniform(self.obj_range_low, self.obj_range_high),
        )
        for i in range(1, 1 + self.num_distractors):
            self.sim.create_box(
                body_name=f"distractor{i}",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=np.zeros(3),
                rgba_color=np.concatenate((np.random.uniform(0, 1, size=3), np.array([1]))),
            )

    def get_obs(self) -> np.ndarray:
        observation = np.array([])
        for i in range(1, 1 + self.num_distractors):
            observation_obj = self.sim.get_base_position(f"distractor{i}")
            rotation_obj = self.sim.get_base_rotation(f"distractor{i}")
            velocity_obj = self.sim.get_base_velocity(f"distractor{i}")
            observation = np.concatenate([observation,
                                          observation_obj,
                                          rotation_obj,
                                          velocity_obj])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        cabinet_state = self.sim.get_object_state(self.object_name, 1)
        return np.concatenate((ee_position, cabinet_state))

    def reset(self) -> None:
        self.obj_position = self._sample_obj()
        self.sim.set_object_state(self.object_name, self.obj_position)
        self.goal = np.concatenate((self.obj_position, np.array([1.])))

        for i in range(1, 1 + self.num_distractors):
            distractor_i = self._sample_obj()
            self.sim.set_base_pose(f"distractor{i}", distractor_i, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_obj(self) -> np.ndarray:
        """Randomize goal."""
        if not self.domain_randomize:
            position = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        else:
            position = self.np_random.uniform(self.ood_obj_range_low, self.ood_obj_range_high)
        return position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
