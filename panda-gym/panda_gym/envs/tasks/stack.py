from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Stack(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.2,
        obj_xy_range=0.2,
        domain_randomize: bool = False,
        random_init: bool = False,
        num_distractors: int = 1
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.domain_randomize = domain_randomize
        self.random_init = random_init
        self.num_distractors = num_distractors
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.ood_goal_range_low = np.array([-goal_xy_range, -goal_xy_range, 0])
        self.ood_goal_range_high = np.array([goal_xy_range, goal_xy_range, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.ood_obj_range_low = np.array([-obj_xy_range, -obj_xy_range, 0])
        self.ood_obj_range_high = np.array([obj_xy_range, obj_xy_range, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        for i in range(1, 1 + self.num_distractors):
            self.sim.create_box(
                body_name=f"distractor{i}",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=np.random.uniform(self.obj_range_low, self.obj_range_high),
                rgba_color=np.concatenate((np.random.uniform(0, 1, size=3), np.array([1.]))),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))

        observation = np.concatenate([object1_position,
                                      object1_rotation,
                                      object1_velocity,
                                      object2_position,
                                      object2_rotation,
                                      object2_velocity])
        
        for i in range(1, 1 + self.num_distractors):
            distractor_position = np.array(self.sim.get_base_position(f"distractor{i}"))
            distractor_rotation = np.array(self.sim.get_base_rotation(f"distractor{i}"))
            distractor_velocity = np.array(self.sim.get_base_velocity(f"distractor{i}"))
            observation = np.concatenate([observation,
                                          distractor_position,
                                          distractor_rotation,
                                          distractor_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position, object2_position = self._sample_objects()
        self.object1_position = object1_position
        self.object2_position = object2_position
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))

        for i in range(1, 1 + self.num_distractors):
            distractor_position = self._sample_distractor()
            self.sim.set_base_pose(f"distractor{i}", distractor_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center
        if not self.domain_randomize:
            noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        else:
            noise = self.np_random.uniform(self.ood_goal_range_low, self.ood_goal_range_high)
        goal1 += noise
        goal2 += noise
        return np.concatenate((goal1, goal2))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, self.object_size / 2])
        while True:
            if not self.domain_randomize:
                noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            else:
                noise1 = self.np_random.uniform(self.ood_obj_range_low, self.ood_obj_range_high)
                noise2 = self.np_random.uniform(self.ood_obj_range_low, self.ood_obj_range_high)
            object1_position += noise1
            object2_position += noise2
            if (np.linalg.norm(object1_position - self.goal[:3]) > 0.1 and np.linalg.norm(object2_position - self.goal[3:]) > 0.1) and np.linalg.norm(object1_position - object2_position) > 0.1:
                break
        return object1_position, object2_position

    def _sample_distractor(self) -> np.ndarray:
        """Randomize start position of object."""
        distractor_position = np.array([-0.1, 0., self.object_size / 2])
        while True:
            noise = self.np_random.uniform([-0.3, -0.3, 0], [0.3, 0.3, 0])
            distractor_position += noise
            if np.linalg.norm(distractor_position - self.goal[:3]) > 0.2 and np.linalg.norm(distractor_position - self.goal[3:]) > 0.2 and np.linalg.norm(distractor_position - self.object1_position) > 0.2 and np.linalg.norm(distractor_position - self.object2_position) > 0.2:
                break
        return distractor_position
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array((d > self.distance_threshold), dtype=np.float32)
        else:
            return -d.astype(np.float32)
