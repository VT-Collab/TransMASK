from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pybullet as p
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from panda_gym.pybullet import PyBullet


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_space: spaces.Space,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)
    
    def get_link_orientation(self, link: int) -> np.ndarray:
        """Returns the orientation of a link as (rx, ry, rz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_orientation(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: PyBulletRobot,
        task: Task,
        render_width: int = 720,
        render_height: int = 480,
        image_obs_width: int = 256,
        image_obs_height: int = 256,
        return_image_obs: bool = False,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        domain_randomize: bool = False,
        random_init: bool = False
    ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."

        self.semantic_name_to_id = {
            "panda": 0,
            "table.urdf": 1,
            "table": 1,
            "object": 2,
            "target": 3,
            "object1": 4,
            "target1": 5,
            "object2": 6,
            "target2": 7,
            "distractor1": 8,
            "distractor2": 9,
            "distractor3": 10,
            "distractor4": 11,
            "distractor5": 12,
            "plane": 255,   # optional / ignore
        }
        self.semantic_id_to_name = {v: k for k, v in self.semantic_name_to_id.items()}

        self.sim = robot.sim
        self.render_mode = self.sim.render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task
        self.domain_randomize = domain_randomize
        self.random_init = random_init
        self.return_image_obs = return_image_obs
        self.image_obs_height = image_obs_height
        self.image_obs_width = image_obs_width
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        if not return_image_obs:
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                    desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                    achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                )
            )
        else:
            n = self.sim.physics_client.getNumBodies()
            feature_dim = n * 4
            self.observation_space = spaces.Dict(
                dict(
                    static_image=spaces.Box(0, 255, shape=(image_obs_width, image_obs_height), dtype=np.uint8),
                    ee_image=spaces.Box(0, 255, shape=(image_obs_width, image_obs_height), dtype=np.uint8),
                    static_seg=spaces.Box(0., 1., shape=(feature_dim,), dtype=np.float32),
                    ee_seg=spaces.Box(0., 1., shape=(feature_dim,), dtype=np.float32),
                    observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                    desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                    achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                )
            )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _build_body_id_mapping(self) -> None:
        """Build bodyUniqueId -> semantic_id mapping (must be called after reset)."""
        self.body_uid_to_semantic = {}

        pc = self.sim.physics_client

        num_bodies = pc.getNumBodies()
        for i in range(num_bodies):
            body_uid = pc.getBodyUniqueId(i)

            body_info = pc.getBodyInfo(body_uid)
            body_name = body_info[1].decode("utf-8") if body_info[1] else ""

            if body_name == "":
                uid = pc.getUserDataId(body_uid, "body_name")
                if uid >= 0:
                    body_name = pc.getUserData(uid)
                    if isinstance(body_name, bytes):
                        body_name = body_name.decode("utf-8")
            if not body_name:
                raise RuntimeError(
                    f"Body {body_uid} has no URDF name and no userData 'body_name'"
                )
            if body_name not in self.semantic_name_to_id.keys():
                raise ValueError(f"Unknown body name '{body_name}' — add to semantic_name_to_id")

            self.body_uid_to_semantic[body_uid] = self.semantic_name_to_id[body_name]

    def get_segmentation_feature(self, image_seg: np.ndarray) -> np.ndarray:
        """
        Convert PyBullet segmentation mask to feature vector.
        """
        feature = np.array([])
        for body_uid, semantic_name in self.body_uid_to_semantic.items():
            ys, xs = np.where(image_seg == body_uid)
            ys = ys.astype(np.float32) / self.image_obs_height
            xs = xs.astype(np.float32) / self.image_obs_width
            if len(xs) == 0:
                feature = np.concatenate((feature, np.zeros(4)))
            else:
                feature = np.concatenate((feature, np.array([xs.min(), ys.min(), xs.max(), ys.max()])))
        return feature
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs, image_obs = self.robot.get_obs()  # robot state
        robot_obs = robot_obs.astype(np.float32)
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        if self.return_image_obs:
            return {
                "static_image": image_obs['static_image'],
                "ee_image": image_obs['ee_image'],
                "static_seg": self.get_segmentation_feature(image_obs['static_seg']),
                "ee_seg": self.get_segmentation_feature(image_obs['ee_seg']),
                "observation": observation,
                "achieved_goal": achieved_goal,
                "desired_goal": self.task.get_goal().astype(np.float32),
            }
        else:
            return {
                "observation": observation,
                "achieved_goal": achieved_goal,
                "desired_goal": self.task.get_goal().astype(np.float32),
            }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random = self.np_random
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        self._build_body_id_mapping()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), observation["observation"]))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
