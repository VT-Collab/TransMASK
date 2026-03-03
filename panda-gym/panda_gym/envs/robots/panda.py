from typing import Optional

import numpy as np
from gymnasium import spaces
import pybullet as p

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        random_init: bool = False,
        camera_width: int = 256,
        camera_height: int = 256,
        return_image_obs: bool = False
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.random_init = random_init
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.return_image_obs = return_image_obs
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )
        
        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        if self.return_image_obs:
            self.static_cam_params = {
                "target_position": np.zeros(3),
                "distance": 1.4,
                "yaw": 45,
                "pitch": -30,
                "roll": 0,
            }

            self.static_view_matrix = self.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.static_cam_params["target_position"],
                distance=self.static_cam_params["distance"],
                yaw=self.static_cam_params["yaw"],
                pitch=self.static_cam_params["pitch"],
                roll=self.static_cam_params["roll"],
                upAxisIndex=2,
            )
            
            self.ee_cam_offset_pos = np.array([0.05, 0.0, 0.0])   # 5cm forward
            self.ee_cam_offset_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])

            self.proj_matrix = self.sim.physics_client.computeProjectionMatrixFOV(
                fov=60,
                aspect=self.camera_width / self.camera_height,
                nearVal=0.01,
                farVal=2.0,
            )

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        if self.control_type == 'ee':
            # end-effector position and velocity
            ee_position = np.array(self.get_ee_position())
            # ee_velocity = np.array(self.get_ee_velocity())
            # fingers opening
            if not self.block_gripper:
                fingers_width = self.get_fingers_width()
                # observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
                observation = np.concatenate((ee_position, [fingers_width]))
            else:
                # observation = np.concatenate((ee_position, ee_velocity))
                observation = ee_position
            if self.return_image_obs:
                static_image, static_seg = self._get_static_image_obs()
                ee_image, ee_seg = self._get_ee_image_obs()
            else:
                static_image, static_seg = None, None
                ee_image, ee_seg = None, None
            image_observation = {'static_image': static_image,
                                 'ee_image': ee_image,
                                 'static_seg': static_seg,
                                 'ee_seg': ee_seg}
        else:
            # joint position and velocities
            joint_position = np.array([self.get_joint_angle(j_idx) for j_idx in range(7)])
            # joint_velocity = np.array([self.get_joint_velocity(j_idx) for j_idx in range(7)])
            # fingers opening
            if not self.block_gripper:
                fingers_width = self.get_fingers_width()
                # observation = np.concatenate((joint_position, joint_velocity, [fingers_width]))
                observation = np.concatenate((joint_position, [fingers_width]))
            else:
                # observation = np.concatenate((joint_position, joint_velocity))
                observation = joint_position
            if self.return_image_obs:
                static_image, static_seg = self._get_static_image_obs()
                ee_image, ee_seg = self._get_ee_image_obs()
            else:
                static_image, static_seg = None, None
                ee_image, ee_seg = None, None
            image_observation = {'static_image': static_image,
                                'ee_image': ee_image,
                                'static_seg': static_seg,
                                'ee_seg': ee_seg}
        return observation, image_observation

    def _get_static_image_obs(self):
        _, _, rgba, _, seg = self.sim.physics_client.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.static_view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        rgba = np.frombuffer(rgba, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )
        seg = np.array(seg).reshape(self.camera_height, self.camera_width)

        background_mask = seg == -1
        rgb = rgba[:, :, :3]
        rgb[background_mask] = np.uint8(self.sim.background_color * 255)
        return rgb, seg

    def _get_ee_image_obs(self):
        ee_pos = np.array(self.get_ee_position())
        ee_orn = np.array(self.get_ee_orientation())

        cam_pos, cam_orn = self.sim.physics_client.multiplyTransforms(
            ee_pos,
            ee_orn,
            self.ee_cam_offset_pos.tolist(),
            self.ee_cam_offset_orn,
        )

        rot_mat = np.array(
            self.sim.physics_client.getMatrixFromQuaternion(cam_orn)
        ).reshape(3, 3)

        cam_forward = rot_mat @ np.array([1, 0, 0])
        cam_up = rot_mat @ np.array([0, 0, 1])

        view_matrix = self.sim.physics_client.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_pos + 0.2 * cam_forward,
            cameraUpVector=cam_up,
        )

        _, _, rgba, _, seg = self.sim.physics_client.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        rgba = np.frombuffer(rgba, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )
        seg = np.array(seg).reshape(self.camera_height, self.camera_width)

        background_mask = seg == -1
        rgb = rgba[:, :, :3]
        rgb[background_mask] = np.uint8(self.sim.background_color * 255)
        return rgb, seg

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        if self.random_init:
            noise = np.random.randn(len(self.neutral_joint_values)) * 0.1
        else:
            noise = 0
        neutral_joint_values = self.neutral_joint_values + noise
        self.set_joint_angles(neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as (rx, ry, rz)"""
        return self.get_link_orientation(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
