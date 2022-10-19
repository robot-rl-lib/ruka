from dataclasses import dataclass
import numpy as np
from ruka.robot.env import RobotEnv
from ruka.robot import ArmVelocityController, Camera, Gripper, Robot
import time
import gym
from manipulation_main.common import transformations
from ruka.robot.xarm import XArmError


class VelocityControlRobotEnv(RobotEnv):
    def __init__(
            self,
            camera: Camera,
            robot: Robot,
            arm: ArmVelocityController,
            gripper: Gripper,
            stop_time: float,
            gripper_open_close_time: float,
            max_xyz_velocity: float,
            max_roll_velocity: float,
            gripper_open_position: float,
            gripper_close_position: float,
            gripper_open_position_reset: float,
            max_z: float,
            dt: float
        ):
        self.robot = robot
        self.camera = camera
        self.arm = arm
        self.gripper = gripper
        self.stop_time = stop_time
        self.gripper_open_close_time = gripper_open_close_time
        self.max_xyz_velocity = max_xyz_velocity
        self.max_roll_velocity = max_roll_velocity
        self.gripper_open_position = gripper_open_position
        self.gripper_close_position = gripper_close_position
        self.gripper_open_position_reset = gripper_open_position_reset
        self.max_z = max_z
        self.dt = dt

        self._gripper_open = None
        self.__observation_ts = None

    def reset(self):
        self.robot.reset()

        self.arm.set_velocity(0, 0, 0, 0, 0, 0)
        time.sleep(self.gripper_open_close_time)
        self.gripper.set_gripper_position(self.gripper_open_position)
        time.sleep(self.gripper_open_close_time)
        self._gripper_open = True

        return self._get_observation()

    def step(self, action):
        # Parse action.

        vx, vy, vz, vroll, gripper = np.squeeze(action)

        vx = np.clip(vx, -1, 1)
        vy = np.clip(vy, -1, 1)
        vz = np.clip(vz, -1, 1)
        vroll = np.clip(vroll, -1, 1)
        gripper = np.clip(gripper, -1, 1)

        x, y, z, roll, pitch, yaw = self.arm.position

        assert -1 <= vx <= 1
        assert -1 <= vy <= 1
        assert -1 <= vz <= 1
        assert -1 <= vroll <= 1
        assert -1 <= gripper <= 1

        # Un-scale.
        vx *= self.max_xyz_velocity
        vy *= self.max_xyz_velocity
        vz *= self.max_xyz_velocity
        vroll *= self.max_roll_velocity

        if x < 150 or x > 600:
            raise XArmError('X coordinate exceeds limit')
        if y < -300 or y > 300:
            raise XArmError('Y coordinate exceeds limit')
        if z > 650:
            raise XArmError('Z coordinate exceeds limit')

        if yaw < 0:
            yaw += 360
        if yaw < 120 or yaw > 250:
            raise XArmError('YAW coordinate exceeds limit')

        # Transform from tool-space to world-space.
        world_to_tool = transformations.compose_matrix(angles=[roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180])
        tool_to_camera = transformations.compose_matrix(angles=[0, 0, np.pi * 1.5])
        world_to_camera = world_to_tool @ tool_to_camera
        vx, vy, vz, _ = world_to_camera @ np.array([vx, vy, vz, 1])

        # Open gripper.
        if gripper > 0. and not self._gripper_open:
            self.gripper.set_gripper_position(self.gripper_open_position)
            self._gripper_open = True

        # Close gripper.
        elif gripper < 0. and self._gripper_open:
            self.gripper.set_gripper_position(self.gripper_close_position)
            self._gripper_open = False

        # Move.
        else:
            self.arm.set_velocity(vx, vy, vz, 0, 0, -vroll)

        return self._get_observation()

    @property
    def action_space(self):
        """
        vx, vy, vz, vroll, gripper
        """
        return gym.spaces.Box(-1., 1., shape=(5,), dtype=np.float32)

    @property
    def observation_space(self):
        """
        RGBD + gripper + z
        """
        return gym.spaces.Box(0, 1000, shape=(camera.w, camera.h, 7))

    def close(self):
        self.robot.disable()
        self.camera.stop()

    def _get_observation(self):
        if self.__observation_ts:
            time.sleep(max(0, self.__observation_ts + self.dt - time.time()))
        else:
            self.camera.start()

        # RGBD
        frame = self.camera.capture()
        self.__observation_ts = time.time()

        # Gripper + Z.
        x, y, z, _, _, yaw = self.arm.position
        gripper = self.gripper.gripper_position

        z_pad = np.zeros((self.camera.height, self.camera.width)) + z
        z_pad = (z_pad / self.max_z) * 255

        pos = np.zeros_like(z_pad)
        pos[0, 0], pos[0, 1], pos[0, 2] = x, y, yaw

        gripper_pad = np.zeros((self.camera.height, self.camera.width)) + gripper
        gripper_pad = (gripper_pad / max(self.gripper_open_position, self.gripper_open_position_reset)) * 255
        return np.dstack((frame, gripper_pad, z_pad, pos))
