
import cv2
import gym
import time
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

from ruka.robot.env import RobotEnv
from ruka.robot.xarm_ import XArmError
from ruka.observation import Observe, Observation
from ruka.robot.robot import ArmPosVelControlled, Camera, GripperPosControlled, \
                             Robot, ControlMode
from manipulation_main.common import transformations
from ruka.util.x3d import Vec3


@dataclass
class MinMaxLimit:
    min: float
    max: float

@dataclass
class Limits:
    x: MinMaxLimit
    y: MinMaxLimit
    z: MinMaxLimit


class VelocityControlRobotEnv(RobotEnv):
    def __init__(
            self,
            camera: Camera,
            robot: Robot,
            arm: ArmPosVelControlled,
            gripper: GripperPosControlled,
            stop_time: float,
            gripper_open_close_time: float,
            max_xyz_velocity: float,
            max_roll_velocity: float,
            gripper_open_position: float,
            gripper_close_position: float,
            gripper_open_position_reset: float,
            max_z: float,
            dt: float,
            limits_x: Tuple[float, float] = (150, 600),
            limits_y: Tuple[float, float] = (-300, 300),
            limits_z: Tuple[float, float] = (-1000, 650),
            dict_like: bool = False,
            observation_types: List[Observe] = list((Observe.DEPTH,))):
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
        self._limits = Limits(x=MinMaxLimit(*limits_x),
                              y=MinMaxLimit(*limits_y),
                              z=MinMaxLimit(*limits_z))

        self._dict_like = dict_like

        self._gripper_open = None
        self.__observation_ts = None
        self._observation_types=observation_types

    def reset(self):
        self.robot.steady(ControlMode.VEL)

        self.arm.set_vel([0, 0, 0], [0, 0, 0])
        time.sleep(self.gripper_open_close_time)
        self.gripper.set_gripper_pos(self.gripper_open_position)
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

        x, y, z = self.arm.pos
        roll, pitch, yaw = self.arm.angles

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

        if x < self._limits.x.min or x > self._limits.x.max:
            raise XArmError('X coordinate exceeds limit')
        if y < self._limits.y.min or y > self._limits.y.max:
            raise XArmError('Y coordinate exceeds limit')
        if z > self._limits.z.max:
            raise XArmError('Z coordinate exceeds limit')

        if yaw < 0:
            yaw += 360
        if yaw < 70 or yaw > 290:
            raise XArmError('YAW coordinate exceeds limit')

        # Transform from tool-space to world-space.
        world_to_tool = transformations.compose_matrix(angles=[roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180])
        tool_to_camera = transformations.compose_matrix(angles=[0, 0, np.pi * 1.5])
        world_to_camera = world_to_tool @ tool_to_camera
        vx, vy, vz, _ = world_to_camera @ np.array([vx, vy, vz, 1])

        # Open gripper.
        if gripper > 0. and not self._gripper_open:
            self.gripper.set_gripper_pos(self.gripper_open_position)
            self._gripper_open = True

        # Close gripper.
        elif gripper < 0. and self._gripper_open:
            self.gripper.set_gripper_pos(self.gripper_close_position)
            self._gripper_open = False

        # Move.
        else:
            self.arm.set_vel([vx, vy, vz], [0, 0, -vroll])

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
        if not self._dict_like:
            return gym.spaces.Box(0, 1000, shape=(self.camera.height, self.camera.width, 7))
        else:
            observation_space = gym.spaces.Dict({})

            if Observe.RGB in self._observation_types:
                observation_space[Observe.RGB.value] = gym.spaces.Box(low=0, high=255, shape=(self.camera.height, self.camera.width, 3))
            if Observe.GRAY in self._observation_types:
                observation_space[Observe.GRAY.value] = gym.spaces.Box(low=0, high=255, shape=(self.camera.height, self.camera.width, 1))
            if Observe.DEPTH in self._observation_types:
                # which correct low high?
                observation_space[Observe.DEPTH.value] = gym.spaces.Box(low=0, high=255, shape=(self.camera.height, self.camera.width, 1))
            if Observe.ROBOT_POS in self._observation_types:
                observation_space[Observe.ROBOT_POS.value] = gym.spaces.Box(low=-1, high=1, shape=(6,))
            if Observe.GRIPPER in self._observation_types:
                observation_space[Observe.GRIPPER.value] = gym.spaces.Box(low=-1, high=1, shape=(1,))
        return observation_space

    def close(self):
        self.robot.park()
        self.camera.stop()

    def _get_observation(self):       
        if self.__observation_ts:
            time.sleep(max(0, self.__observation_ts + self.dt - time.time()))
        else:
            self.camera.start()

        # RGBD
        frame = self.camera.capture()
        self.__observation_ts = time.time()

        if self._dict_like:
            return self._dict_like_observation(frame)
        else:
            return self._dstack_observation_depricated(frame)

    def _dstack_observation_depricated(self, frame):
        # Gripper + Z.
        x, y, z = self.arm.pos
        roll, pitch, yaw = self.arm.angles
        gripper = self.gripper.gripper_pos

        z_pad = np.zeros((self.camera.height, self.camera.width)) + z
        z_pad = (z_pad / self.max_z) * 255

        pos = np.zeros_like(z_pad)
        pos[0, 0], pos[0, 1], pos[0, 2], pos[0, 3], pos[0, 4], pos[0, 5] = x, y, z, roll, pitch, yaw

        gripper_pad = np.zeros((self.camera.height, self.camera.width)) + gripper
        gripper_pad = (gripper_pad / max(self.gripper_open_position, self.gripper_open_position_reset)) * 255
        return np.dstack((frame, gripper_pad, z_pad, pos))

    def _dict_like_observation(self, frame):

        rgb = frame[:,:, :3]
        depth = frame[:,:,3][:,:,None]

        # Gripper + Z.
        x, y, z = self.arm.pos
        roll, pitch, yaw = self.arm.angles
        gripper = self.gripper.gripper_pos / 100.

        obs = Observation()

        if Observe.RGB in self._observation_types:
            obs[Observe.RGB.value] = rgb.astype(np.uint8)
        if Observe.GRAY in self._observation_types:
            obs[Observe.GRAY.value] = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[:,:,None].astype(np.uint8)
        if Observe.DEPTH in self._observation_types:
            obs[Observe.DEPTH.value] = depth.astype(np.float32)
        if Observe.ROBOT_POS in self._observation_types:
            obs[Observe.ROBOT_POS.value] = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
        if Observe.GRIPPER in self._observation_types:
            obs[Observe.GRIPPER.value] = np.array([gripper], dtype=np.float32)
    
        return obs

    def go_to_random(self, x_min, x_max, y_min, y_max):
        self.robot.steady(ControlMode.POS)

        x, y, z = self.arm.pos
        roll, pitch, yaw = self.arm.angles
        x_new = np.random.uniform(x_min+1, x_max-1)
        y_new = np.random.uniform(y_min+1, y_max-1)
        self.arm.set_pos([x_new, y_new, z], [roll, pitch, yaw])
        self.gripper.set_gripper_pos(self.gripper_open_position)
        self._gripper_open = True


