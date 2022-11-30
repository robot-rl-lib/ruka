
import cv2
import gym
import time
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

from ruka.robot.env import RobotEnv
from ruka.observation import Observe, Observation
from ruka.robot.robot import Camera, ControlMode, ArmInfo, GripperInfo, Robot
from manipulation_main.common import transformations
from ruka.util.x3d import Vec3
from ruka.environments.common.controller import Controller
from ruka.robot.actions import RobotAction


class VelocityControlRobotEnv(RobotEnv):
    def __init__(
            self,
            camera: Camera,
            robot: Robot,
            arm_info: ArmInfo,
            action_controller: Controller,
            gripper_info: GripperInfo,
            gripper_open_position: float,
            gripper_close_position: float,
            dt: float,
            observation_types: List[Observe] = list((Observe.DEPTH,))):
        self.robot = robot
        self.arm_info= arm_info
        self.camera = camera
        self.action_controller = action_controller
        self.gripper_info = gripper_info

        self.gripper_open_position = gripper_open_position
        self.gripper_close_position = gripper_close_position

        self.dt = dt

        self._gripper_open = None
        self._observation_ts = None
        self._observation_types=observation_types

    def reset(self):

        # TODO: understand why we need to reset controller on env reset
        self.action_controller.reset()

        # dont use RobotAction.Go_HOME because it reset control to POS
        # TODO: fix it
        self.robot.go_home()
        self.robot.steady(ControlMode.VEL)

        self.action_controller.act({
                'robot': RobotAction.NOP.value,
                'gripper': self.gripper_open_position,
                'move': {'xyz': (0,0,0), 'rpy': (0,0,0)}
                })
        self._gripper_open = True

        return self.get_observation()

    def step(self, action):
        # Parse action.
        self.action_controller.act({'robot': RobotAction.NOP.value,
                                    'gripper': action['gripper'],
                                    'move': action['tool_vel'], })

        if action['gripper'] > self.gripper_close_position:
            self._gripper_open = True
        elif action['gripper'] == self.gripper_close_position:
            self._gripper_open = False
        return self.get_observation()

    @property
    def action_space(self):
        """
        {
            'gripper': gripper_position, #  (0..100),
            'tool_vel': {'xyz': (0,0,0), 'rpy': (0,0,0)}, # velocity over each dims
        }
        """
        action_space =  gym.spaces.Dict({'tool_vel': self.action_controller.action_space['move'],
                                         'gripper': self.action_controller.action_space['gripper'],})
        return action_space

    @property
    def observation_space(self):
        """
        Dict like observation
        """
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
        if Observe.GRIPPER_OPEN in self._observation_types:
            observation_space[Observe.GRIPPER_OPEN.value] = gym.spaces.Box(low=0, high=1, shape=(1,))

        return observation_space

    def close(self):
        self.action_controller.act({
            'robot': RobotAction.PARK.value,
            'gripper': self.gripper_open_position,
            'move': {'xyz': (0,0,0), 'rpy': (0,0,0)}
            })
        self.camera.stop()

    def get_observation(self):
        if self._observation_ts:
            time.sleep(max(0, self._observation_ts + self.dt - time.time()))
        else:
            self.camera.start()

        # RGBD
        frame = self.camera.capture()
        self._observation_ts = time.time()

        rgb = frame[:,:, :3]
        depth = frame[:,:,3][:,:,None]

        # Gripper + Z.
        x, y, z = self.arm_info.pos
        roll, pitch, yaw = self.arm_info.angles
        gripper = self.gripper_info.gripper_pos

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
        if Observe.GRIPPER_OPEN in self._observation_types:
            obs[Observe.GRIPPER_OPEN.value] = np.array([self._gripper_open], dtype=np.uint8)

        return obs

    def go_to_random(self, x_min, x_max, y_min, y_max, yaw_min=None, yaw_max=None):
        self.robot.steady(ControlMode.POS)

        x, y, z = self.arm_info.pos
        roll, pitch, yaw = self.arm_info.angles
        x_new = np.random.uniform(x_min+1, x_max-1)
        y_new = np.random.uniform(y_min+1, y_max-1)
        yaw_new = np.random.uniform(yaw_min, yaw_max) if yaw_min is not None and yaw_max is not None else yaw

        # TODO: not use robot.set_pos directly -> move to separate pos actions
        self.robot.set_pos([x_new, y_new, z], [roll, pitch, yaw_new])
        self.robot.set_gripper_pos(self.gripper_open_position)
        self._gripper_open = True


