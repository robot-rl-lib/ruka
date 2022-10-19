import os
import time
import pybullet as p
import numpy as np
import functools
import sys
import gym
import collections
from enum import Enum
import cv2

from manipulation_main.common import transformations
from . import actuator, sensor
from .simulation.simulation import World 
from .rewards import Reward, ShapedCustomReward
from .curriculum import WorkspaceCurriculum


class RobotEnv(World):
    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3

    def __init__(self, config, validate):
        super().__init__(config, validate)

        self.config = config

        # Robot.
        self._actuator = actuator.Actuator(self, config)
        self.action_space = self._actuator.action_space

        # Reward.
        self._reward_fn = ShapedCustomReward(config['reward'], self)

        # Assign the sensors (depth + actuator).
        self._camera = sensor.RGBDSensor(config['sensor'], self, full_obs=False)
        self._sensors = [self._camera, self._actuator]
        shape = self._camera.state_space.shape
        self.observation_space = \
            gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], 2))

        # Curriculum.
        self._initial_height = 0.3
        self._init_ori = transformations.quaternion_from_euler(np.pi, 0., 0.)

        self.curriculum = WorkspaceCurriculum(config['curriculum'], self, validate)
        self.history = self.curriculum._history
        self.sr_mean = 0.
        self._last_rgb = None

        # Initial reset.
        self.reset()

    def reset(self):
        self.reset_sim() #world + scene reset
        self._actuator.reset()
        self._camera.reset()
        self._reward_fn.reset()

        self.episode_step = 0
        self.episode_rewards = np.zeros(self.config.time_horizon)
        self.status = RobotEnv.Status.RUNNING
        self.obs = self._observe()

        return self.obs

    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns:
            A tuple (obs, reward, done, info), where done is a boolean flag
            indicating whether the current episode finished.
        """
        self._last_rgb = None
        self._actuator.step(action)

        new_obs = self._observe()

        reward, self.status = self._reward_fn(self.obs, action, new_obs)
        self.episode_rewards[self.episode_step] = reward

        if self.status != RobotEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.config.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        if done:
            self.curriculum.update(self)

        self.episode_step += 1
        self.obs = new_obs
        if len(self.curriculum._history) != 0:
            self.sr_mean = np.mean(self.curriculum._history)
        super().step_sim()
        return self.obs, reward, done, {"is_success":self.status==RobotEnv.Status.SUCCESS, "is_time_limit": self.status==RobotEnv.Status.TIME_LIMIT}

    def _observe(self):
        rgb, depth, _ = self._camera.get_state()
        self._last_rgb = rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)/255
        sensor_pad = np.zeros(self._camera.state_space.shape[:2])
        sensor_pad[0][0] = self._actuator.get_state()
        pose, orientation = self.get_pose()
        sensor_pad[0, 1] = pose[2]
        obs_stacked = np.dstack((gray, sensor_pad))
        return obs_stacked

    def get_pose(self):
        return self._actuator._model.get_pose()

    def get_image(self):
        if self._last_rgb is None:
            rgb, depth, _ = self._camera.get_state()
        else:
            rgb = self._last_rgb
        return rgb

    # -------------

    def is_simplified(self):
        return False

    def __getattr__(self, attr):
        if attr == 'depth_obs':
            return True
        if attr == 'full_obs':
            return False
        raise AttributeError()