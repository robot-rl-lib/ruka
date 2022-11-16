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
from .rewards import Reward
from .curriculum import WorkspaceCurriculum
from .configs import EnvironmentConfig, Observe
from .simulation.model import Model


class RobotEnv(World):
    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        PICKED_WRONG = 2
        TIME_LIMIT = 3
        CLEARING_OBJECT_PICKED = 4

    def __init__(
        self,
        config: EnvironmentConfig,
        validate: bool):
        super().__init__(config, validate)

        self.config = config

        # Robot.
        self._actuator = actuator.Actuator(self, config)
        self.action_space = self._actuator.action_space

        # Reward.
        self._reward_fn = Reward(config['reward'], self)

        # Assign the sensors (depth + actuator).
        self._camera = sensor.OnGripperCamera(
            config.on_gripper_camera_config,
            self,
        )
        # Curriculum.
        self._initial_height = 0.3
        self._init_ori = transformations.quaternion_from_euler(np.pi, 0., 0.)

        self.curriculum = WorkspaceCurriculum(config['curriculum'], self, validate)
        self.history = self.curriculum._history
        self.sr_mean = 0.
        self._last_sim_time = 0
        self._last_rgb = None
        self.target_object = None

        self._last_pos = {}
        self._timestep = 0
        self._observation_types = config.observation_types

        self.reset()

        self.observation_space = self.get_naive_space(self.obs)

    @staticmethod
    def get_naive_space(obs):
        assert len(obs) > 0

        space = dict()
        for key, value in obs.items():
            space[key] = gym.spaces.Box(low=0, high=1, shape=value.shape)

        return gym.spaces.Dict(space)

    def reset(self):
        self._timestep = 0
        self.reset_sim()
        self._actuator.reset()
        self._camera.reset()
        self._reward_fn.reset()

        self.target_object = self._scene.pickable_objects[-1]
        self.pickable_objects = self._scene.pickable_objects
        self._reward_fn.reset()


        self.episode_step = 0
        self.status = RobotEnv.Status.RUNNING
        self._last_sim_time = self.sim_time
        
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
        self._last_sim_time = self.sim_time
        self._timestep += 1
        self._last_rgb = None
        self._actuator.step(action)

        new_obs = self._observe()

        reward, self.status = self._reward_fn()

        if (self.status != RobotEnv.Status.RUNNING) and \
            (self.status != RobotEnv.Status.CLEARING_OBJECT_PICKED):
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
        return self.obs, reward, done, dict(
                                            is_success=self.status==RobotEnv.Status.SUCCESS,
                                            is_time_limit=self.status==RobotEnv.Status.TIME_LIMIT,
                                            transition_time=self.sim_time - self._last_sim_time
                                        )
            

    def _observe(self):
        rgb, depth, mask = self._camera.get_state()
        observation = dict()
        rgb_for_video = rgb

        if Observe.RGB in self._observation_types:
            observation[Observe.RGB.value] = rgb / 255

        if Observe.GRAY in self._observation_types:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) / 255
            observation[Observe.GRAY.value] = gray[..., None]

        if Observe.DEPTH in self._observation_types:
            observation[Observe.DEPTH.value] = depth[..., None]

        if Observe.TARGET_SEGMENTATION in self._observation_types:
            mask = (mask==self.target_object.model_id).astype(float)
            observation[Observe.TARGET_SEGMENTATION.value] = mask[..., None]

            rgb_for_video = np.concatenate(
                [rgb_for_video, (mask[..., None] * np.ones((1,1,3)) * 255).astype(np.uint8)],
                axis=1)

        if Observe.ROBOT_POS in self._observation_types:
            pos, quat = self.get_pose()
            observation[Observe.ROBOT_POS.value] =  np.concatenate([pos, quat])
            
        if Observe.GRIPPER in self._observation_types:
            observation[Observe.GRIPPER.value] =  np.array([self._actuator.get_state()])

        if Observe.HEIGHT in self._observation_types:
            pos, quat = self.get_pose()
            observation[Observe.HEIGHT.value] = np.array([pos[2]])

        if Observe.SENSOR_PAD in self._observation_types:
            sensor_pad = np.zeros(rgb.shape[:2])
            sensor_pad[0, 0] = self._actuator.get_state()
            observation[Observe.SENSOR_PAD.value] = sensor_pad[..., None]
        
        if Observe.TIMESTEP in self._observation_types:
            observation[Observe.TIMESTEP.value] = np.array([self._timestep])

        if Observe.TRANSITION_TIME in self._observation_types:
            observation[Observe.TRANSITION_TIME.value] = np.array([self.sim_time - self._last_sim_time])

        self._last_rgb = rgb_for_video
        self._last_pos = {'pose': self.get_pose(), 'target': self.target_object.getBase()}
        
        return observation

    def get_pose(self):
        return self._actuator._model.get_pose()

    def get_image(self):
        if self._last_rgb is None:
            rgb, depth, mask = self._camera.get_state()
            if Observe.TARGET_SEGMENTATION in self._observation_types:
                rgb = np.concatenate(
                    [rgb, (np.zeros_like(mask)[..., None]*np.ones((1,1,3)) * 255).astype(np.uint8)],
                    axis=1)
        else:
            rgb = self._last_rgb
        return rgb
