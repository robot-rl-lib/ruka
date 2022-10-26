import gym
import cv2
import numpy as np
from copy import deepcopy
from typing import Callable, Dict
from ruka.util.debug import smart_shape, smart_stats

class SquareAndResize(gym.ObservationWrapper):
    def __init__(self, env, keys, hw_size, square=True):
        super().__init__(env)
        self._keys = keys
        self._hw_size = hw_size
        self._square = square

        self.observation_space = deepcopy(env.observation_space)
        for k in self._keys:
            cur = self.observation_space[k]
            low, high = cur.low, cur.high
            if isinstance(cur.low, np.ndarray):
                low = low.flatten()[0]
                high = high.flatten()[0]
            self.observation_space[k] = gym.spaces.Box(low=low, high=high, shape=(*hw_size, cur.shape[-1]))

    def observation(self, obs):
        if self._square:
            def clip_to_square(img):
                return img[:, ((img.shape[1] - img.shape[0]) // 2):((img.shape[1] - img.shape[0]) // 2 + img.shape[0])]
            for k in self._keys:
                obs[k] = clip_to_square(obs[k])

        for k in self._keys:
            channels = obs[k].shape[-1]
            tgt_size = self._hw_size[::-1]
            obs[k] = cv2.resize(obs[k], tgt_size)
            if channels == 1:
                obs[k] = np.expand_dims(obs[k], -1)

        return obs

class Process(gym.ObservationWrapper):

    def __init__(self, env, process: Callable):
        super().__init__(env)
        self._process = process or (lambda x: x)

    def observation(self, obs):
        return self._process(obs)
