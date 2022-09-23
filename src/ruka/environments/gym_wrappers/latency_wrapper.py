import collections
import gym
import numpy as np

class LatencyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, latency_steps: int = 0):
        super().__init__(env)
        self.buffer_size = latency_steps + 1
        self.buffer = collections.deque(maxlen=self.buffer_size)
    def reset(self):
        obs = self.env.reset()
        self.buffer = collections.deque([obs], maxlen=self.buffer_size)
        return obs
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.buffer.append(obs)
        return self.buffer[0], rew, done, info

class LatencyVideoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, latency_steps: int = 0):
        super().__init__(env)
        self.buffer_size = latency_steps + 1
        self.buffer = collections.deque(maxlen=self.buffer_size)
        self.rgb_buffer = collections.deque(maxlen=self.buffer_size)
        
    def reset(self):
        obs = self.env.reset()
        self.buffer = collections.deque([obs], maxlen=self.buffer_size)
        self.rgb_buffer = collections.deque([self.env.get_image()], maxlen=self.buffer_size)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.buffer.append(obs)
        self.rgb_buffer.append(self.env.get_image())
        return self.buffer[0], rew, done, info

    def get_image(self):
        rgb = self.env.get_image()
        latency_rgb = self.rgb_buffer[0] if len(self.rgb_buffer) else rgb
        return np.concatenate([rgb/255, latency_rgb], axis=1)
