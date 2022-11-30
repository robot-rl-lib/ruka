import gym
import time
import numpy as np
from ruka.robot.robot import RobotError
from ruka.util.debug import smart_shape


class RealRobotEnv(gym.Env):
    def __init__(self, robot, reward):
        self._robot = robot
        self._reward = reward
        self._broken = True
        self._last_obs = None

        self.action_space = gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        self.observation_space = robot.observation_space

    def reset(self):
        self._broken = False
        self._reward.reset()
        obs = self._robot.reset()
        self._last_obs = obs
        return obs

    def get_observation(self):
        return self._robot.get_observation()

    def step(self, action):
        start_time = time.time()
        try:
            if not self._broken:
                obs = self._robot.step(action)
        except RobotError as e:
            print(f"ERROR: {e}")
            self._broken = True
            obs = self._last_obs

        done = False
        info = {"broken": self._broken}


        reward, is_success = self._reward(obs, info)
        if is_success or self._broken:
            done = True

        self._last_obs = obs

        info.update({"is_success": is_success,
                     "timestamp_start_step": start_time,
                     "timestamp_finish_step": time.time()})
        return obs, reward, done, info
