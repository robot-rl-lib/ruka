import gym
import numpy as np

class SRWrapper(gym.Wrapper):
    """ Calculate success rate"""

    def __init__(self, env):
        self.env = env
        self._srall = []
        self._step = 0
        self._rewards = []

    def reset(self):
        self._step = 0
        self._rewards = []
        return self.env.reset()

    @property
    def sr10(self):
        return np.mean(self._srall[-10:])

    @property
    def sr50(self):
        return np.mean(self._srall[-50:])

    @property
    def sr100(self):
        return np.mean(self._srall[-100:])

    @property
    def srall(self):
        return np.mean(self._srall)

    @property
    def sr_mean(self):
        return self.sr10

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        is_success = False
        if 'is_success' in info and info['is_success']:
            is_success = True
    
        self._step += 1
        self._rewards.append(rew)

        if done:
            self._srall.append(is_success)
            print(f'Ep {len(self._srall)} success {is_success} len {self._step} rew {np.sum(self._rewards):.3f} SR10 {self.sr10:.3f} SR50 {self.sr50:.3f} SR_ALL {self.srall:.3f}', flush=True)

        return obs, rew, done, info

