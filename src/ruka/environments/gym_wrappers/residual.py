import gym
import numpy as np
from ruka.observation import Observation


class ResidualWrapper(gym.Wrapper):
    """ Add base policy action to step action """

    def __init__(self, env, base_policy):
        self.env = env
        self.policy = base_policy
        self._vec_env = None
        self._last_obs = None

    def step(self, action):
        if isinstance(self._last_obs, dict):
            # TODO: how to select right one actions vs action?
            if hasattr(self.policy, 'get_actions'):
                base_action = self.policy.get_actions(Observation(self._last_obs).to_pytorch())
            else:
                base_action = self.policy.get_action(Observation(self._last_obs).to_pytorch())
        else:
            if len(action.shape) == 1:
                base_action = self.policy.get_action(self._last_obs)
            else:
                base_action = self.policy.get_actions(self._last_obs)

        action = np.clip(-1,1, action + base_action)
        self._last_obs, rew, done, info = self.env.step(action)
        return self._last_obs, rew, done, info

    def reset(self):
        self.policy.reset()
        self._last_obs = self.env.reset()
        return self._last_obs
