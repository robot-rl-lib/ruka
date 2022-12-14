import gym
import numpy as np
from gym.wrappers import TimeLimit
from ruka.observation import Observation

class ImageToPyTorchDictLike(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env, add_batch_dim = True):
        super(ImageToPyTorchDictLike, self).__init__(env)
        self._add_batch_dim = add_batch_dim

        space_dict = dict()
        for key, space in env.observation_space.items():
            if len(space.shape) == 3:
                space_dict[key] = gym.spaces.Box(low=0, high=1, shape=(space.shape[-1], space.shape[0], space.shape[1],))
            else:
                space_dict[key] = gym.spaces.Box(low=space.low, high=space.high, shape=space.shape)

        self.observation_space = gym.spaces.Dict(space_dict)

    def observation(self, observation):
        result = Observation()
        for key, value in observation.items():
            if len(value.shape) == 3:
                result[key] = np.transpose(value, axes=(2, 0, 1))
            else:
                result[key] = value
            if self._add_batch_dim:
                result[key] = result[key][None]
            
        return result


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            # high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float64
            # shape=old_shape,
            # dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))
        # return np.transpose(np.uint8(observation*255), axes=(2, 0, 1))


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))
