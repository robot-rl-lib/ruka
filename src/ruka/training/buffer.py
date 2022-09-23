# from gym.spaces import Dict
import abc
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, OrderedDict, Tuple
import functools

import gym.spaces
import numpy as np
from gym.spaces import Box, Discrete

from stable_baselines3.common.vec_env import VecNormalize
import itertools

Array = Any 

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return



class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace = True,
    ):
        # if isinstance(observation_dim, int):
        #     self._observation_dim = (observation_dim,)

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,) + observation_dim)

        self._next_obs = np.zeros((max_replay_buffer_size,) + observation_dim)
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def get_snapshot(self):
        return {}
        return {
            "observations": self._observations,
            "next_obs": self._next_obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "terminals": self._terminals,
            "env_infos": self._env_infos,
        }

    def load_from_snapshot(self, snapshot):
        self._observations = snapshot["observations"]
        self._next_obs = snapshot["next_obs"]
        self._actions = snapshot["actions"]
        self._rewards = snapshot["rewards"]
        self._terminals = snapshot["terminals"]
        self._env_infos = snapshot["env_infos"]


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            # observation_dim=get_dim(self._ob_space),
            observation_dim=self._ob_space.shape,
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

class NormReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env: VecNormalize,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=self._ob_space.shape,
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, 
                    next_observation, terminal, env_info, **kwargs):
        observation = self.env.unnormalize_obs(observation)
        reward = self.env.unnormalize_reward(reward)
        next_observation = self.env.unnormalize_obs(next_observation)

        super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            env_info=env_info,
            **kwargs
        )

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        batch["observations"] = self.env.normalize_obs(batch["observations"])
        batch["rewards"] = self.env.normalize_reward(batch["rewards"])
        batch["next_observations"] = self.env.normalize_obs(batch["next_observations"])
        return batch
        
        
        


class VecEnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env: VecNormalize,
            env_info_sizes=None,
            handle_tl: str = False,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        self._handle_tl = handle_tl

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=self._ob_space.shape,
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, 
                    next_observation, terminal, env_info, **kwargs):
        if self._handle_tl and env_info.get('is_time_limit', False) and terminal:
            return
            if 'terminal_observation' not in env_info: return
            next_observation = self.env.unnormalize_obs(env_info['terminal_observation'])
            terminal = False
        else:
            observation = self.env.unnormalize_obs(observation)
        reward = self.env.unnormalize_reward(reward)
        next_observation = self.env.unnormalize_obs(next_observation)

        super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            env_info=env_info,
            **kwargs
        )

    def random_batch(self, batch_size: int):
        batch = super().random_batch(batch_size)
        batch["observations"] = self.env.normalize_obs(batch["observations"])
        batch["rewards"] = self.env.normalize_reward(batch["rewards"])
        batch["next_observations"] = self.env.normalize_obs(batch["next_observations"])
        return batch
    
    def add_vec_transitions(self, transitions):
        n_transitions = transitions["observations"].shape[0] * transitions["observations"].shape[1]

        transitions["observations"] = transitions["observations"].reshape((n_transitions,) + transitions["observations"].shape[2:])
        transitions["actions"] = transitions["actions"].reshape((n_transitions,) + transitions["actions"].shape[2:])
        transitions["rewards"] = transitions["rewards"].reshape((n_transitions,) + transitions["rewards"].shape[2:])
        transitions["next_observations"] = transitions["next_observations"].reshape((n_transitions,) + transitions["next_observations"].shape[2:])
        transitions["terminals"] = transitions["terminals"].reshape((n_transitions,) + transitions["terminals"].shape[2:])
        
        transitions["env_infos"] = list(itertools.chain(*transitions["env_infos"]))

        self.add_path(transitions)
        

class RadVecEnvReplayBuffer(VecEnvReplayBuffer):
    def __init__(self, crop_size: int, *args, num_direct_features: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._crop_size = crop_size
        self._num_direct_features = num_direct_features

    def random_batch(self, batch_size: int):
        batch = super().random_batch(batch_size)
        batch["observations"] = self.crop_augment(batch["observations"])
        batch["next_observations"] = self.crop_augment(batch["next_observations"])
        return batch

    def crop_augment(self, imgs):
        n, c, h, w = imgs.shape
        crop_max = h - self._crop_size + 1
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)

        
        cropped = np.empty((n, c, self._crop_size, self._crop_size), dtype=imgs.dtype)

        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cropped[i] = img[:, h11:h11 + self._crop_size, w11:w11 + self._crop_size]

        if self._num_direct_features is not None:
            directs = imgs[:, -1, :, :].reshape((n, -1))
            directs = directs[:, :self._crop_size * self._crop_size]
            directs = directs.reshape((n, self._crop_size, self._crop_size))
            cropped[:, -1, :, :] = directs

        return cropped
