import pytest
import gym
import itertools
import numpy as np
import functools
from ruka.environments.common.env import Policy
from ruka.environments.common.path_iterators import collect_episodes,collect_transitions, \
                                                    async_thread_collect_transitions, \
                                                    async_thread_collect_episodes, \
                                                    async_process_collect_transitions, \
                                                    async_process_collect_episodes
global_env = None
def make_env():
    global global_env
    env = gym.make("CartPole-v1")
    env = gym.wrappers.TimeLimit(env, 2)
    global_env = env
    return env


class MyPolicy(Policy):
    def __init__(self, env, name):
        self.env = env
        self.name = name
    def get_action(self, observation):
        return self.env.action_space.sample()
    def to(self, device):
        return self
    def eval(self):
        return self
    def train(self, train):
        return self
    def state_dict(self):
        print("Get state dict", self.name)
        return {}
    def load_state_dict(self, state_dict):
        print("Load state dict", self.name)

def make_policy(name):
    global global_env
    return MyPolicy(global_env, name)

class TestIterators:

    def setup(self):
        self.env = make_env()
        self.policy = MyPolicy(self.env, 'main')

    def _test_episode(self, episode_iter):
        for i,episode in enumerate(itertools.islice(episode_iter, 10)):
            assert len(episode.actions) == 2, len(episode.actions)
            assert len(episode.rewards) == 2, len(episode.rewards)
            assert len(episode.infos) == 2, len(episode.infos)
            assert len(episode.observations) == 3, len(episode.observations)
            assert episode.observations[0].shape == self.env.observation_space.shape, (episode.observations[0].shape, self.env.observation_space.shape)
            assert isinstance(episode.actions[0], int), (type(episode.actions[0]), episode.actions)
            assert isinstance(episode.infos[0], dict), (type(episode.infos[0]), episode.infos)
            assert isinstance(episode.rewards[0], float), (type(episode.rewards[0]), episode.rewards)
            
            last_tr = None
            for j,tr in enumerate(episode.transitions):
                if last_tr is not None:
                    assert np.all(last_tr.next_observation == tr.observation), (last_tr.next_observation, tr.observation, episode.observations)
                last_tr = tr
                assert tr.done == (j == len(episode) - 1)

    def _test_transition(self, tr_iter):
        last_obs = None
        for i,tr in enumerate(itertools.islice(tr_iter, 10)):
            assert isinstance(tr.action, int), type(tr.action)
            assert tr.observation.shape == self.env.observation_space.shape, (tr.observation.shape, self.env.observation_space.shape)
            assert tr.next_observation.shape == self.env.observation_space.shape, (tr.next_observation.shape, self.env.observation_space.shape)
            assert isinstance(tr.reward, float), type(tr.reward)
            assert isinstance(tr.done, bool), type(tr.done)
            assert isinstance(tr.info, dict), type(tr.dict)
            
            if last_obs is not None:
                assert np.all(last_obs == tr.observation)
            
            if tr.done:
                last_obs = None
            else:
                last_obs = tr.next_observation

            assert tr.done == ((i + 1) % 2 == 0)

    def test_sync_episode_iter(self):
        episode_iter = collect_episodes(self.env, self.policy)
        self._test_episode(episode_iter)

    def test_sync_transition_iter(self):
        tr_iter = collect_transitions(self.env, self.policy)
        self._test_transition(tr_iter)

    def test_async_thread_transition(self):
        name = 'thread_transition'
        _make_policy = functools.partial(make_policy, name)
        tr_iter = async_thread_collect_transitions(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_transition(tr_iter)

    def test_async_process_transition(self):
        name = 'process_transition'
        _make_policy = functools.partial(make_policy, name)
        tr_iter = async_process_collect_transitions(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_transition(tr_iter)

    def test_async_thread_episode(self):
        name = 'thread_episode'
        _make_policy = functools.partial(make_policy, name)        
        episode_iter = async_thread_collect_episodes(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_episode(episode_iter)

    def test_async_process_episode(self):
        name = 'process_episode'
        _make_policy = functools.partial(make_policy, name)          
        episode_iter = async_process_collect_episodes(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_episode(episode_iter)
