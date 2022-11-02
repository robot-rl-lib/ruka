import pytest
import gym
import itertools
import numpy as np
import functools
from ruka.environments.common.env import Policy
from ruka.environments.common.path_iterators import collect_paths,collect_transitions, \
                                                    async_thread_collect_transitions, \
                                                    async_thread_collect_paths, \
                                                    async_process_collect_transitions, \
                                                    async_process_collect_paths
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

    def _test_path(self, path_iter):
        for i,path in enumerate(itertools.islice(path_iter, 10)):
            assert len(path.actions) == 2, len(path.actions)
            assert len(path.rewards) == 2, len(path.rewards)
            assert len(path.infos) == 2, len(path.infos)
            assert len(path.observations) == 3, len(path.observations)
            assert path.observations[0].shape == self.env.observation_space.shape, (path.observations[0].shape, self.env.observation_space.shape)
            assert isinstance(path.actions[0], int), (type(path.actions[0]), path.actions)
            assert isinstance(path.infos[0], dict), (type(path.infos[0]), path.infos)
            assert isinstance(path.rewards[0], float), (type(path.rewards[0]), path.rewards)
            
            last_tr = None
            for j,tr in enumerate(path.transitions):
                if last_tr is not None:
                    assert np.all(last_tr.next_observation == tr.observation), (last_tr.next_observation, tr.observation, path.observations)
                last_tr = tr
                assert tr.done == (j == len(path) - 1)

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

    def test_sync_path_iter(self):
        path_iter = collect_paths(self.env, self.policy)
        self._test_path(path_iter)

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

    def test_async_thread_path(self):
        name = 'thread_path'
        _make_policy = functools.partial(make_policy, name)        
        path_iter = async_thread_collect_paths(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_path(path_iter)

    def test_async_process_path(self):
        name = 'process_path'
        _make_policy = functools.partial(make_policy, name)          
        path_iter = async_process_collect_paths(make_env, _make_policy, self.policy, 10,  name=name)
        self._test_path(path_iter)
