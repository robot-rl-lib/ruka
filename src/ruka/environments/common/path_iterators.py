import os
import uuid
import itertools
import random
import queue as q
from typing import Callable, Iterator, Optional
from threading import Thread
import torch.multiprocessing as mp
import ruka_os.distributed_fs_v2 as dfs

from ruka.logging.episode import load_episode
from ruka.environments.common.env import Episode, Env, Transition, Policy

# ------------------------------------------------------------------------------------
#                           Simple synchronous version
# ------------------------------------------------------------------------------------

def collect_transitions(env: Env, policy: Policy):
    """ Iterate over Transitions, reset env when it is done """
    done = True
    while True:
        if done:
            last_obs = env.reset()
            policy.reset()

        action = policy.get_action(last_obs)
        obs, reward, done, info = env.step(action)
        yield Transition(last_obs, action, reward, obs, done, info)
        last_obs = obs


def collect_episodes(env: Env, policy: Policy, reset_env: bool = True) -> Iterator[Episode]:
    """
    Iterate over Episodes
    Args:
        env (Env):
        policy (Policy):
        reset_env (bool, optional): Reset env before collecting the episode. Defaults to True.

    Yields:
        Episode: collected episode.
    """
    done = True
    while True:
        if reset_env:
            obs = env.reset()
        else:
            obs = env.get_observation()
        policy.reset()

        obs_list = [obs]
        acts_list = []
        rews_list = []
        dones_list = []
        infos_list = []
        done = False

        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            acts_list.append(action)
            rews_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)

        yield Episode(obs_list, acts_list, rews_list, dones_list, infos_list)

def download_episodes(remote_path: str,
    max_n: Optional[int] = None,
    extention='.rlog',
    shuffle=True):
    files_in_folder = sorted(dfs.ls(remote_path))
    episode_paths = [os.path.join(remote_path, file) for file in files_in_folder if file.endswith(extention)]
    if shuffle:
        random.shuffle(episode_paths)
    if max_n:
        episode_paths = episode_paths[:max_n]
    for path in episode_paths:
        yield load_episode(path)

# ------------------------------------------------------------------------------------
#                             Async iterators
# ------------------------------------------------------------------------------------

def async_thread_collect_transitions(make_env: Callable,
                                     make_policy: Callable,
                                     orig_policy: Policy,
                                     chunk_size: int,
                                     name: Optional[str] = None):
        """
        Thread based async Transitions iterator
        Args:
            make_env: function for create env
            make_policy: function for create policy. Result of creation should have same state_dict as orig_policy
            orig_policy: will syncronized with remote policy (created by make_policy) every chunk size
            chunk_size: how many transitions we can collect without synchronization
            name: name for logs and debug
        """
        return AsyncThreadIterator(make_env, make_policy, orig_policy, chunk_size, collect_transitions, name)


def async_thread_collect_episodes(make_env: Callable,
                               make_policy: Callable,
                               orig_policy: Policy,
                               chunk_size: int,
                               name: Optional[str] = None):
        """
        Thread based async Episodes iterator
        Args:
            make_env: function for create env
            make_policy: function for create policy. Result of creation should have same state_dict as orig_policy
            orig_policy: will syncronized with remote policy (created by make_policy) every chunk size
            chunk_size: how many episodes we can collect without synchronization
            name: name for logs and debug
        """
        return AsyncThreadIterator(make_env, make_policy, orig_policy, chunk_size, collect_episodes, name)


def async_process_collect_transitions(make_env: Callable,
                                      make_policy: Callable,
                                      orig_policy: Policy,
                                      chunk_size: int,
                                      name: Optional[str] = None):
        """
        Process based async Transitions iterator
        Args:
            make_env: function for create env
            make_policy: function for create policy. Result of creation should have same state_dict as orig_policy
            orig_policy: will syncronized with remote policy (created by make_policy) every chunk size
            chunk_size: how many transitions we can collect without synchronization
            name: name for logs and debug
        """
        return AsyncProcessIterator(make_env, make_policy, orig_policy, chunk_size, collect_transitions, name)


def async_process_collect_episodes(make_env: Callable,
                                make_policy: Callable,
                                orig_policy: Policy,
                                chunk_size: int,
                                name: Optional[str] = None):
        """
        Process based async Episodes iterator
        Args:
            make_env: function for create env
            make_policy: function for create policy. Result of creation should have same state_dict as orig_policy
            orig_policy: will syncronized with remote policy (created by make_policy) every chunk size
            chunk_size: how many episodes we can collect without synchronization
            name: name for logs and debug
        """
        return AsyncProcessIterator(make_env, make_policy, orig_policy, chunk_size, collect_episodes, name)

# ------------------------------------------------------------------------------------
#                 Base classes for async versions. Not for use outside
# ------------------------------------------------------------------------------------

class AsyncIterator:
    """ Base class for all async iterators """

    def __init__(self, make_env: Callable, make_policy: Callable, orig_policy: Policy, chunk_size, make_iter: Callable, name: Optional[str] = None):
        self._make_env = make_env
        self._make_policy = make_policy
        self._orig_policy = orig_policy
        self._chunk_size = chunk_size
        self._make_iter = make_iter
        self._name = name if name is not None else uuid.uuid4()

        self._data_queue = None
        self._policy_queue = None

        self._chunk = []
        self._cur_item = 0

        self._started = False

    def __iter__(self):
        # self._started = True
        raise NotImplementedError()

    def _recv_data(self):
        return self._data_queue.get()

    def _send_policy(self, state_dict):
        return self._policy_queue.put(state_dict)

    def __next__(self):
        assert self._started, 'Need to call __iter__ before __next__'
        if self._chunk and self._cur_item < len(self._chunk) - 1:
            self._cur_item += 1
        else:
            self._chunk = self._recv_data()
            self._cur_item = 0

        return self._chunk[self._cur_item]

    def _worker_args(self):
        return (self._make_env, self._make_policy, self._chunk_size, self._make_iter,
                self._policy_queue, self._data_queue, self._name)


    @staticmethod
    def _worker_loop(make_env, make_policy, chunk_size, make_iter, policy_queue, data_queue, name):
        print(f'Worker start {name}', flush=True)

        env = make_env()
        policy = make_policy().eval()

        data_iterator = make_iter(env, policy)

        while True:
            new_policy_state = policy_queue.get()

            if new_policy_state is None:
                break
            else:
                policy.load_state_dict(new_policy_state)

            chunk = []
            for data in itertools.islice(data_iterator, chunk_size):
                chunk.append(data)
            data_queue.put(chunk)

        print(f'Worker terminated {name}', flush=True)


class AsyncThreadIterator(AsyncIterator):
    """ Base class for thread based iterators """

    def __init__(self, make_env: Callable, make_policy: Callable, orig_policy: Policy, chunk_size, make_iter: Callable, name: Optional[str] = None):

        super().__init__(make_env, make_policy, orig_policy, chunk_size, make_iter, name)
        self._policy_queue = q.Queue(maxsize=1)
        self._data_queue = q.Queue(maxsize=1)

        self._thread = Thread(target=AsyncIterator._worker_loop,
                              args=self._worker_args())

    def __iter__(self):
        if not self._started:
            print(f'Starting {self._name}...', flush=True)
            self._thread.start()
            self._send_policy(self._orig_policy.state_dict())
            self._started = True
        return self

    def __del__(self):
        if self._started:
            print(f'Terminating {self._name}...', flush=True)
            self._policy_queue.put(None)
            self._thread.join()
            print(f'Stopped {self._name}', flush=True)


class AsyncProcessIterator(AsyncIterator):
    """ Base class for process based iterators """

    def __init__(self, make_env: Callable, make_policy: Callable, orig_policy: Policy, chunk_size, make_iter: Callable, name: Optional[str] = None):

        super().__init__(make_env, make_policy, orig_policy, chunk_size, make_iter, name)

        self._policy_queue = mp.Queue(maxsize=1)
        self._data_queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(target=AsyncIterator._worker_loop,
                                  args=self._worker_args())

    def __iter__(self):
        if not self._started:
            print(f'Starting {self._name}...', flush=True)
            self._worker.start()
            self._send_policy(self._orig_policy.state_dict())
            self._started = True
        return self

    def __del__(self):
        if self._started:
            print(f'Terminating {self._name}...', flush=True)
            self._worker.terminate()
            self._worker.join()
            print(f'Stopped {self._name}', flush=True)

    def _send_policy(self, policy_data):
        if isinstance(policy_data, dict):
            # for prevent CUDA eror while send via Queue
            for k,v in policy_data.items():
                policy_data[k] = v.to('cpu:0')

        return super()._send_policy(policy_data)
