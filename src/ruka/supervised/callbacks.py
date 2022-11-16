import collections
from typing import Dict, Iterator
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

import ruka.util.tensorboard as tb
import ruka.util.distributed_fs as dfs
from ruka.environments.common.env import Episode, TorchAware, Policy
from ruka.types import Dictator

""" Classes that return stateful functions that log something on call
"""


class Callback:
    """ Being called at the end of each train step 
    """
    def __call__(
        self, 
        step: int, 
        batch: Dict[str, Dictator],
        config
        ):
        raise NotImplementedError()
        
class ADPPHCallback(Callback):
    """
    Adjusted Picks Per Hour metrics. Additionaly logs success rate and
    disengagement rate.
    """
    def __init__(
        self, 
        episode_iterator: Iterator[Episode], 
        calculate_every: int = 200_000,
        for_episodes: int = 100, 
        idle_after_broken: float = 120,
        prefix: str = ''
        ):
        self.episode_iterator = episode_iterator
        self.calculate_every = calculate_every
        self.for_episodes = for_episodes
        self.idle_after_broken = idle_after_broken
        self.prefix = prefix
        

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self.calculate_every) != 0:
            return

        elapsed_time = 0
        npicks = 0
        nbreaks = 0
        successess = []
        ep_times = []
        
        for _ in range(self.for_episodes):
            ep = next(self.episode_iterator)
            elapsed_time += np.sum([info['transition_time'] for info in ep.infos])
            ep_times.append(len(ep.actions))
            successess.append(ep.infos[-1]['is_success'])
            if ep.infos[-1]['is_success']:
                npicks += 1
            else:
                nbreaks += 1
                elapsed_time += self.idle_after_broken

            print(f"SUCCESSFUL: {ep.infos[-1]['is_success']}, LEN: {len(ep.actions)}")

        metric_output = dict(
                disengagement_rate=(nbreaks / elapsed_time * 3600),
                picks_per_hour=(npicks / elapsed_time * 3600),
                success_rate=np.mean(successess),
                mean_path_len=np.mean(ep_times),
            )

        tb.step(step)
        for k, v in metric_output.items():
            tb.scalar(self.prefix + k, v)
            print(self.prefix + k, "=", v)
            
class SRCallback(Callback):
    """
    Calculates and logs Success Rate using window of episodes
    """
    def __init__(
        self,
        paths_iterator: Iterator[Episode], 
        calculate_every: int = 1,
        paths_per_run: int = 1,
        window_size: int = 100,
        prefix: str = ''
        ):
        self.paths_iterator = paths_iterator
        self.calculate_every = calculate_every
        self.paths_per_run = paths_per_run
        self.window = collections.deque(maxlen=window_size)
        self.prefix = prefix

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self.calculate_every) != 0:
            return
        n_paths = 0

        while n_paths < self.paths_per_run:
            path = next(self.paths_iterator)

            self.window.append(path.infos[-1]['is_success'])
            n_paths += 1

        tb.step(step)
        if len(self.window) > 0:
            tb.scalar(self.prefix + 'success_rate', np.mean(self.window))
            print(f'Step {step} success rate = {np.mean(self.window)}')


class ValLossCallback(Callback):
    """ Calculates and logs losses on validation set.
    """
    def __init__(
        self, 
        loss: nn.Module, 
        test_data: Iterator, 
        n_batches: int,
        calculate_every: int,
        prefix: str = ''
        ):
        self._test_data = test_data 
        self._n_batches = n_batches
        self._loss = loss
        self._calculate_every = calculate_every
        self.prefix = prefix

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self._calculate_every) != 0:
            return
        loss = 0
        with torch.no_grad():
            for _ in range(self._n_batches):
                batch = next(self._test_data)
                loss += self._loss(batch).cpu().detach().item()
    
        tb.step(step)
        tb.scalar(self.prefix + 'Validation loss', loss / self._n_batches)
        print(f'Step {step} validation loss = {loss / self._n_batches}')
        

class SaveCallback(Callback):
    """ Calculates and logs losses on validation set.
    """
    def __init__(
        self, 
        model: TorchAware, 
        every: int,
        save_to: str = '',
        prefix: str = '',
        ):
        self._model = model
        self._prefix = prefix
        self._every = every
        self._save_to = save_to

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self._every) != 0:
            return
        pathlib.Path(self._save_to).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self._save_to, self._prefix + f"{step}.pth")
        torch.save(self._model.state_dict(), filename)
        dfs.upload_maybe(filename)

class SavePolicy(SaveCallback):
    def __init__(self, policy: Policy, every: int, save_to: str = ''):
        super().__init__(policy, every, save_to, prefix='policy_')