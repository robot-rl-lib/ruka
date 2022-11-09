import collections
from typing import Dict, Iterator

import numpy as np
import torch
import torch.nn as nn

import ruka.util.tensorboard as tb
from ruka.environments.common.env import Episode, Transition
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
        transitions_iterator: Iterator[Transition], 
        calculate_every: int = 200_000,
        for_time: int = 3600, 
        idle_after_broken: float = 120
        ):
        self.transitions_iterator = transitions_iterator
        self.calculate_every = calculate_every
        self.for_time = for_time
        self.idle_after_broken = idle_after_broken

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self.calculate_every) != 0:
            return

        time = 0
        npicks = 0
        nbreaks = 0
        successess = []

        while time < self.for_time:
            transition = next(self.transitions_iterator)
            time += transition.observation['transition_time']
            if transition.done:
                successess.append(transition.info['is_success'])
                if not transition.info['is_success']:
                    time += self.idle_after_broken
                    nbreaks += 1
                else:
                    npicks += 1

                print(f"EVALUATION SECONDS: {time.item():.1f}/{self.for_time:.1f}")
        
        metric_output = dict(
                disengagement_rate=(nbreaks / time * 3600),
                picks_per_hour=(npicks / time * 3600),
                success_rate=np.mean(successess)
            )
        tb.step(step)
        for k, v in metric_output.items():
            tb.scalar(k, v)


class SRCallback(Callback):
    """
    Calculates and logs Success Rate using window of episodes
    """
    def __init__(
        self,
        paths_iterator: Iterator[Episode], 
        calculate_every: int = 1,
        paths_per_run: int = 1,
        window_size: int = 100
        ):
        self.paths_iterator = paths_iterator
        self.calculate_every = calculate_every
        self.paths_per_run = paths_per_run
        self.window = collections.deque(maxlen=window_size)

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
            tb.scalar('success_rate', np.mean(self.window))
            print(f'Step {step} success rate = {np.mean(self.window)}')


class ValLossCallback(Callback):
    """ Calculates and logs losses on validation set.
    """
    def __init__(
        self, 
        loss: nn.Module, 
        test_data: Iterator, 
        n_batches: int,
        calculate_every: int
        ):
        self._test_data = test_data 
        self._n_batches = n_batches
        self._loss = loss
        self._calculate_every = calculate_every

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
        tb.scalar(f'Validation loss', loss / self._n_batches)
        print(f'Step {step} validation loss = {loss / self._n_batches}')
        

    