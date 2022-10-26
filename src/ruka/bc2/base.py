from typing import Any, List, Generator, Tuple, Iterator
from collections import namedtuple
import torch.nn as nn
import torch 

Observation = Any 
Action = Any 
Reward = float
done = bool 
info = dict
Transition = namedtuple(
    'Transition', 
    ('observation', 'action', 'reward', 'next_observation', 'done', 'info')
)


class Path:
    def __init__(
        self, 
        observations: List[Observation], 
        actions: List[Action], 
        rewards: List[float],
        dones: List[bool],
        infos: List[dict]
        ):

        assert len(actions) + 1 == len(observations), f"{len(actions) + 1} != {len(observations)}"
        assert len(actions) == len(rewards)
        assert len(actions) == len(dones)
        assert len(actions) == len(infos)
        assert (not any(dones[:-1])) 
        # done[-1] may be False too if path is incomplete

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._dones = dones
        self._infos = infos

    def __len__(self):
        return len(self._actions)

    def transition_generator(self) -> Iterator[Transition]:
        for i in range(len(self)):
            yield Transition(
                observation=self._observations[i],
                action=self._actions[i],
                reward=self._rewards[i],
                next_observation=self._observations[i + 1],
                done=self._dones[i],
                info=self._infos[i],
            )

    def is_finished(self):
        return self._dones[-1]

    @property
    def observations(self):
        return self._observations
    @property
    def actions(self):
        return self._actions
    @property
    def rewards(self):
        return self._rewards
    @property
    def dones(self):
        return self._dones
    @property
    def infos(self):
        return self._infos
    


class BaseBuffer:

    def add_path(self, path: Path):
        raise NotImplementedError

    def add_paths(self, paths: List[Path]):
        for path in paths:
            self.add_path(path)

    def get_diagnostics(self):
        return {}

    def random_batch(self, batch_size: int):
        pass


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, input_dims: Tuple[int, int, int], features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._input_dims = input_dims
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations) -> torch.Tensor:
        raise NotImplementedError()