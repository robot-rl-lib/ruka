import gym
import numpy as np

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Dict
from ruka.pytorch_util import TorchAware
from ruka.util.nested_dict import NestedDict


Observation = NestedDict
Action = NestedDict
StepInfo = namedtuple('StepInfo', ('observation', 'reward', 'done', 'info'))


EnvInfo = Dict[str, Any]
""" EnvInfo must contain next keys:
        - transition_time - how much time since prev step passed
        - is_time_limit - if episode was terminated by the time limit

    Optional standartized:
        - is_success - flag for success
"""

class Env:
    def reset(self) -> Observation:
        raise NotImplementedError()

    def step(self, action: Action) -> StepInfo:
        raise NotImplementedError()

    @property
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @property
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    def close(self):
        pass


class Policy(TorchAware):
    def reset(self):
        pass

    def get_action(self, observation: Observation) -> Action:
        raise NotImplementedError()


@dataclass
class Transition:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    done: bool
    info: EnvInfo


@dataclass
class Episode:
    observations: List[Observation] = field(default_factory=list)    # L + 1
    actions: List[Action] = field(default_factory=list)              # L
    rewards: List[float] = field(default_factory=list)               # L
    dones: List[bool] = field(default_factory=list)                  # L
    infos: List[EnvInfo] = field(default_factory=list)               # L
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.actions)

    def __post_init__(self):
        assert len(self.actions) == 0 or len(self.actions) + 1 == len(self.observations)
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.dones)
        assert len(self.actions) == len(self.infos)
        assert not any(self.dones[:-1])

    @property
    def transitions(self) -> Iterator[Transition]:
        for i in range(len(self)):
            yield Transition(
                observation=self.observations[i],
                action=self.actions[i],
                reward=self.rewards[i],
                next_observation=self.observations[i + 1],
                done=self.dones[i],
                info=self.infos[i]
            )

