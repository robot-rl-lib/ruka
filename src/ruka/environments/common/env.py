import gym

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Iterator, List


Observation = Any
Action = Any
StepInfo = namedtuple('StepInfo', ('observation', 'reward', 'done', 'info'))


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


class Policy:
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
    info: dict


@dataclass
class Path:
    observations: List[Observation]  # L + 1
    actions: List[Action]            # L
    rewards: List[float]             # L
    infos: List[Any]                 # L

    def __len__(self):
        return len(self.actions)

    def __post_init__(self):
        assert len(self.actions) + 1 == len(self.observations)
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.infos)

    def __len__(self):
        return len(self._actions)

    @property
    def transitions(self) -> Iterator[Transition]:
        for i in range(len(self)):
            yield Transition(
                observation=self._observations[i],
                action=self._actions[i],
                reward=self._rewards[i],
                next_observation=self._observations[i + 1],
                info=self._infos[i]
            )