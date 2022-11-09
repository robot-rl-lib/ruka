import gym

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Iterator, List
from ruka.pytorch_util import TorchAware


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
    info: dict


@dataclass
class Episode:
    observations: List[Observation]  # L + 1
    actions: List[Action]            # L
    rewards: List[float]             # L
    dones: List[bool]                # L
    infos: List[dict]                # L

    def __len__(self):
        return len(self.actions)

    def __post_init__(self):
        assert len(self.actions) + 1 == len(self.observations)
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.dones)
        assert len(self.actions) == len(self.infos)
        assert not any(self.dones[:-1])

    def append(self, obs: Observation, action: Action, reward: float, done: bool, info: dict):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        
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