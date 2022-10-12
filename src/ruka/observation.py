import numpy as np
from typing import Dict, List
import gym
import copy
import ruka.pytorch_util as ptu
import torch

class Observation(dict):
    def __init__(self, state: Dict[str, np.array] = {}):
        super().__init__(state)

    def select_by_index(self, idx):
        result = Observation()
        for key, value in self.items():
            result[key] = value[idx][None] if isinstance(idx, int) else value[idx]
        return result

    def __len__(self):
        return len(next(iter(self.values())))

    def get_space(self):
        assert len(self) > 0

        space = dict()
        for key, value in self.items():
            space[key] = gym.spaces.Box(low=0, high=1, shape=value.shape)

        return gym.spaces.Dict(space)

    def to_legacy_format(self):
        if isinstance(self['depth'], np.ndarray):
            return np.concatenate([self['depth'], self['mask'], self['sensor_pad']], axis=1)
        else:
            return torch.cat([self['depth'], self['mask'], self['sensor_pad']], axis=1)

    def to_pytorch(self):
        result = Observation()
        for k, v in self.items():
            result[k] = ptu.from_numpy(v).float()
        return result

    @staticmethod
    def stack(observations):
        result = Observation()
        for key in observations[0].keys():
            result[key] = np.concatenate([o[key] for o in observations], axis=0)
        return result
