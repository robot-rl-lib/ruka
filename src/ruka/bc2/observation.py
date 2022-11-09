import numpy as np
from typing import Dict
import ruka.pytorch_util as ptu
import gym

class Observation(dict):
    def __init__(self, state: Dict[str, np.array] = {}):
        super().__init__(state)

    def to_pytorch(self, device=None):
        out = Observation()
        for k, v in self.items():
            out[k] = ptu.from_numpy(v, other_device=device)
        return out
    
    def add_dim(self):
        out = Observation()
        for k, v in self.items():
            out[k] = v[None]
        return out

    def get_space(self):
        assert len(self) > 0

        space = dict()
        for key, value in self.items():
            space[key] = gym.spaces.Box(low=0, high=1, shape=value.shape)

        return gym.spaces.Dict(space)
