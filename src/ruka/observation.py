import numpy as np
from typing import Dict, List
import gym
import copy
import ruka.pytorch_util as ptu
import torch
import enum

class Observe(enum.Enum):
    DEPTH = 'depth'
    RGB = 'rgb'
    GRAY = 'gray'
    TARGET_SEGMENTATION = "mask"
    ROBOT_POS = "robot_pos"
    GRIPPER = "gripper"
    SENSOR_PAD = 'sensor_pad'
    GRIPPER_OPEN = 'gripper_open'
    TRACKER_OBJECT_BBOX = 'tracker_object_bbox'


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
        if isinstance(self[Observe.DEPTH.value], np.ndarray):
            return np.concatenate([self[Observe.DEPTH.value],
                                   self[Observe.TARGET_SEGMENTATION.value],
                                   self[Observe.SENSOR_PAD.value]], axis=1)
        else:
            return torch.cat([self[Observe.DEPTH.value],
                              self[Observe.TARGET_SEGMENTATION.value],
                              self[Observe.SENSOR_PAD.value]], axis=1)

    def to_pytorch(self, other_device=None):
        result = Observation()
        for k, v in self.items():
            result[k] = ptu.from_numpy(v, other_device=other_device).float()
        return result

    @staticmethod
    def stack(observations):
        result = Observation()
        for key in observations[0].keys():
            result[key] = np.concatenate([o[key] for o in observations], axis=0)
        return result
