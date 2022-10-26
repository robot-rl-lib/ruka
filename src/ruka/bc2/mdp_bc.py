from typing import List, Optional, Callable
import torch.nn as nn
from .base import Action, BaseFeaturesExtractor, Observation
import numpy as np
from ruka.models.mlp import Mlp
import ruka.pytorch_util as ptu
import torch
from .utils import default_collate_fn, numpy_tree_to_torch

class MDPPolicy(nn.Module):
    def __init__(
        self, 
        observation_encoder: BaseFeaturesExtractor,
        hidden_sizes: List[int],
        action_size: int,
        ):
        super().__init__()
        self._observation_encoder = observation_encoder
        self._mlp = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=observation_encoder.features_dim,
            output_size=action_size,
        )
    def forward(
        self, 
        observation_sequences_stacked: dict, 
        action_sequences_stacked
        ):
        encoded_obs = self._observation_encoder(observation_sequences_stacked)
        actions = self._mlp(encoded_obs)
        return actions

    def get_action(self, obs_seq: List[Observation], act_seq):
        obs_stacked = self._default_collate_fn([obs_seq])
        obs_stacked = numpy_tree_to_torch(obs_stacked)
        with torch.no_grad():
            encoded_obs = self._observation_encoder(obs_stacked)
            actions = self._mlp(encoded_obs)
            actions = actions.cpu().detach().numpy()
        return actions[0]

    @staticmethod
    def _default_collate_fn(sequence_list):
        # if not isinstance(sequence_list[0][0], dict):
        #     return ptu.from_numpy(np.array(sequence_list))
        # out = dict()
        # for key in sequence_list[0][0].keys():
        #     val = [[x[key] for x in sequence] for sequence in sequence_list]
        #     out[key] = ptu.from_numpy(np.array(val)) # BS, T, *DIMS 
        # return out
        return default_collate_fn(sequence_list)

class StatefulPolicy:
    def __init__(self, policy: MDPPolicy):
        self.policy = policy
        self.actions = []
        self.observations = []
    def get_action(self, observation):
        self.observations.append(observation)
        return self.policy.get_action(self.observations, self.actions)
    def reset(self):
        self.actions = []
        self.observations = []
