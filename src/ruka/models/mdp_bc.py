from typing import List
import torch.nn as nn
from .cnn_encoders import BaseFeaturesExtractor
from ruka.models.mlp import Mlp
import torch
from ruka.util.collate import _collate_sequence_list
import ruka.pytorch_util as ptu
from ruka.types import Dictator
from ruka.environments.common.env import Policy
from ruka.environments.common.env import Observation

class MDPModel(nn.Module):
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
    def forward(self, batch: Dictator):
        encoded_obs = self._observation_encoder(batch['observation_sequence_batch'])
        actions = self._mlp(encoded_obs)
        return actions

    def get_action(self, obs_seq: List[Observation], act_seq):
        obs_stacked = self._default_collate_fn([obs_seq])
        obs_stacked = ptu.numpy_tree_to_torch(obs_stacked)
        with torch.no_grad():
            encoded_obs = self._observation_encoder(obs_stacked)
            actions = self._mlp(encoded_obs)
            actions = actions.cpu().detach().numpy()
        return actions[0]

    @staticmethod
    def _default_collate_fn(sequence_list):
        return _collate_sequence_list(sequence_list)

class StatefulPolicy(Policy):

    def __init__(self, policy: MDPModel):
        self.policy = policy
        self.actions = []
        self.observations = []

    def get_action(self, observation):
        self.observations.append(observation)
        return self.policy.get_action(self.observations, self.actions)

    def reset(self):
        self.actions = []
        self.observations = []
    
    @property
    def model(self) -> MDPModel:
        return self.policy
    
    def to(self, destination):
        self.policy = self.policy.to(destination)

    def state_dict(self):
        return self.policy.state_dict()

    def load_state_dict(self, state_dict):
        return self.policy.load_state_dict(state_dict)
