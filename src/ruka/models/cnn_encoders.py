
from typing import Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym
import torch


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


class NatureCNN(BaseFeaturesExtractor):
    def __init__(self, input_dims: Tuple[int, int, int], features_dim: int = 512):
        super().__init__(input_dims, features_dim)
        n_input_channels = input_dims[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
            n_flatten = self.cnn(torch.zeros((1,) + input_dims).float()).shape[1]
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class AugmentedNatureCNN(BaseFeaturesExtractor):
    """ LAST CHANNEL HAS TO CONTAIN DIRECT FEATURES
    """
    def __init__(self, input_dims, num_direct_features, image_features_dim: int = 512):
        features_dim = image_features_dim + num_direct_features
        super().__init__(input_dims, features_dim)
        self.num_direct_features = num_direct_features
        nature_cnn_inputs = (input_dims[0]-1,) + input_dims[1:]
        self.nature_cnn = NatureCNN(nature_cnn_inputs, image_features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        other_features = self.flatten(observations[:, -1, ...])[:, :self.num_direct_features]
        return torch.cat([
            self.nature_cnn(observations[:, :-1, ...]),
            other_features,
            ], axis=-1)


class AugmentedNatureCNNDictLike(BaseFeaturesExtractor):
    def __init__(self, input_dims, num_direct_features, image_features_dim: int = 512):
        features_dim = image_features_dim + num_direct_features
        super().__init__(input_dims, features_dim)

        self.num_direct_features = num_direct_features
        self.nature_cnn = NatureCNN(input_dims, image_features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations) -> torch.Tensor:
        other_features = self.flatten(observations['sensor_pad'])[:, :self.num_direct_features]
        return torch.cat([
            self.nature_cnn(torch.cat([observations['depth'], observations['mask']], axis=1)),
            other_features,
            ], axis=-1)


class LightCNN(BaseFeaturesExtractor):
    def __init__(self, input_dims: Tuple[int, int, int], features_dim: int = 512):
        super().__init__(input_dims, features_dim)
        n_input_channels = input_dims[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
            n_flatten = self.cnn(torch.zeros((1,) + input_dims).float()).shape[1]
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.ln = nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.ln(self.linear(self.cnn(observations)))



class AugmentedLightEncoder(BaseFeaturesExtractor):
    """ LAST CHANNEL HAS TO CONTAIN DIRECT FEATURES
    """
    def __init__(self, input_dims, num_direct_features, image_features_dim: int = 50):
        features_dim = image_features_dim + num_direct_features
        super().__init__(input_dims, features_dim)
        self.num_direct_features = num_direct_features
        nature_cnn_inputs = (input_dims[0]-1,) + input_dims[1:]
        self.nature_cnn = LightCNN(nature_cnn_inputs, image_features_dim)
        self.flatten = nn.Flatten()


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        other_features = self.flatten(observations[:, -1, ...])[:, :self.num_direct_features]
        return torch.cat([
            self.nature_cnn(observations[:, :-1, ...]),
            other_features,
            ], axis=-1)