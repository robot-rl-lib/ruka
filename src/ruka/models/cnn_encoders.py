from typing import Tuple, Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import os
import gym
import torch
import ruka_os.distributed_fs_v2 as dfs_v2
import torchvision.models as tm

from ruka.util.distributed_fs import cached_download

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


class AugmentedNatureCNNDictLikeGray(BaseFeaturesExtractor):
    def __init__(self, input_dims, num_direct_features, image_features_dim: int = 512):
        features_dim = image_features_dim + num_direct_features
        super().__init__(input_dims, features_dim)

        self.num_direct_features = num_direct_features
        self.nature_cnn = NatureCNN(input_dims, image_features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations) -> torch.Tensor:
        other_features = self.flatten(observations['sensor_pad'])[:, :self.num_direct_features]
        return torch.cat([
            self.nature_cnn(torch.cat([observations['gray']], axis=1)),
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


class ResNetEnc(BaseFeaturesExtractor):

    _weight_paths = {'resnet18': 'aux_data/models/resnet18-f37072fd.pth'}

    def __init__(self, input_dims: Tuple[int, int, int],
                       features_dim: int = 512,
                       model: str='resnet18',
                       pretrained: bool = True,
                       imgnet_norm=True,
                       freeze=False,):
        super().__init__(input_dims, features_dim)
        print(f"Make {model} encoder...")
        assert model in self._weight_paths.keys()
        self._resnet = self._make_resnet(model, pretrained)

        self._imgnet_norm = imgnet_norm and self._input_dims[0] == 3
        if self._imgnet_norm:
            assert self._input_dims[0] == 3, 'Only for RGB'
            self.register_buffer('_mean',
                                    torch.tensor([0.485, 0.456, 0.406],
                                            dtype=torch.float32).view(1, -1, 1, 1))
            self.register_buffer('_std',
                                    torch.tensor([0.229, 0.224, 0.225],
                                            dtype=torch.float32).view(1, -1, 1, 1))

        if freeze:
            self.freeze()

    def load_weights(self, model, resnet):
        print('Load weights...')
        local_path = cached_download(self._weight_paths[model], other_dfs = dfs_v2)
        state_dict = torch.load(local_path)
        resnet.load_state_dict(state_dict)

        
    def _make_resnet(self, model, pretrained):
        model_cls = getattr(tm, model)
        resnet = model_cls(pretrained=False)
        if pretrained:
            self.load_weights(model, resnet)

        if self._input_dims[0] != 3:
            first_conv = nn.Conv2d(self._input_dims[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            setattr(resnet, 'conv1', first_conv)

        avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        setattr(resnet, 'avgpool', avgpool)

        fc = nn.Linear(resnet.fc.in_features, self._features_dim)
        setattr(resnet, 'fc', fc)

        return resnet

    def _normalize_img(self, x: torch.Tensor):
        if self._imgnet_norm:
            x.sub(self._mean).div(self._std)
        return x

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self._normalize_img(x)
        x = self._resnet(x).reshape(x.shape[0], -1)
        return x

    def freeze(self):
        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False
        _freeze_module(self._resnet)
        
        if self._imgnet_norm:
            self._mean.requires_grad = False
            self._std.requires_grad = False

        if self._input_dims[0] != 3:
            for p in self._resnet.conv1.parameters():
                p.requires_grad = True
                
        for p in self._resnet.fc.parameters():
            p.requires_grad = True           

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)


        print("Trainable parameters in the encoder:")
        print(trainable_params)      