# import numpy as np
# import tensorflow as tf

# from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

# def create_augmented_nature_cnn(num_direct_features):
#     """
#     Create and return a function for augmented_nature_cnn
#     used in stable-baselines.

#     num_direct_features tells how many direct features there
#     will be in the image.
#     """

#     def augmented_nature_cnn(scaled_images, **kwargs):
#         """
#         Copied from stable_baselines policies.py.
#         This is nature CNN head where last channel of the image contains
#         direct features.

#         :param scaled_images: (TensorFlow Tensor) Image input placeholder
#         :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#         :return: (TensorFlow Tensor) The CNN output layer
#         """
#         activ = tf.nn.relu

#         # Take last channel as direct features
#         other_features = tf.contrib.slim.flatten(scaled_images[..., -1])
#         # Take known amount of direct features, rest are padding zeros
#         other_features = other_features[:, :num_direct_features]

#         scaled_images = scaled_images[..., :-1]

#         layer_1 = activ(conv(scaled_images, 'cnn1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
#         layer_2 = activ(conv(layer_1, 'cnn2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
#         layer_3 = activ(conv(layer_2, 'cnn3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#         layer_3 = conv_to_fc(layer_3)

#         # Append direct features to the final output of extractor
#         img_output = activ(linear(layer_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))
#         concat = tf.concat((img_output, other_features), axis=1)

#         return concat

#     return augmented_nature_cnn



from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym
import torch

class AugmentedNatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, num_direct_features, image_features_dim: int = 512):
        features_dim = image_features_dim + num_direct_features
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(observation_space, check_channels=False), (
        #     "You should use NatureCNN "
        #     f"only with images not with {observation_space}\n"
        #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
        #     "If you are using a custom environment,\n"
        #     "please check it using our env checker:\n"
        #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        # )
        self.num_direct_features = num_direct_features
        n_input_channels = observation_space.shape[0] - 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten = nn.Flatten()
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None][:, :-1, ...]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, image_features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        other_features = observations[:, -1, ...]
        other_features = self.flatten(other_features)
        other_features = other_features[:, :self.num_direct_features]
        return torch.cat([
            self.linear(self.cnn(observations[:, :-1, ...])), 
            other_features,
            ], axis=-1)

