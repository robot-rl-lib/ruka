from .mlp import Mlp
import abc
from ruka.sac.util import elem_or_tuple_to_numpy, torch_ify
import ruka.pytorch_util as ptu
import numpy as np
import torch.nn as nn
import torch
from .distributions import Delta, TanhNormal, DistributionGenerator
from .cnn_encoders import BaseFeaturesExtractor
from ruka.observation import Observation

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass

class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs):
        actions = self.get_actions(obs.to_pytorch() if isinstance(obs, Observation) else obs[None])
        return actions[0, :]

    def get_actions(self, obs_np):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist


class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

class EncoderTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self, encoder: BaseFeaturesExtractor, hidden_sizes, action_dim, std=None, init_w=0.001, **kwargs):
        super().__init__(hidden_sizes, encoder.features_dim, action_dim, std, init_w, **kwargs)
        self.encoder = encoder
    def forward(self, obs):
        encoded_obs = self.encoder(obs)
        return super().forward(encoded_obs)

class SharedEncoderTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self, encoder: BaseFeaturesExtractor, hidden_sizes, action_dim, std=None, init_w=0.001, **kwargs):
        super().__init__(hidden_sizes, encoder.features_dim, action_dim, std, init_w, **kwargs)
        self.encoder = encoder
    def forward(self, obs):
        with torch.no_grad():
            encoded_obs = self.encoder(obs)
        return super().forward(encoded_obs)

class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())


class TorchDeterministicPolicy(
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs):
        actions = self.get_actions(obs.to_pytorch() if isinstance(obs, Observation) else obs[None])
        return actions[0, :]

    def get_actions(self, obs_np):
        actions = self._get_actions_from_np(obs_np)
        return elem_or_tuple_to_numpy(actions)

    def _get_actions_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        actions = self(*torch_args, **torch_kwargs)
        return actions

class SimpeDeterministicPolicy(Mlp, TorchDeterministicPolicy):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )


class EncoderSimpeDeterministicPolicy(SimpeDeterministicPolicy):
    def __init__(self, encoder: BaseFeaturesExtractor, hidden_sizes, action_dim, init_w=0.001, **kwargs):
        super().__init__(hidden_sizes, encoder.features_dim, action_dim, init_w, **kwargs)
        self.encoder = encoder
    def forward(self, obs):
        encoded_obs = self.encoder(obs)
        return super().forward(encoded_obs)


class EncoderSimpeDeterministicPolicyHeuristic(EncoderSimpeDeterministicPolicy):
    def __init__(self, encoder: BaseFeaturesExtractor, hidden_sizes, action_dim, init_w=0.001, heuristics=[], **kwargs):
        super().__init__(encoder, hidden_sizes, action_dim, init_w, **kwargs)
        self.encoder = encoder
        self.apply_heuristics = False
        self.heuristics = heuristics

    def forward(self, obs):
        action = super().forward(obs)

        if self.apply_heuristics:
            assert len(action.shape) == 2
            assert action.shape[0] == 1, 'Applying heuristics is only supported for single action, not batch'
            for h in self.heuristics:
                action = h(action, obs)

        return action

        
class ZeroPolicy:
    def __init__(self, env):
        self._action_shape = env.action_space.shape
    def get_actions(self, *args, **kwargs):
        return np.zeros(self._action_shape, dtype=np.float32)[None]
    def reset(self):
        pass
