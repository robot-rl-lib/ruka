from collections import OrderedDict, namedtuple
from typing import Tuple, Iterable, Any, Union, Optional
import abc

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import ruka.pytorch_util as ptu
from ruka.training.eval_util import create_stats_ordered_dict
from .util import add_prefix, np_to_pytorch_batch
from ruka.util.scheduler import Scheduler

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qfs_loss alpha_loss',
)

class Trainer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[Any]:
        pass


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qfs,
            target_qfs,
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            max_grad_norm=1e6,

            soft_target_tau=1e-2,
            target_update_period=1,
            critic_coef=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            action_loss: Optional[Union[float, Scheduler]] = None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.critic_coef = critic_coef
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.action_loss = action_loss
        if self.action_loss is not None and not isinstance(self.action_loss, Scheduler):
            self.action_loss = Scheduler({0: self.action_loss})

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )
        self.max_grad_norm = max_grad_norm

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):

        grad_norm_stats = OrderedDict()
        """
        Update networks
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Sample new actions from states
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        """
        Alpha Loss + Update
        """
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            total_norm = nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            grad_norm_stats['alpha'] = total_norm.item()
            self.alpha_optimizer.step()

        """
        QF Loss + Update
        """
        q1_pred, q2_pred = self.qfs(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            *self.target_qfs(next_obs, new_next_actions)
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        self.qfs_optimizer.zero_grad()
        (self.critic_coef * (qf1_loss + qf2_loss)).backward()
        total_norm = nn.utils.clip_grad_norm_(self.qfs.parameters(), self.max_grad_norm)
        grad_norm_stats['qfs'] = total_norm.item()
        self.qfs_optimizer.step()

        """
        Policy loss + Update
        """

        q_new_actions = torch.min(*self.qfs(obs, new_obs_actions))
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        if self.action_loss is not None:
            action_loss = self.action_loss.value(self._n_train_steps_total) * torch.mean(new_obs_actions**2)
            policy_loss += action_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        grad_norm_stats['policy'] = total_norm.item()
        self.policy_optimizer.step()

        eval_statistics = OrderedDict()
        if self._need_to_update_eval_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.action_loss is not None:
                eval_statistics['Action Loss'] = np.mean(ptu.get_numpy(action_loss))
            
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()


        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = eval_statistics
            self.eval_statistics.update(add_prefix(grad_norm_stats, 'grad_norm/'))
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):

        ptu.soft_update_from_to(
            self.qfs, self.target_qfs, self.soft_target_tau
        )

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qfs_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy.state_dict(),
            qfs=self.qfs.state_dict(),
            target_qfs=self.target_qfs.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
            qfs_optimizer=self.qfs_optimizer.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict()
        )

    def load(self, snapshot):
        self.policy.load_state_dict(snapshot["policy"])
        self.qfs.load_state_dict(snapshot["qfs"])
        self.target_qfs.load_state_dict(snapshot["target_qfs"])
        self.alpha_optimizer.load_state_dict(snapshot["alpha_optimizer"])
        self.qfs_optimizer.load_state_dict(snapshot["qfs_optimizer"])
        self.policy_optimizer.load_state_dict(snapshot["policy_optimizer"])
