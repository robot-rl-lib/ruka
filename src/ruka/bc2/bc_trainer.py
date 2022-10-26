from pyparsing import Or
import torch.optim as optim
from collections import OrderedDict
from torch import nn
import numpy as np

from ruka.sac.sac import TorchTrainer
import ruka.pytorch_util as ptu
from .utils import crop_augment, default_collate_fn, numpy_tree_to_torch
import functools


class BCRADTrainer:
    def __init__(
            self,
            policy,
            crop_size=60,
            to_crop=['depth', 'target_segmentation', 'gray'],
            collate_fn=default_collate_fn,
            lr=1e-3,
            optimizer_class=optim.Adam,
    ):
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.bc_criterion = nn.MSELoss()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self._augment = functools.partial(crop_augment, to_crop=to_crop, crop_size=crop_size)
        self._collate_fn = collate_fn

    def train(self, batch):

        """
        Update networks
        """
        # obs = batch['ob']
        if 'action_sequences' in batch:
            del batch['action_sequences']

        for key in batch:
            if key=='action_sequences' and (not batch[key][0]):
                continue
            batch[key] = self._collate_fn(batch[key])
        batch['observation_sequences'] = self._augment(batch['observation_sequences'])
        batch = numpy_tree_to_torch(batch)
        actions = batch['targets']
        

        """
        Sample new actions from states
        """
        pred_actions = self.policy(batch['observation_sequences'], None)

        """
        Policy loss + Update
        """

        policy_loss = self.bc_criterion(pred_actions, actions)


        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        eval_statistics = OrderedDict()
        if self._need_to_update_eval_statistics:
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            self.eval_statistics = eval_statistics
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False


    def get_diagnostics(self):
        stats = OrderedDict([
            ('num train calls', self._n_train_steps_total),
        ])
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
        ]

    @property
    def optimizers(self):
        return [
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict()
        )

    def load(self, snapshot):
        self.policy.load_state_dict(snapshot["policy"])
        self.policy_optimizer.load_state_dict(snapshot["policy_optimizer"])
