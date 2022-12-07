from collections import OrderedDict
from typing import Optional
from torch import nn
import numpy as np

import ruka.pytorch_util as ptu
from ruka.util.augmentation import crop_augment, center_crop
from .base import Loss
import functools
import ruka.util.tensorboard as tb
import torch

class BCRADLoss(Loss):
    def __init__(
            self,
            model: nn.Module,
            crop_size=60,
            to_crop=['depth', 'target_segmentation', 'gray'],
            log_stats_every: Optional[int] = None,
            log_hist_every: Optional[int] = None,
    ):  
        super().__init__()
        self._model = model
        self._bc_criterion = nn.MSELoss()
        self._augment = functools.partial(crop_augment, to_crop=to_crop, crop_size=crop_size)  
        self._augment_eval = functools.partial(center_crop, to_crop=to_crop, crop_size=crop_size)  
        self._log_stats_every = log_stats_every
        self._log_hist_every = log_hist_every
        self._stats = None

    def forward(self, batch) -> torch.Tensor:
        """ Expecting shapes:
                observation_sequence_batch: key -> (batch, seqlen, *obs[key]shape)
                action_sequence_batch: (batch, seqlen, *actshape)

            Outputs total loss
        """
        # augmentation
        if self.training:
            batch['observation_sequence_batch'] = self._augment(batch['observation_sequence_batch'])
        else:
            batch['observation_sequence_batch'] = self._augment_eval(batch['observation_sequence_batch'])

        batch = ptu.numpy_tree_to_torch(batch)

        # target selection
        actions = batch['action_sequence_batch'][:, -1, ...]
        batch['action_sequence_batch'] = batch['action_sequence_batch'][:, :-1, ...]
        
        # loss computation
        pred_actions = self._model(batch)
        bc_loss = self._bc_criterion(pred_actions, actions)

        # saving stats
        if self._stats is None:
            self._stats = OrderedDict()
            self._stats['scalars'] = OrderedDict()
            self._stats['scalars']['BC Loss'] = np.mean(ptu.get_numpy(bc_loss))

            self._stats['hist'] = OrderedDict()
            for act_num in range(pred_actions.shape[-1]):
                self._stats['hist'][f'pred_action[{act_num}]'] = pred_actions[..., act_num]

        return bc_loss

    def log_stats(self, step: int, prefix: Optional[str] = None):
        # logging
        if (self._log_stats_every is None) or \
            (self._stats is None):
            return
        
        if step % self._log_stats_every == 0:
            tb.step(step)
            prefix = prefix if prefix is not None else ''
            for k, v in self._stats['scalars'].items():
                tb.scalar(prefix + k, v)

        if (self._log_hist_every is not None) and step % self._log_hist_every == 0:
            for k, v in self._stats['hist'].items():
                tb.add_histogram(prefix + k, v)

        self._stats = None
