from collections import OrderedDict
from typing import Optional, List, Callable
from torch import nn
import numpy as np

import ruka.pytorch_util as ptu
from ruka.util.augmentation import crop_augment, center_crop
from .base import Loss
import functools
import ruka.util.tensorboard as tb
import torch

class BCWithCELoss(Loss):
    def __init__(
            self,
            model: nn.Module,
            con_action_nums,
            crop_size=60,
            to_crop=['depth', 'target_segmentation', 'gray'],
            bin_action_slices: List[slice] = [],
            bin_action_weights: Optional[List[float]] = None,
            ce_loss_weight = 1,
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

        self._con_action_nums = con_action_nums
        self._bin_action_slices = bin_action_slices
        self._bin_action_weights = bin_action_weights
        self._ce_loss_weight = ce_loss_weight

        self._ce_criterion = []
        for i in range(len(self._bin_action_slices)):
            if bin_action_weights:
                assert len(bin_action_weights) == len(self._bin_action_slices), (len(bin_action_weights), len(self._bin_action_slices))
                self._ce_criterion.append(nn.CrossEntropyLoss(weight=torch.Tensor(bin_action_weights[i]).to(ptu.device)))
                print('Add CrossEntropyLoss weights', bin_action_weights[i])
            else: 
                self._ce_criterion.append(nn.CrossEntropyLoss())

        self._step = 0

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

        con_actions = actions[:, :self._con_action_nums]
        
        
        # loss computation
        pred_actions, pred_bin_actions = self._model(batch)
        bc_loss = self._bc_criterion(pred_actions, con_actions)

        ce_losses = []
        for i, bin_slice in enumerate(self._bin_action_slices):
            bin_target = actions[:, self._con_action_nums:][:,bin_slice]
            bin_pred = pred_bin_actions[:,bin_slice]
            ce_loss = self._ce_criterion[i](bin_pred, bin_target)
            ce_losses.append(ce_loss)

        ce_loss = torch.stack(ce_losses, dim=-1).sum(dim=-1)

        if self._step % self._log_stats_every == 0:
            tb.scalar('training/bc_loss', np.mean(ptu.get_numpy(bc_loss)))
            tb.scalar('training/ce_loss', np.mean(ptu.get_numpy(ce_loss)))

            for i, ce_loss_part in enumerate(ce_losses):
                tb.scalar(f'training/ce_loss_{i}', np.mean(ptu.get_numpy(ce_loss_part)))

        if self._step % self._log_hist_every == 0:
            for act_num in range(pred_actions.shape[-1]):
                tb.add_histogram(f'pred_action[{act_num}]', pred_actions[..., act_num])

        self._step += 1
        losses = {'loss': bc_loss + (self._ce_loss_weight * ce_loss),
                'bc_loss': bc_loss,
                'ce_loss': ce_loss,}
        for i, ce_loss_part in enumerate(ce_losses):
            losses[f'ce_loss_{i}'] = ce_loss_part
        return losses