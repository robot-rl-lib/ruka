import collections
from typing import Dict, Iterator, Callable, Optional
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

import ruka.util.tensorboard as tb
import ruka.util.distributed_fs as dfs
from ruka.environments.common.env import Episode, TorchAware, Policy
from ruka.types import Dictator
from ruka.vis.batch_viz import viz_batch_img, get_batch_statistics

""" Classes that return stateful functions that log something on call
"""


class Callback:
    """ Being called at the end of each train step 
    """
    def __call__(
        self, 
        step: int, 
        batch: Dict[str, Dictator],
        config
        ):
        raise NotImplementedError()
        
class ADPPHCallback(Callback):
    """
    Adjusted Picks Per Hour metrics. Additionaly logs success rate and
    disengagement rate.
    """
    def __init__(
        self, 
        episode_iterator: Iterator[Episode], 
        calculate_every: int = 200_000,
        for_episodes: int = 100, 
        idle_after_broken: float = 120,
        prefix: str = ''
        ):
        self.episode_iterator = episode_iterator
        self.calculate_every = calculate_every
        self.for_episodes = for_episodes
        self.idle_after_broken = idle_after_broken
        self.prefix = prefix
        

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self.calculate_every) != 0:
            return

        elapsed_time = 0
        npicks = 0
        nbreaks = 0
        successess = []
        ep_times = []
        
        for _ in range(self.for_episodes):
            ep = next(self.episode_iterator)
            elapsed_time += np.sum([info['timestamp_finish_step'] - info['timestamp_start_step'] for info in ep.infos])
            ep_times.append(len(ep.actions))
            successess.append(ep.infos[-1]['is_success'])
            if ep.infos[-1]['is_success']:
                npicks += 1
            else:
                nbreaks += 1
                elapsed_time += self.idle_after_broken

            print(f"SUCCESSFUL: {ep.infos[-1]['is_success']}, LEN: {len(ep.actions)}")

        metric_output = dict(
                disengagement_rate=(nbreaks / elapsed_time * 3600),
                picks_per_hour=(npicks / elapsed_time * 3600),
                success_rate=np.mean(successess),
                mean_path_len=np.mean(ep_times),
            )

        tb.step(step)
        for k, v in metric_output.items():
            tb.scalar(self.prefix + k, v)
            print(self.prefix + k, "=", v)
            
class SRCallback(Callback):
    """
    Calculates and logs Success Rate using window of episodes
    """
    def __init__(
        self,
        paths_iterator: Iterator[Episode], 
        calculate_every: int = 1,
        paths_per_run: int = 1,
        window_size: int = 100,
        prefix: str = ''
        ):
        self.paths_iterator = paths_iterator
        self.calculate_every = calculate_every
        self.paths_per_run = paths_per_run
        self.window = collections.deque(maxlen=window_size)
        self.prefix = prefix

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self.calculate_every) != 0:
            return
        n_paths = 0

        while n_paths < self.paths_per_run:
            path = next(self.paths_iterator)

            self.window.append(path.infos[-1]['is_success'])
            n_paths += 1

        tb.step(step)
        if len(self.window) > 0:
            tb.scalar(self.prefix + 'success_rate', np.mean(self.window))
            print(f'Step {step} success rate = {np.mean(self.window)}')


class ValLossCallback(Callback):
    """ Calculates and logs losses on validation set.
    """
    def __init__(
        self, 
        loss: nn.Module, 
        test_data: Iterator, 
        calculate_every: int,
        n_batches: Optional[int] = None,
        prefix: str = ''
        ):
        self._test_data = test_data 
        self._n_batches = n_batches
        self._loss = loss
        self._calculate_every = calculate_every
        self.prefix = prefix

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self._calculate_every) != 0:
            return
        loss = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_data):
                loss += self._loss(batch).cpu().detach().item()
                if self._n_batches and self._n_batches == batch_num + 1:
                    break                

        tb.step(step)
        tb.scalar(self.prefix + 'Validation loss', loss / (batch_num + 1))
        print(f'Step {step} validation loss = {loss / (batch_num + 1)}')
        

class SaveCallback(Callback):
    """ Calculates and logs losses on validation set.
    """
    def __init__(
        self, 
        model: TorchAware, 
        every: int,
        save_to: str = '',
        prefix: str = '',
        ):
        self._model = model
        self._prefix = prefix
        self._every = every
        self._save_to = save_to

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):
        if ((step + 1) % self._every) != 0:
            return
        pathlib.Path(self._save_to).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self._save_to, self._prefix + f"{step}.pth")
        torch.save(self._model.state_dict(), filename)
        dfs.upload_maybe(filename)

class SavePolicy(SaveCallback):
    def __init__(self, policy: Policy, every: int, save_to: str = ''):
        super().__init__(policy, every, save_to, prefix='policy_')


class VisBatchCallback(Callback):
    
    def __init__(
        self, 
        every: int,
        nbins: int = 20,
        process_act_fn: Callable = lambda x: x,
        process_bbox_fn: Optional[Callable] = None,
        process_point_fn: Optional[Callable] = None,
        process_pos_fn: Optional[Callable] = None,
        process_gripper_fn: Optional[Callable] = None,
        resize: int = 256,
        img_num: int = 9,
        obs_k: str = 'observation',
        act_k: str = 'action',
        img_k: str = 'rgb',
        gripper_k: str = 'gripper',
        pos_k: str = 'robot_pos',
        bbox_k: str = 'tracker_object_bbox',
        point_k: str = 'bb_center',
        prefix: str = ''
        ):
        self._every = every

        self.nbins = nbins
        self.process_act_fn = process_act_fn
        self.process_bbox_fn = process_bbox_fn
        self.process_point_fn = process_point_fn
        self.process_pos_fn = process_pos_fn
        self.process_gripper_fn = process_gripper_fn
        self.resize = resize
        self.img_num = img_num
        self.obs_k = obs_k
        self.act_k = act_k
        self.img_k = img_k
        self.gripper_k = gripper_k
        self.pos_k = pos_k
        self.bbox_k = bbox_k
        self.point_k = point_k

        self.prefix = prefix

    def __call__(
        self, 
        step: int,  
        batch: Dict[str, Dictator],
        config
        ):    
        if ((step + 1) % self._every) != 0:
            return

        img = viz_batch_img(
            batch, 
            process_act_fn=self.process_act_fn,
            process_bbox_fn=self.process_bbox_fn,
            process_point_fn=self.process_point_fn,
            process_pos_fn=self.process_pos_fn,
            process_gripper_fn=self.process_gripper_fn,
            resize=self.resize,
            img_num=self.img_num,
            obs_k=self.obs_k,
            act_k=self.act_k,
            img_k=self.img_k,
            gripper_k=self.gripper_k,
            pos_k=self.pos_k,
            bbox_k=self.bbox_k,
            point_k=self.point_k,
)
        tb.add_image('viz_batch_img', img.transpose((2,0,1)))
        stats = get_batch_statistics(
            batch, 
            obs_k=self.obs_k,
            act_k=self.act_k,
            pos_k=self.pos_k,
            nbins=self.nbins,
            )
        tb.step(step)
        for k, v in stats.items():
            k = self.prefix + k
            if k.endswith('_hist'):
                tb.add_histogram(k, v)
            elif k.endswith('_img'):
                tb.add_image(k, v, dataformats='HW')
            else:
                tb.scalar(k, v)

class TrainStatsCallback(Callback):
    def __init__(self, model: nn.Module, every: int, norm_type: float = 2.0, prefix: str = ''):
        self._model = model
        self._every = every
        self._norm_type = norm_type
        self.prefix = prefix

    def _get_lr(self, config):
        for param_group in config.optimizer.param_groups:
            return param_group['lr']

    def __call__(self, step: int, batch: Dict[str, Dictator], config):

        if ((step + 1) % self._every) != 0:
            return
        parameters = self._model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        norm_type = float(self._norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        device = grads[0].device
        if norm_type == torch._six.inf:
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        
        tb.step(step)
        tb.scalar(self.prefix + 'gradnorm', total_norm.cpu().detach().numpy())
        tb.scalar(self.prefix + 'lr', self._get_lr(config))
