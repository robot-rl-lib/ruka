import pathlib
from dataclasses import dataclass
from typing import Iterator, List, Optional, Callable
import os
from torch.optim.lr_scheduler import _LRScheduler

import torch
from ruka.models.losses.base import Loss
from ruka.types import Dictator

from .util import load_checkpoint, save_checkpoint

""" Main train loop for supervised learning
"""

"""
train data - iterator, gets train batches
loss_module - thing that on __call__(batch) returns a loss
optimitze - torch optimizer 
callbacks - list of functions that accept current step and current batch

num_updates - total number of updates
checkpoint_every - freq of saving checkpoints
checkpoint_path - where to save and where to seek checkpoints

dfs_checkpoint - download checkpoint and place it in 'checkpoint_path'
local_checkpoint - use local checkpoint to start training
"""

@dataclass
class TrainConfig:
    train_data: Iterator[Dictator] 
    loss_module: Loss
    optimizer: torch.optim.Optimizer
    callbacks: List[Callable]

    num_updates: int
    checkpoint_every: int
    checkpoint_path: str

    dfs_checkpoint: Optional[str] = None
    local_checkpoint: Optional[str] = None

    lr_scheduler: Optional[_LRScheduler] = None

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(config: TrainConfig) -> None:
    train_data = config.train_data
    loss_module = config.loss_module
    callbacks = config.callbacks
    optimizer = config.optimizer

    start_step = load_checkpoint(dict(loss=loss_module, optimizer=optimizer),
                                config.checkpoint_path, 
                                config.dfs_checkpoint,
                                config.local_checkpoint)

    # for batch in train_data:
    for step in range(start_step, config.num_updates):
        
        # train step
        loss_module.train()
        batch = next(train_data)
        loss = loss_module(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation step
        loss_module.eval()
        [cb(step, batch, config) for cb in callbacks]
        loss_module.log_stats(step, prefix='training/')
        if (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(step, dict(loss=loss_module, optimizer=optimizer), config.checkpoint_path) 
        print(f"Step {step} LR {_get_lr(optimizer)}")
        if config.lr_scheduler is not None:
            config.lr_scheduler.step()
    
    print("finished!")
