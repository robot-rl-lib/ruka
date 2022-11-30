import glob
import os
from typing import Iterator, Optional, Union, Dict, List
import pathlib

import numpy as np
import ruka.pytorch_util as ptu
import ruka.util.distributed_fs as dfs
import torch
from ruka.environments.common.env import Episode
import ruka.util.tensorboard as tb

def filter_successes(it: Iterator[Episode]):
    while True:
        ep = next(it)
        if ep.infos[-1]['is_success']:
            yield ep

from typing import Iterator, List, Optional
import numpy as np

def iterator_mixin(
    its: List[Iterator], 
    probs: Optional[List[float]] = None
    ) -> Iterator:
    assert (probs is None) or (len(its) == len(probs))
    if probs is None:
        probs = np.ones((len(its),))
    probs = np.array(probs) / np.array(probs).sum()
    while True:
        idx = np.random.choice(len(probs), p=probs)
        yield next(its[idx])
        
        
def save_checkpoint(step, model_dict: Dict[str, Union[ptu.TorchAware, torch.optim.Optimizer]], checkpoints_folder: str):
    pathlib.Path(checkpoints_folder).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(checkpoints_folder, str(step) + '.pth')
    state_dict_dict = {key: model.state_dict() for key, model in model_dict.items()} 
    torch.save(state_dict_dict, filename)
    dfs.upload_maybe(filename)


def get_latest_checkpoint_filename(checkpoint_filenames: List[str]) -> str:
    from_steps = [int(os.path.basename(file).split('.')[0]) for file in checkpoint_filenames]
    ind = np.argmax(from_steps)
    return checkpoint_filenames[ind], from_steps[ind]


def load_checkpoint(
    model_dict: Dict[str, Union[ptu.TorchAware, torch.optim.Optimizer]],
    checkpoints_folder: str, 
    dfs_checkpoint: Optional[str] = None,
    local_checkpoint: Optional[str] = None) -> int:

    pathlib.Path(checkpoints_folder).mkdir(parents=True, exist_ok=True)

    if (local_checkpoint is not None) and (dfs_checkpoint is not None):
        raise ValueError("minimum one of local_checkpoint or dfs_checkpoint should be set to None")  

    if local_checkpoint is not None:
        latest_checkpoint = local_checkpoint
        start_from = 0
    elif dfs_checkpoint is not None:
        download_to = os.path.join(checkpoints_folder, dfs_checkpoint)
        dfs.download(dfs_checkpoint, download_to)
        latest_checkpoint = download_to
        start_from = 0
    else:
        # continue training
        found_checkpoints = glob.glob(os.path.join(checkpoints_folder, '*.pth'))
        if not found_checkpoints:
            print("Checkpoints not found.")
            return 0

        latest_checkpoint, start_from = get_latest_checkpoint_filename(found_checkpoints)

    loaded = torch.load(latest_checkpoint)
    for key, model in model_dict.items():
        model.load_state_dict(loaded[key])
    print(f"Loaded model from '{latest_checkpoint}'")
    
    return start_from


def tb_data_size(episodes, prefix=''):
    transitions = None
    for s in [0, 1000]:
        tb.step(s)
        tb.scalar(f'data/{prefix}_ep_num', len(episodes))
        if episodes and hasattr(episodes[0], 'actions'):
            transitions = np.sum([len(ep.actions) for ep in episodes])
            tb.scalar(f'data/{prefix}_transition_num', transitions)

    print(f"{prefix} episodes:", len(episodes))
    if transitions:
        print(f"{prefix} transitions:", transitions)
