import glob
import os
from typing import Iterator, Optional

import numpy as np
import ruka.pytorch_util as ptu
import ruka.util.distributed_fs as dfs
import torch
from ruka.environments.common.env import Episode


def filter_successes(it: Iterator[Episode]):
    while True:
        ep = next(it)
        if ep.infos[-1]['is_success']:
            yield ep


def save_checkpoint(step, model: ptu.TorchAware, checkpoints_folder: str):
    filename = os.path.join(checkpoints_folder, str(step) + '.pth')
    torch.save(model.state_dict(), filename)
    dfs.upload_maybe(filename)

def load_checkpoint(
    model: ptu.TorchAware, 
    checkpoints_folder: str, 
    dfs_checkpoint: Optional[str] = None,
    local_checkpoint: Optional[str] = None) -> int:
    
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

        from_steps = [int(os.path.basename(file).split('.')[0]) for file in found_checkpoints]
        latest_checkpoint = found_checkpoints[np.argmax(from_steps)]    
        start_from = max(from_steps)

    model.load_state_dict(torch.load(latest_checkpoint))
    print(f"Loaded model from '{latest_checkpoint}'")
    return start_from
