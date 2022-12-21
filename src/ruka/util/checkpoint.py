import tempfile
import torch as t
import ruka.util.distributed_fs as dfs
from types import ModuleType


def get_state_subdict(state_dict, prefix: str):
    """
    Returns state dict which is subset of given state dict.
    Includes keys starts with prefix.
    Also removes prefix from keys in resulting state dict.
    Args:
        state_dict: PyTorch state dict
        prefix (str): state dict keys prefix.
                      Please do not include point.

    Returns:
        state dict: PyTorch state dict
    """
    model_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith(prefix):
            model_state_dict[key[len(prefix) + 1 :]] = val
    return model_state_dict


def load_remote_state_dict(remote_ckpt_path: str, other_dfs: ModuleType = None):
    """
    Loads policy state dict from the checkpoint on dfs.
    Args:
        remote_ckpt_path (str): checkpoint path on dfs.

    Returns:
        state dict: PyTorch state dict.
    """
    local_path = dfs.cached_download(remote_ckpt_path, other_dfs=other_dfs)
    return t.load(local_path)
