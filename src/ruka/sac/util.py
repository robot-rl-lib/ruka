import abc
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn

import ruka.pytorch_util as ptu
from ruka.observation import Observation


class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass


def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    return elem_or_tuple_to_numpy(outputs)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    if isinstance(elem_or_tuple, Observation):
        return elem_or_tuple.to_pytorch()
    return ptu.from_numpy(elem_or_tuple).float()


def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if isinstance(v, Observation):
            yield k, v
        elif v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {
            k: _elem_or_tuple_to_variable(x)
            for k, x in _filter_batch(np_batch)
            if isinstance(x, Observation) or x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
        }
    else:
        return _elem_or_tuple_to_variable(np_batch)

def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix