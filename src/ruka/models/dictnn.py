import torch as t
import torch.nn as nn

from typing import Dict, Union


class Filter(nn.Module):
    """Filter keys from dict obsrcation"""

    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(self, x: Dict[str, t.Tensor]):
        return {k: x[k] for k in self.keys}


class Concat(nn.Module):
    """Concatenates tensors from dict in single embedding in last denension"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Dict[str, t.Tensor]):
        return t.cat([x[k] for k in sorted(x)], dim=-1)

class Parallel(nn.Module):
    """Apply modules to dict observation in parallel"""

    def __init__(self, **modules: Dict[str, nn.Module]):
        super().__init__()
        self._dict_modules = nn.ModuleDict(modules)

    def forward(self, x: Dict[str, t.Tensor]) -> Dict[str, t.Tensor]:
        return {k: (self._dict_modules[k](t) if k in self._dict_modules else t)
                for k, t in x.items()}


def FilterConcat(*keys):
    """Cocatenate specific items from dict"""

    return nn.Sequential(Filter(*keys), Concat())

class StackImages(nn.Module):
    """Concatenate selected elements from dict input over -3 (channel) axis,
       and save into new out key"""

    def __init__(self, input_keys, out_key):
        super().__init__()
        self._keys = input_keys
        self._out = out_key

    def forward(self, x: Dict[str, t.Tensor]):
        x[self._out] = t.cat([x[k] for k in self._keys], axis=-3)
        return x

class PrintShape(nn.Module):

    def __init__(self, prefix='Print shape'):
        super().__init__()
        self._prefix = prefix

    def forward(self, x: Union[Dict[str, t.Tensor], t.Tensor]):
        if isinstance(x, dict):
            print(self._prefix, {k:v.shape for k,v in x.items()}, flush=True)
        else:
            print(self._prefix, x.shape, flush=True)
        return x
