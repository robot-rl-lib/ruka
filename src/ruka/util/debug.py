import numpy as np
import torch

def smart_shape(object):
    """ Custom object shape for debug """
    if object is None:
        return None
    elif isinstance(object, (int, float, str)):
        return f'{type(object)}: {object}'
    elif isinstance(object, (list, tuple)):
        return [f'len: {len(object)}'] + [smart_shape(i) for i in object]
    elif hasattr(object, 'shape'):
        return object.shape
    elif isinstance(object, dict):
        return {k: smart_shape(v) for k,v in object.items()}
    else:
        return f'Unknow type: {type(object)}'


def smart_stats(object):
    """ Calc min, mean, max for list ot dict """
    if object is None:
        return None
    elif isinstance(object, (int, float, str)):
        return f'scalar: {object}'
    elif isinstance(object, (list, tuple)):
        return f'listlike min {np.min(np.array(object))} mean {np.mean(np.array(object))} max {np.max(np.array(object))}'
    elif hasattr(object, 'shape'):
        if isinstance(object, torch.Tensor):
            object = object.detach().cpu().numpy()
        return f'tensor min {np.min(np.array(object))} mean {np.mean(np.array(object))} max {np.max(np.array(object))}'
    elif isinstance(object, dict):
        return '\n'.join([f'{k}: {smart_stats(v)}' for k,v in object.items()])
    else:
        return f'Unknow type: {type(object)}'
