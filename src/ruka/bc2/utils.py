import numpy as np
import ruka.pytorch_util as ptu
def crop_augment(observation, to_crop, crop_size):
    cropped = {}
    *bs, c, h, w = observation['depth'].shape
    crop_max = h - crop_size + 1
    n = np.prod(bs)
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    for k, v in observation.items():
        if k not in to_crop:
            cropped[k] = v
            continue
        v = v.reshape((n, c, h, w))
        cropped[k] = np.zeros((n, c, crop_size, crop_size))
        for i, (w11, h11) in enumerate(zip(w1, h1)):
            cropped[k][i] = v[i, :, h11:(h11 + crop_size), w11:(w11 + crop_size)]
        cropped[k] = cropped[k].reshape((*bs, c, crop_size, crop_size))
    return cropped

def default_collate_fn(sequence_list):
    """ Collates lists of lists of paths
    """
    if not isinstance(sequence_list[0][0], dict):
        return (np.array(sequence_list))
    out = dict()
    for key in sequence_list[0][0].keys():
        val = [[x[key] for x in sequence] for sequence in sequence_list]
        out[key] = (np.array(val)) # BS, T, *DIMS 
    return out

def numpy_tree_to_torch(dct):
    if isinstance(dct, dict):
        out = {}
        for key, val in dct.items():
            out[key] = numpy_tree_to_torch(val)
        return out
    else:
        return ptu.from_numpy(dct)