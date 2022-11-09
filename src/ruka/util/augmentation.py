import numpy as np

def crop_augment(observation, to_crop, crop_size):
    cropped = {}
    *bs, c, h, w = observation[to_crop[0]].shape
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