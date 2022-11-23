import numpy as np

def crop_augment(obs_batch, to_crop, crop_size):
    """
    random crops [H, W, C] images
    """
    cropped = {}
    *bs, h, w, c = obs_batch[to_crop[0]].shape
    crop_max = h - crop_size + 1
    n = np.prod(bs)
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    for k, v in obs_batch.items():
        if k not in to_crop:
            cropped[k] = v
            continue
        v = v.reshape((n, h, w, c))
        cropped[k] = np.zeros((n, crop_size, crop_size, c))
        for i, (w11, h11) in enumerate(zip(w1, h1)):
            cropped[k][i] = v[i, h11:(h11 + crop_size), w11:(w11 + crop_size), :]
        cropped[k] = cropped[k].reshape((*bs, crop_size, crop_size, c))
    return cropped

def center_crop(obs_batch, to_crop, crop_size):  
    """
    center crops [H, W, C] images
    """
    for key in to_crop:
        *_, w, h, c = obs_batch[key].shape
        lhw = (w - crop_size)//2
        rhw = crop_size + lhw

        lhh = (h - crop_size)//2
        rhh = crop_size + lhh
        obs_batch[key] = obs_batch[key][..., lhw:rhw, lhh:rhh, :]
        
    return obs_batch
