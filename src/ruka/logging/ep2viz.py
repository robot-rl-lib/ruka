import numpy as np
from typing import Any

from ruka.logging.episode import EpisodeLogger
from ruka.environments.common.env import Episode



def log_visualize_episode(ep_logger: EpisodeLogger, episode: Episode, img_format: str = 'HWC', has_batch_dim: bool=False) -> None:
    """ 
    Args:
        ep_logger: episode logger instance
        episode: episode data instance
        img_format: HWC or CHW
        has_batch_dim: dose all observations have additional leading dimension 

    Supported types:
        float, int, bool     - logs as scalar
        1d ndarray           - logs as scalars
        2d,3d images: 
            If dtype == uint8     assume value in range [0, 255]
            if dtype == any float assume value in range [0, 1] -> will be converted to 0..255 uint8

            2d ndarray       - logs as grayscale image
            3d ndarray:
                1ch - grayscale
                2ch - 2 grayscale images 
                3ch - RGB
                >3  - N grayscale images 
    """
    for i, obs in enumerate(episode.observations):
        maybe_log_item('obs', obs, ep_logger, img_format, has_batch_dim)

        # len(obs) == len(actions) + 1
        if i < len(episode.actions):
            maybe_log_item('actions', episode.actions[i], ep_logger, img_format, has_batch_dim)
            maybe_log_item('reward', episode.rewards[i], ep_logger, img_format, has_batch_dim)
            maybe_log_item('info', episode.infos[i], ep_logger, img_format, has_batch_dim)
        ep_logger.step()


def maybe_log_item(name: str, value: Any, ep_logger: EpisodeLogger, img_format, has_batch_dim) -> None:
    # scalars
    if isinstance(value, (float, int, bool)):
        ep_logger.add_scalar(name, value)
    
    # arrays
    if isinstance(value, np.ndarray):
        if has_batch_dim:
            if value.shape[0] != 1:
                raise ValueError(f"has_batch_dim is True but first dim is not 1, value shape {value.shape}")
            value = value[0]

        if len(value.shape) == 1 and value.shape[0] <= 10:
            # just vector with scalars
            ep_logger.add_scalars(name, value.tolist())
        elif len(value.shape) == 2:
            # 1 channel img
            if img_format == 'HWC':
                value = value[:,:, None]
            else:
                value = value[None, :,:]
            log_img(name, value, ep_logger)
        elif len(value.shape) == 3:
            
            if (img_format=='HWC' and (value.shape[2] > 3 and value.shape[0] in [1, 3])) or \
               (img_format=='CHW' and (value.shape[0] > 3 and value.shape[2] in [1, 3])):
                print(f'WARNING: you have img shape {value.shape} and format {img_format}')            

            if img_format == 'CHW':
                value = value.transpose((1,2,0))
            # HWC case
            if value.shape[2] == 1 or value.shape[2] == 3:
                # HW1 or HW3
                log_img(name, value, ep_logger)
            else:
                # HW-N
                for ch in range(value.shape[2]):
                    log_img(f"{name}_ch{ch}", value[:,:,ch:ch+1], ep_logger), 

    
    # dicts
    if isinstance(value, dict):
        for k,v in value.items():
            maybe_log_item(f"{name}/{k}", v, ep_logger, img_format, has_batch_dim)


def log_img(name: str, img: np.ndarray, ep_logger: EpisodeLogger) -> None:
    """ 
    Args:    
        img: image with shape HWC where C is 1 - grayscale or 3 - rgb 
            If dtype == uint8     assume value in range [0, 255]
            if dtype == any float assume value in range [0, 1] -> will be converted to 0..255 uint8    
    """
    assert len(img.shape) == 3 and (img.shape[2] == 1 or img.shape[2] == 3), img.shape
    
    if img.dtype == np.float32 or img.dtype == np.float16 or img.dtype == np.float64:
        # 0..1 range to 0..255
        img = np.clip(img * 255, 0, 255)

    img = img.astype(np.uint8)

    if img.shape[2] == 1:
        # just repeat 1 ch 3 times
        img = np.tile(img, (1, 1, 3))

    ep_logger.add_video_frame(name, img, width=img.shape[1], height=img.shape[0])
