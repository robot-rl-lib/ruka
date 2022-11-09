import traceback
import threading
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor, Future

from ruka.environments.common.env import Episode
from ruka.logging.episode import create_episode_logger
from ruka.logging.ep2viz import log_visualize_episode


_LOCK = threading.Lock()
_EXECUTOR = None


def _get_executor():
    global _EXECUTOR

    if _EXECUTOR:
        return _EXECUTOR

    with _LOCK:
        if _EXECUTOR:
            return _EXECUTOR

        _EXECUTOR = ProcessPoolExecutor(max_workers=1)
        return _EXECUTOR

def _log_episode(episode: Episode, tags: Optional[Union[str, List[str]]]=None, **kwargs) -> Union[str, None]:
    """ Log episode function for call in sparate process 
        Return:
            String episode id or None if fail
    """
    try:        
        ep_logger = create_episode_logger()
        if tags is not None:
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:
                ep_logger.assign_tag(tag)
        log_visualize_episode(ep_logger, episode, **kwargs)
        id = ep_logger.close(episode_time=len(episode.actions) * 0.1)
        return id
    except:
        traceback.print_exc()
        raise    

def log_episode(episode: Episode, tags: Optional[Union[str, List[str]]]=None, wait=True, **kwargs) -> Union[str, Future]:
    """ Async log episode in separate process,
        Return:
            String episode id if wait=True else concurrent.futures.Future
    """
    if wait:
        _log_episode(episode, tags, **kwargs)
    else:
        executor = _get_executor()
        executor.submit(_log_episode, episode, tags, **kwargs)


