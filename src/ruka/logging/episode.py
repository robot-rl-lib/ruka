import copy
import ruka.util.nested_dict as nd
import tempfile
import traceback
import threading

from dataclasses import dataclass
from functools import partial
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, Future
from ruka.environments.common.env import Episode
from ruka.util.compression import \
    compress_everything_maybe, decompress_everything_maybe
from ruka_os import distributed_fs_v2 as dfs
from ruka.util import distributed_fs as dfs_util

from .logger import \
    Logger, LogReader, FPSParams, create_ruka_logger, create_ruka_log_reader
from .visualize import visualize_nested_dict


@dataclass
class EpisodeLogParams:
    """
    video_fps: must be passed if visualizing and there are videos in
        the episode.

    visualize: if True, do episode visualization.

    save_data: if True save raw episode data.

    compress: if True, compress episode with jpg/png (relevant only if
        save_as_data == True).
    """
    video_fps: Optional[FPSParams] = None
    visualize: bool = True
    save_as_data: bool = True
    compress: bool = True
    compress_heuristic: bool = False


def log_episode(
        logger: Logger,
        episode: Episode,
        params: EpisodeLogParams = EpisodeLogParams(),
    ):
    """
    Store episode to logger. All keys will have 'episode/' prefix.
    """
    return _log_episode(logger, episode, params)


def save_episode(
        remote_path: str,
        episode: Episode,
        params: EpisodeLogParams = EpisodeLogParams(),
        wait: bool = True
    ) -> Optional[Future]:
    """
    Create logger, store the episode, and then upload to DFS.

    Return
        If wait=True, return concurrent.futures.Future (None otherwise).
    """
    return _save_episode(remote_path, episode, params, wait)


def get_episode(log_reader: LogReader) -> Episode:
    """
    From log_reader, obtain an episode saved by log_episode().
    """
    return _get_episode(log_reader)


def load_episode(remote_path: str) -> Episode:
    """
    Load episode, previously saved by save_episode().
    """
    return _load_episode(remote_path)


def cached_load_episode(remote_path: str) -> Episode:
    """
    Load episode, previously saved by save_episode().
    """
    return _cached_load_episode(remote_path)


# ----------------------------------------------------------- Implementation --


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


def _guard(*args, **kwargs):
    fn, args = args[0], args[1:]

    try:
        return fn(*args, **kwargs)
    except:
        traceback.print_exc()
        raise


def _log_episode(logger: Logger, episode: Episode, params: EpisodeLogParams):
    # Copy.
    episode = copy.deepcopy(episode)

    # Visualize.
    if params.visualize:
        for i in range(len(episode.observations)):
            logger.step(i)
            visualize_nested_dict(logger, episode.observations[i], 'episode/observation', auto=True)
            if i < len(episode):
                visualize_nested_dict(logger, episode.actions[i], 'episode/action', auto=True)
                visualize_nested_dict(logger, episode.rewards[i], 'episode/reward', auto=True)
                visualize_nested_dict(logger, episode.infos[i], 'episode/info', auto=True)
        if params.video_fps is not None:
            logger.set_video_fps(params.video_fps)

    # Data.
    if params.save_as_data:
        # - Compression.
        if params.compress:
            compress_fn = partial(
                compress_everything_maybe, heuristic=params.compress_heuristic)
            episode.observations = [
                nd.map_inplace(compress_fn, obs)
                for obs in episode.observations
            ]
            episode.infos = [
                nd.map_inplace(compress_fn, info)
                for info in episode.infos
            ]

        # - Save as data at 0'th step.
        logger.add_data('episode/data', episode, step_no=0)


def _save_episode(
        remote_path: str,
        episode: Episode,
        params: EpisodeLogParams = EpisodeLogParams(),
        wait: bool = True
    ) -> Optional[Future]:

    # Async.
    if not wait:
        return _get_executor().submit(_guard,
            _save_episode, remote_path, episode, params, wait=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Log.
        local_path = f'{tmpdir}/log'
        with create_ruka_logger(local_path) as logger:
            _log_episode(logger, episode, params)

        # Upload.
        dfs.pack_and_upload(local_path, remote_path)


def _get_episode(log_reader: LogReader) -> Episode:
    # Get & copy.
    episode = log_reader.get_value('episode/data')[0]
    episode = copy.deepcopy(episode)

    # Decompress
    episode.observations = [
        nd.map_inplace(decompress_everything_maybe, obs)
        for obs in episode.observations
    ]
    episode.infos = [
        nd.map_inplace(decompress_everything_maybe, info)
        for info in episode.infos
    ]

    return episode


def _load_episode(remote_path: str) -> Episode:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download.
        local_path = f'{tmpdir}/log'
        dfs.download_and_unpack(remote_path, local_path)

        # Get episode.
        log_reader = create_ruka_log_reader(local_path)
        return _get_episode(log_reader)


def _cached_load_episode(remote_path: str) -> Episode:
        local_path = dfs_util.cached_download_and_unpack(remote_path, other_dfs=dfs)
        
        # Get episode.
        log_reader = create_ruka_log_reader(local_path)
        return _get_episode(log_reader)