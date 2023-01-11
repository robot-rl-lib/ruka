import os
import pybullet_data

from ruka_os import distributed_fs_v2 as dfs
from ruka.util.distributed_fs import cached_download, _get_cache_path


def get_data(
        remote_path: str,
        download_parent_dir: bool = False
        ) -> str:

    remote_path = os.path.normpath(os.path.join('data', remote_path))

    # What is it?
    isdir = dfs.isdir(remote_path)
    isfile = dfs.isfile(remote_path)
    if isfile and isdir:
        raise RuntimeError(f'Remote path {remote_path} is both a file and a dir')

    # List of files to download.
    to_download = None
    if download_parent_dir:
        # - Download parent dir.
        remote_dir = os.path.dirname(remote_path)
        to_download = [
            f'{remote_dir}/{file}'
            for file in dfs.ls_files_recursive(remote_dir)
        ]
    elif isfile:
        # - Download file
        to_download = [remote_path]
    elif isdir:
        # - Download a directory.
        to_download = [
            f'{remote_path}/{file}'
            for file in dfs.ls_files_recursive(remote_path)
        ]

    # Download.
    for rp in to_download:
        cached_download(rp, other_dfs=dfs)

    # Get path.
    return _get_cache_path(remote_path)


def get_pybullet_data(path: str) -> str:
    path = os.path.normpath(path)
    assert not os.path.isabs(path)
    assert not path.startswith('..')
    return os.path.join(pybullet_data.getDataPath(), path)