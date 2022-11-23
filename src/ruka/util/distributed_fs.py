import datetime
import os
import pytz

from filelock import FileLock
from ruka_os import distributed_fs as dfs_v1, in_cloud, DFS_CWD
from ruka_os import distributed_fs_v2 as dfs_v2
from ruka_os.globals import RUKA_LOCAL_STORAGE
from typing import List

dfs = dfs_v1

_SYNC_IF_LOCAL = False
DFS_CACHE_CONTENT_PATH = os.path.join(RUKA_LOCAL_STORAGE, 'dfs_cache', 'content')
DFS_CACHE_LOCKS_PATH = os.path.join(RUKA_LOCAL_STORAGE, 'dfs_cache', 'locks')


def set_dfs_v2():
    global dfs
    dfs = dfs_v2


def should_sync() -> bool:
    return in_cloud() or _SYNC_IF_LOCAL


def get_sync_if_local() -> bool:
    return _SYNC_IF_LOCAL


def set_sync_if_local(flag: bool = True):
    global _SYNC_IF_LOCAL
    _SYNC_IF_LOCAL = flag


def upload(local_path, wait=True):
    """
    Uploads file 'local_path' to a path in dfs specified by DFS_CWD variable.
    """
    # Check local path.
    local_path = os.path.normpath(local_path)
    assert not local_path.startswith('..')

    # Construct remote path.
    if os.path.isabs(local_path):
        remote_path = f'{DFS_CWD}/{os.path.basename(local_path)}'
    else:
        remote_path = f'{DFS_CWD}/{local_path}'

    # Upload.
    dfs.mkdir(os.path.dirname(remote_path))
    dfs.upload(local_path, remote_path, wait=wait)


def upload_maybe(local_path, wait=True):
    """Upload if should_sync()."""
    if should_sync():
        upload(local_path, wait=wait)


def download(remote_path, local_path):
    return dfs.download(remote_path, local_path)


def download_if_not_exists(remote_path, local_path=None):
    if local_path is None:
        local_path = os.path.basename(remote_path)

    if not os.path.exists(local_path):
        try:
            print(f'Downloading: {remote_path} -> {local_path}')
            dfs.download(remote_path, local_path)
        except:
            print(f'Error while downloading. Removing {local_path}')
            os.remove(local_path)
            raise
    else:
        print('Local path exists, skipping download')

    return local_path


def _get_modification_time(local_path: str) -> datetime.datetime:
    time = datetime.datetime.fromtimestamp(os.path.getmtime(local_path))
    time = time.replace(tzinfo=pytz.UTC)
    return time


def download_zip_and_extract(remote_path, local_path=None):
    if local_path is None:
        local_path = os.path.basename(remote_path)

    download_if_not_exists(remote_path, local_path)

    if not os.path.exists(os.path.splitext(local_path)[0]):
        print('Extracting data')
        os.system(f'unzip -q {local_path}')
    else:
        print('Skipping extraction')


def _get_cache_path(remote_path: str, lock: bool = False) -> str:
    if lock:
        path = [DFS_CACHE_LOCKS_PATH, remote_path + '.lock']
    else:
        path = [DFS_CACHE_CONTENT_PATH, remote_path]
    return os.path.join(*path)


def _lock_cache_file(remote_path: str) -> str:
    path = _get_cache_path(remote_path, lock=True)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return FileLock(path)


def cached_download(remote_path: str):
    '''
    Download single file.
    If local path exists and is modified after remote_path, does nothing.

    Args:
        remote_path (str): dfs path to the file

    Returns:
        local_path (str): local path, where the file was downloaded to
    '''

    with _lock_cache_file(remote_path):
        local_path = _get_cache_path(remote_path)
        do_download = False

        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path))
            do_download = True
        elif dfs.stat(remote_path).modification_time > _get_modification_time(local_path):
            do_download = True

        if do_download:
            print(f'Downloading {remote_path}...')
            dfs.download(remote_path, local_path)

        return local_path


def cached_download_and_unpack(remote_path: str) -> str:
    '''
    Download .tar.gz archive and unpack it.
    If local path exists and is modified after remote_path, does nothing.

    Args:
        remote_path (str): dfs path to the archive file

    Returns:
        local_path (str): local path, where the archive was unpacked to
    '''

    with _lock_cache_file(remote_path):
        local_path = _get_cache_path(remote_path)
        do_download = False

        if not os.path.exists(local_path):
            os.makedirs(local_path)
            do_download = True
        elif dfs.stat(remote_path).modification_time > _get_modification_time(local_path):
            do_download = True

        if do_download:
            print(f'Downloading {remote_path}...')
            dfs.download_and_unpack(remote_path, local_path)

        return local_path


def remove_from_cache(remote_path: str):
    with _lock_cache_file(remote_path):
        import shutil
        shutil.rmtree(_get_cache_path(remote_path), ignore_errors=True)
