import os.path

from typing import List
from ruka_os import distributed_fs as dfs, in_cloud, DFS_CWD


_SYNC_IF_LOCAL = False


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


def download_zip_and_extract(remote_path, local_path=None):
    if local_path is None:
        local_path = os.path.basename(remote_path)

    download_if_not_exists(remote_path, local_path)

    if not os.path.exists(os.path.splitext(local_path)[0]):
        print('Extracting data')
        os.system(f'unzip -q {local_path}')
    else:
        print('Skipping extraction')
