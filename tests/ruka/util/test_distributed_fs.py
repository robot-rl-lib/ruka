import os
import tarfile
import tempfile
import pytest

from ruka.util.distributed_fs import (
    cached_download_and_unpack,
    cached_download,
    remove_from_cache,
    set_dfs_v2,
)
from ruka_os.distributed_fs_v2 import upload, remove


def _check_file(path, contents):
    with open(path, 'r') as f:
        assert f.read() == contents


@pytest.mark.external
def test_cached_download():
    remote_path = 'test/test_cached_download/foo.txt'

    def upload_file(contents):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(contents)
            f.flush()
            upload(f.name, remote_path)

    try:
        set_dfs_v2()
        remove_from_cache(remote_path)

        upload_file('foofoo')

        local_path = cached_download(remote_path)
        _check_file(local_path, 'foofoo')

        local_path = cached_download(remote_path)
        _check_file(local_path, 'foofoo')

        upload_file('barbar')

        local_path = cached_download(remote_path)
        _check_file(local_path, 'barbar')

    finally:
        remove(remote_path)


@pytest.mark.external
def test_cached_download_and_unpack():
    remote_path = 'test/test_cached_download_and_unpack/archive.tar.gz'

    def upload_archive(file_contents):
        with tempfile.TemporaryDirectory() as dir:
            tar_path = f'{dir}/archive.tar.gz'
            with open(os.path.join(dir, 'foo.txt'), 'w') as f:
                f.write(file_contents)
            with tarfile.open(tar_path, 'x:gz') as tar:
                tar.add(os.path.join(dir, 'foo.txt'), arcname='foo.txt')
            upload(tar_path, remote_path)

    try:
        set_dfs_v2()
        remove_from_cache(remote_path)

        upload_archive('foofoo')

        local_path = cached_download_and_unpack(remote_path)
        _check_file(os.path.join(local_path, 'foo.txt'), 'foofoo')

        local_path = cached_download_and_unpack(remote_path)
        _check_file(os.path.join(local_path, 'foo.txt'), 'foofoo')

        upload_archive('barbar')

        local_path = cached_download_and_unpack(remote_path)
        _check_file(os.path.join(local_path, 'foo.txt'), 'barbar')

    finally:
        remove(remote_path)
