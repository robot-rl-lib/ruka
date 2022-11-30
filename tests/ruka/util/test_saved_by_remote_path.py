import json
import numpy as np
import pytest

from dataclasses import dataclass
from ruka.util.compression import img2jpg, jpg2img
from ruka.util.json import xdumps, xloads
from ruka.util.saved_by_remote_path import \
    SavedByRemotePath, JSONSavedByRemotePath
from ruka.util.migrating import Migrating
from ruka_os import distributed_fs_v2 as dfs
from typing import List


@dataclass
class BPL(SavedByRemotePath):
    '''
    A [b]ig [p]ersistent [l]ist.
    '''

    remote_path: str
    l: List[float]


@pytest.mark.external
def test_simple():
    remote_dir = 'test/ruka/util/test_saved_by_remote_path/test_simple'
    remote_path = f'{remote_dir}/my_list.bpl.json'

    my_list = BPL(remote_path, list(range(999999)))
    assert 'l' not in my_list.__getstate__()
    my_list.save()
    assert dfs.isfile(remote_path)
    my_list_copy = BPL.load(remote_path)
    assert my_list == my_list_copy


@dataclass
class JPEGImage(SavedByRemotePath):
    remote_path: str  # [H, W, 3]
    img: np.ndarray

    def __eq__(self, other):
        return type(other) is type(self) and \
            self.remote_path == other.remote_path and \
            np.allclose(self.img, other.img)

    def save(self):
        dfs.upload_bytes(img2jpg(self.img), self.remote_path)

    @classmethod
    def load(cls, remote_path: str) -> 'JPEGImage':
        return JPEGImage(
            remote_path=remote_path,
            img=jpg2img(dfs.download_bytes(remote_path))
        )


@pytest.mark.external
def test_custom():
    remote_dir = 'test/ruka/util/test_saved_by_remote_path/test_custom'
    remote_path = f'{remote_dir}/img.jpg'

    img = JPEGImage(remote_path, np.zeros((64, 64, 3)))
    assert 'img' not in img.__getstate__()
    img.save()
    assert img == JPEGImage.load(remote_path)


@dataclass
class Outer(Migrating):
    inner: 'Inner'


@dataclass
class Inner(JSONSavedByRemotePath):
    remote_path: str
    img: JPEGImage


@pytest.mark.external
def test_xdump():
    remote_dir = 'test/ruka/util/test_saved_by_remote_path/test_xdump'

    # Create.
    img = JPEGImage(f'{remote_dir}/img.jpg', np.zeros((8192, 8192, 3)))
    inner = Inner(f'{remote_dir}/inner.json', img)
    outer = Outer(inner)

    # Save
    img.save()
    inner.save()

    # Img.
    img_b = dfs.download_bytes(f'{remote_dir}/img.jpg')
    assert len(img_b) > 8192

    # Inner.
    inner_s = dfs.download_str(f'{remote_dir}/inner.json')
    json.loads(inner_s)
    assert len(inner_s) < 8192
    assert xloads(inner_s) == inner

    # Outer.
    outer_s = xdumps(outer)
    assert len(outer_s) < 8192
    assert xloads(outer_s) == outer