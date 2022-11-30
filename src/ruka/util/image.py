import cv2
import numpy as np
import ruka.util.distributed_fs as dfs
import tempfile
import os

from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.util.saved_by_remote_path import SavedByRemotePath
from typing import List


def load_rgb_image(path: str) -> NDArray[np.uint8]:
    '''
    Loads rgb image from .jpg .jpeg or .png file.

    Args:
        path (str): path to the image to load

    Returns:
        image (np.array): image in RGB HWC format
    '''

    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb_image(image: NDArray[np.uint8], path: str):
    '''
    Saves rgb image to file.

    Args:
        image (np.array): RGB HWC image to save
        path (str): path to the file to save the image to
    '''

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


@dataclass
class SavedRGBImage(SavedByRemotePath):
    remote_path: str
    image: NDArray  # [H, W, C], RGB image

    def save(self):
        with tempfile.NamedTemporaryFile() as f:
            save_rgb_image(self.image, f.name)
            upload(f.name, f.remote_path)

    @classmethod
    def load(self, remote_path: str) -> 'SavedRGBImage':
        local_path = dfs.cached_download(remote_path)
        image = load_rgb_image(local_path)
        return SavedRGBImage(remote_path=remote_path, image=image)


@dataclass
class SavedRGBImages(SavedByRemotePath):
    remote_path: str  # should be .tar.gz archive
    images: List[NDArray]  # [H, W, C], RGB image

    def save(self):
        assert self.remote_path.endswith('.tar.gz')

        with tempfile.TemporaryDirectory() as dir:
            name_len = len(str(len(self.images)))
            for i, image in enumerate(self.images):
                save_rgb_image(image, f'{dir}/{str(i).zfill(name_len)}.jpg')

            dfs.dfs.pack_and_upload(dir, self.remote_path)

    @classmethod
    def load(self, remote_path: str) -> 'SavedRGBImages':
        local_path = dfs.cached_download_and_unpack(remote_path)
        fnames = sorted(os.listdir(local_path))

        images = list([load_rgb_image(os.path.join(local_path, f)) for f in fnames])

        return SavedRGBImages(images=images, remote_path=remote_path)
