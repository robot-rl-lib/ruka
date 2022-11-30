import cv2
import json
import numpy as np
import os
import ruka.util.distributed_fs as dfs

from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.cv.detect_and_track import TargetObject
from ruka.cv.util import AxisAlignedBBox, OrientedBBox
from ruka.util.image import load_rgb_image
from typing import Iterable


@dataclass
class FrameWithBox:
    frame: NDArray[np.uint8]
    box: OrientedBBox


@dataclass
class DetectAndTrackEpisode:
    episode: Iterable[FrameWithBox]
    target: TargetObject


class _IterableFromIteratorFactory:
    def __init__(self, create_iterator):
        self._create_iterator = create_iterator

    def __iter__(self):
        return self._create_iterator()


def read_detect_and_track_dataset(dfs_path: str) -> Iterable[DetectAndTrackEpisode]:
    dfs.set_dfs_v2()
    data_path = dfs.cached_download_and_unpack(dfs_path)

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        meta = json.load(f)

    def iterate_dataset(data_path):
        def iterate_episode(data_path, frames, boxes):
            for frame_path, box in zip(frames, boxes):
                image = load_rgb_image(os.path.join(data_path, frame_path))
                box = AxisAlignedBBox(np.array(box)).to_oriented()
                yield FrameWithBox(frame=image, box=box)

        for episode in meta['episodes']:
            reference_image_path = os.path.join(data_path, 'reference', episode['object'] + '.jpeg')
            target = TargetObject(
                reference_image=load_rgb_image(reference_image_path),
            )

            def make_episode_iterator():
                return iterate_episode(
                    data_path,
                    meta['runs'][episode['run']],
                    episode['boxes'])

            yield DetectAndTrackEpisode(
                episode=_IterableFromIteratorFactory(make_episode_iterator),
                target=target,
            )

    def make_dataset_iterator():
        return iterate_dataset(data_path)

    return _IterableFromIteratorFactory(make_dataset_iterator)
