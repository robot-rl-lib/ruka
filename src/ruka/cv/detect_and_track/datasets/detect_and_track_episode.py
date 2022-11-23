from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ruka.cv.util import OrientedBBox
from collections.abc import Iterable
from ruka.cv.detect_and_track import Object


@dataclass
class FrameWithBox:
    frame: NDArray[np.uint8]
    box: OrientedBBox


@dataclass
class DetectAndTrackEpisode:
    episode: Iterable[FrameWithBox]
    target: Object


def read_detect_and_track_dataset(dfs_path: str) -> Iterable[DetectAndTrackEpisode]:
    ...
