from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ruka.cv.util import OrientedBoundingBox
from collections.abc import Iterable
from ruka.cv.detect_and_track import Object


@dataclass
class FrameWithBox:
    frame: NDArray[np.uint8]
    box: OrientedBoundingBox


@dataclass
class DetectAndTrackEpisode:
    episode: Iterable[FrameWithBox]
    target: Object
