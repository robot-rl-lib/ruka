from ruka.cv.util import OrientedBBox
from numpy.typing import NDArray
import numpy as np

from ..detect_and_track import DetectAndTrack
from .tracking.single_object_tracker import SingleObjectTracker
from .detection.one_shot_detector import OneShotDetector
from ..target_object import TargetObject


class GenericDetectAndTrack(DetectAndTrack):
    def __init__(
        self,
        detector: OneShotDetector,
        tracker: SingleObjectTracker):

        self._detector = detector
        self._tracker = tracker

        self._is_tracking = False
        self._target = None

    def to(self, destination):
        self._detector.to(destination)
        self._tracker.to(destination)
        return self

    def train(self, mode: bool = True):
        self._detector.train(mode)
        self._tracker.train(mode)
        return self

    def reset(self, target: TargetObject):
        '''
        Reset DetectAndTrack with new target object

        Args:
            object (TargetObject): new target object
        '''

        self._target = target
        self._is_tracking = False

    def find(self, frame: NDArray[np.uint8]) -> OrientedBBox:
        '''
        Detect or track target object on a frame

        Args:
            frame (np.array): image to find object on in RGB format

        Returns:
            box (OrientedBBox): target object bounding box
        '''

        assert self._target is not None

        if self._is_tracking:
            return self._tracker.find(frame)
        else:
            box = self._detector.find(self._target.reference_image, frame)
            self._tracker.initialize(frame, box)
            self._is_tracking = True
            return box
