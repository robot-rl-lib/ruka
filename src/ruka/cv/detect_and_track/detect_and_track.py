import numpy as np
from ruka.pytorch_util import TorchAware
from ruka.cv.util import OrientedBoundingBox
from numpy.typing import NDArray
from ruka.cv.detect_and_track import Object


class DetectAndTrack(TorchAware):
    def reset(object: Object):
        '''
        Reset DetectAndTrack with new target object

        Args:
            object (Object): new target object
        '''

        raise NotImplementedError()

    def find(frame: NDArray[np.uint8]) -> OrientedBoundingBox:
        '''
        Detect or track target object on a frame

        Args:
            frame (np.array): image to find object on in RGB format

        Returns:
            box (OrientedBoundingBox): target object bounding box
        '''

        raise NotImplementedError()
