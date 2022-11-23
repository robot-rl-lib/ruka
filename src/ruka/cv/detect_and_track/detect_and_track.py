import numpy as np
from ruka.pytorch_util import TorchAware
from ruka.cv.util import OrientedBBox
from numpy.typing import NDArray
from .target_object import TargetObject


class DetectAndTrack(TorchAware):
    def reset(self, object: TargetObject):
        '''
        Reset DetectAndTrack with new target object

        Args:
            object (TargetObject): new target object
        '''

        raise NotImplementedError()

    def find(self, frame: NDArray[np.uint8]) -> OrientedBBox:
        '''
        Detect or track target object on a frame

        Args:
            frame (np.array): image to find object on in RGB format

        Returns:
            box (OrientedBBox): target object bounding box
        '''

        raise NotImplementedError()
