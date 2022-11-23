import numpy as np
from ruka.pytorch_util import TorchAware
from ruka.cv.util import OrientedBBox
from numpy.typing import NDArray


class SingleObjectTracker(TorchAware):
    def initialize(self, image: NDArray[np.uint8], box: OrientedBBox):
        """
        Initialize tracker with target

        Args:
            image (np.array): image in RGB format
            box (OrientedBBox): target object bounding box

        Returns:
            None
        """

        raise NotImplementedError()

    def find(
        self,
        image: NDArray[np.uint8],
        box: OrientedBBox = None) -> OrientedBBox:

        """
        Perform tracking on an image

        Args:
            image (np.array): image in RGB format
            box (OrientedBBox): optional search bbox

        Returns:
            box (OrientedBBox): target object bounding box
        """

        raise NotImplementedError()
