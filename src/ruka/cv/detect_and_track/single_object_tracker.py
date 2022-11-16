import numpy as np
from ruka.pytorch_util import TorchAware
from ruka.cv.util import OrientedBoundingBox
from numpy.typing import NDArray


class SingleObjectTracker(TorchAware):
    def initialize(self, image: NDArray[np.uint8], box: OrientedBoundingBox):
        """
        Initialize tracker with target

        Args:
            image (np.array): image in RGB format
            box (OrientedBoundingBox): target object bounding box

        Returns:
            None
        """

        raise NotImplementedError()

    def find(
        self,
        image: NDArray[np.uint8],
        box: OrientedBoundingBox = None) -> OrientedBoundingBox:

        """
        Perform tracking on an image

        Args:
            image (np.array): image in RGB format
            box (OrientedBoundingBox): optional search bbox

        Returns:
            box (OrientedBoundingBox): target object bounding box
        """

        raise NotImplementedError()
