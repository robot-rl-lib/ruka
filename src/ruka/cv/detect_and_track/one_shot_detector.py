import numpy as np
from ruka.pytorch_util import TorchAware
from ruka.cv.util import OrientedBoundingBox
from numpy.typing import NDArray


class OneShotDetector(TorchAware):
    def find(
        self,
        query_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        search_box: OrientedBoundingBox = None) -> OrientedBoundingBox:

        """
        Perform one shot object detection on target_image conditioned by query_image

        Args:
            query_image (np.array): reference image in RGB format
            target_image (np.array): image to search on in RGB format
            search_box (OrientedBoundingBox): bounding box to crop target image. if None, whole image will be used

        Returns:
            box (OrientedBoundingBox): detected object bounding box
        """

        raise NotImplementedError()
