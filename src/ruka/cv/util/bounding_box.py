import numpy as np
from numpy.typing import NDArray


class AxisAlignedBoundingBox:
    '''
    Contains axis aligned bounding box. Can contain many boxes to support vectorized operations.
    Axis aligned bounding box represents rectangle image region with sides parallel to image sides.
    It is a particular case of oriented bounding box.
    '''

    def __init__(self, box: NDArray):
        '''
        Args:
            box (np.array): bounding box in XYXY format
        '''

        raise NotImplementedError()

    def to_xyxy(self) -> NDArray:
        '''
        Returns box as np.array in XYXY format

        Returns:
            box: np.array of shape (n_boxes, 4,) or (4,) if n_boxes = 1
        '''

        raise NotImplementedError()

    def to_cxcywh(self) -> NDArray:
        '''
        Returns box as np.array in CXCYWH format

        Returns:
            box: np.array of shape (n_boxes, 4,) or (4,) if n_boxes = 1
        '''

        raise NotImplementedError()

    def to_oriented(self) -> OrientedBoundingBox:
        raise NotImplementedError()


class OrientedBoundingBox:
    '''
    Contains oriented bounding box. Can contain many boxes to support vectorized operations.
    Oriented bounding box represents a rectangle image region, whose sides can be not parallel to image sides.

    For example, consider axis-aligned bounding box with center at (c_x, c_y) and size (w, h).
    Then rotate this box bounds by r degrees CCW. This is an oriented bounding box.

    One can represent oriented bounding box with five numbers: (c_x, c_y, w, h, r)
    '''

    def __init__(self, box: NDArray):
        '''
        Args:
            box (np.array): bounding box in CXCYWHR format, R stands for rotation angle in degrees
        '''

        raise NotImplementedError()

    def to_axis_aligned(self):
        raise NotImplementedError()

    def to_cxcywhr(self):
        '''
        Returns box as np.array in CXCYWHR format, R states for rotation angle in degrees

        Returns:
            box: np.array of shape (n_boxes, 5,) or (5,) if n_boxes = 1
        '''

        raise NotImplementedError()
