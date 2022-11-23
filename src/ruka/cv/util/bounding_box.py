import numpy as np
from numpy.typing import NDArray


class AxisAlignedBBox:
    '''
    Contains axis aligned bounding box.
    Can contain many boxes to support vectorized operations.
    Axis aligned bounding box represents rectangle image region
    with sides parallel to image sides.

    It is a particular case of oriented bounding box.
    '''

    def __init__(self, box: NDArray):
        '''
        Args:
            box (np.array): bounding box(es) in XYXY format
                shape: (n_boxes, 4,) or (4,)
                dtype: int or float
        '''

        assert isinstance(box, np.ndarray)

        assert (len(box.shape) == 1 and box.shape[0] == 4) \
            or (len(box.shape) == 2 and box.shape[1] == 4)

        self._box = box

    def __getitem__(self, idx) -> 'AxisAlignedBBox':
        if self.n_boxes == 1:
            assert idx == 0
            return AxisAlignedBBox(self._box.copy())
        else:
            return AxisAlignedBBox(self._box[idx].copy())

    def item(self) -> 'AxisAlignedBBox':
        '''
        Get single bounding box.
        Expects that only one box is stored.
        '''

        assert self.n_boxes == 1
        return AxisAlignedBBox(self._box.squeeze().copy())

    def to_xyxy(self) -> NDArray:
        '''
        Returns box as np.array in XYXY format

        Returns:
            box: np.array of shape (n_boxes, 4,) or (4,) if created with single box
                n_boxes is the number of stored boxes
        '''

        return self._box.copy()

    def to_cxcywh(self) -> NDArray:
        '''
        Returns box as np.array in CXCYWH format

        Returns:
            box: np.array of shape (n_boxes, 4,) or (4,) if created with single box
                n_boxes is the number of stored boxes
        '''

        x1, y1, x2, y2 = [self._box[..., i] for i in range(4)]
        return np.ascontiguousarray(np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]).T)

    def to_oriented(self) -> 'OrientedBBox':
        angles = np.zeros(1) if self.n_boxes == 1 else np.zeros((self.n_boxes, 1))
        return OrientedBBox(np.concatenate([self.to_cxcywh(), angles], axis=-1))

    @property
    def n_boxes(self) -> int:
        '''
        Returns number of stored boxes.
        '''

        return 1 if len(self._box.shape) == 1 else self._box.shape[0]

    @property
    def corners(self) -> NDArray:
        '''
        Get corners of bounding box.

        Returns:
            corners (np.array): for each box has 4 corners in XY format.
                shape: (n_boxes, 4, 2,) or (4, 2,) if created with single box
        '''

        x1, y1, x2, y2 = [self._box[..., i] for i in range(4)]
        res = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        if len(res.shape) > 2:
            res = res.transpose(2, 0, 1)
        return res


class OrientedBBox:
    '''
    Contains oriented bounding box.
    Can contain many boxes to support vectorized operations.

    Oriented bounding box represents a rectangle image region,
    whose sides can be not parallel to image sides.

    For example, consider axis-aligned bounding
    box with center at (c_x, c_y) and size (w, h).
    Then rotate this box bounds by r degrees CCW.
    This is an oriented bounding box.

    One can represent oriented bounding box with five numbers:
        (c_x, c_y, w, h, r)
    '''

    def __init__(self, box: NDArray):
        '''
        Args:
            box (np.array): bounding box(es) in CXCYWHR format,
                R stands for rotation angle in degrees.
                shape: (n_boxes, 5,) or (5,)
                dtype: int or float
        '''

        assert isinstance(box, np.ndarray)

        assert (len(box.shape) == 1 and box.shape[0] == 5) \
            or (len(box.shape) == 2 and box.shape[1] == 5)

        self._box = box

    def __getitem__(self, idx) -> 'OrientedBBox':
        if self.n_boxes == 1:
            assert idx == 0
            return OrientedBBox(self._box.copy())
        else:
            return OrientedBBox(self._box[idx].copy())

    def item(self) -> 'OrientedBBox':
        '''
        Get single bounding box.
        Expects that only one box is stored.
        '''

        assert self.n_boxes == 1
        return OrientedBBox(self._box.squeeze().copy())

    def to_axis_aligned(self) -> AxisAlignedBBox:
        corners = self.corners
        x1, y1 = np.amin(corners[..., 0], axis=-1), np.amin(corners[..., 1], axis=-1)
        x2, y2 = np.amax(corners[..., 0], axis=-1), np.amax(corners[..., 1], axis=-1)
        xyxy = np.vstack([x1, y1, x2, y2]).T
        if len(self._box.shape) == 1:
            xyxy = xyxy.squeeze()
        return AxisAlignedBBox(xyxy)

    def to_cxcywhr(self) -> NDArray:
        '''
        Returns box as np.array in CXCYWHR format,
        R states for rotation angle in degrees

        Returns:
            box: np.array of shape (n_boxes, 5,) or (5,) if created with single box
                n_boxes is the number of stored boxes
        '''

        return self._box.copy()

    @property
    def n_boxes(self) -> int:
        '''
        Returns number of stored boxes.
        '''

        return 1 if len(self._box.shape) == 1 else self._box.shape[0]

    @property
    def corners(self) -> NDArray:
        '''
        Get corners of bounding box.

        Returns:
            corners (np.array): for each box has 4 corners in XY format.
                shape: (n_boxes, 4, 2,) or (4, 2,) if created with single box
        '''

        angles = self._box[..., 4] * np.pi / 180
        mat = np.dstack([
            np.cos(angles),
            np.sin(angles),
            -np.sin(angles),
            np.cos(angles),
        ]).reshape(-1, 2, 2)

        corners = []
        for dx, dy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            cur = np.array([
                dx * self._box[..., 2] / 2,
                dy * self._box[..., 3] / 2,
            ]).T
            if len(cur.shape) == 1:
                cur = cur[None]
            corners.append(cur[None])

        corners = np.concatenate(corners, axis=0)
        corners = corners.transpose(1, 2, 0)

        res = np.matmul(mat, corners).transpose(0, 2, 1)
        res[:, :, 0] += self._box[..., 0][..., None]
        res[:, :, 1] += self._box[..., 1][..., None]

        if len(self._box.shape) == 1:
            res = res.squeeze()
        return res
