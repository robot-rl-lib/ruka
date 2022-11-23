import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Union, Tuple
from .bounding_box import AxisAlignedBBox, OrientedBBox


def cut_box_and_make_square(image, box, expand_k=1.0, min_side=None):
    c_x, c_y, w, h = box.to_cxcywh().squeeze().astype(int)
    half_side = int(max(w, h) * expand_k / 2)
    if min_side is not None:
        half_side = max(half_side, min_side // 2)

    x_min, y_min = max(0, c_x - half_side), max(0, c_y - half_side)
    x_max, y_max = min(image.shape[1], c_x + half_side + 1), min(image.shape[0], c_y + half_side + 1)

    result = image[y_min:y_max, x_min:x_max]

    sides_diff = abs(result.shape[0] - result.shape[1])

    if sides_diff > 0:
        pad = (sides_diff // 2, sides_diff - sides_diff // 2,)
        if result.shape[0] > result.shape[1]:
            result = np.pad(result, ((0, 0,), pad, (0, 0,)), mode='edge')
            x_min -= pad[0]
        else:
            result = np.pad(result, (pad, (0, 0,), (0, 0,),), mode='edge')
            y_min -= pad[0]

    assert result.shape[0] == result.shape[1]

    return result, (x_min, y_min)


def plot_bounding_box(
    image: NDArray[np.uint8],
    box: Union[AxisAlignedBBox, OrientedBBox],
    color: Tuple[int, int, int],
    thickness: int) -> NDArray[np.uint8]:

    '''
    Draw bounding boxes on an image. Supports multiple boxes.

    Args:
        image: RGB image to draw on. Will be copied before drawing.
        box: bounding box(es) to draw
        color: tuple(R, G, B), each channel in range [0, 255]
        thickness: line thickness

    Returns:
        image (NDArray[np.uint8]): image with bounding bow drawn
    '''

    result = image.copy()
    for i in range(box.n_boxes):
        corners = box[i].corners.astype(int)
        for c in range(corners.shape[0]):
            result = cv2.line(
                result,
                tuple(corners[c].astype(int)),
                tuple(corners[(c + 1) % corners.shape[0]].astype(int)),
                color,
                thickness)

    return result
