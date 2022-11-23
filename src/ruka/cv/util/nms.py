import numpy as np
from numpy.typing import NDArray
from .bounding_box import AxisAlignedBBox


def nms(boxes: AxisAlignedBBox, scores: NDArray, iou_threshold: float):
    '''
    Perform Non Max Suppression on a set of boxes.

    Args:
        boxes (AxisAlignedBBox): bounding boxes to process
        scores (np.array): array with boxes scores with shape (n_boxes,)
        iou_threshold: NMS iou threshold. should be float in [0; 1]

    Returns:
        boxes (AxisAlignedBBox): boxes kept by NMS
        scores (np.array): scores of kept boxes
    '''

    xyxy = boxes.to_xyxy()
    x1, y1, x2, y2 = [xyxy[..., i] for i in range(xyxy.shape[1])]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        isec_x1 = np.maximum(x1[i], x1[order[1:]])
        isec_y1 = np.maximum(y1[i], y1[order[1:]])
        isec_x2 = np.minimum(x2[i], x2[order[1:]])
        isec_y2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, isec_x2 - isec_x1 + 1)
        h = np.maximum(0.0, isec_y2 - isec_y1 + 1)

        isec_area = w * h
        iou = isec_area / (areas[i] + areas[order[1:]] - isec_area)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return AxisAlignedBBox(xyxy[keep]), scores[keep]
