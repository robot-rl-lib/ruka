import torch
from typing import Tuple
from ruka.cv.util import box_iou, box_area, intersect_boxes


class TrackingSuccess:
    def __init__(
        self,
        iou_threshold: float,
        skip_out_of_frame: bool,
        image_shape: Tuple[int, int] = None,
        n_skip_after_out: int = None):

        self.iou_threshold = iou_threshold
        self.skip_out_of_frame = skip_out_of_frame
        self.n_skip_after_out = n_skip_after_out
        self.image_box = (0, 0, image_shape[1], image_shape[0],) if image_shape is not None else None

    def __call__(self, predicted_boxes, real_boxes):
        after_out = self.n_skip_after_out
        for predicted, real in zip(predicted_boxes, real_boxes):
            if self.skip_out_of_frame and box_area(intersect_boxes(real, self.image_box)) < box_area(real):
                after_out = 0
                continue
            else:
                after_out += 1

            if box_iou(predicted, real) < self.iou_threshold:
                if self.skip_out_of_frame and after_out < self.n_skip_after_out:
                    continue

                return False

        return True
