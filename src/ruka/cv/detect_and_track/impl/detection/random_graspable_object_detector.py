from .methods.mdef_detr import make_mdef_detr_minus_language
from .one_shot_detector import OneShotDetector

from ruka.cv.util import nms, box_area, intersect_boxes, AxisAlignedBBox, OrientedBBox
from numpy.typing import NDArray
import numpy as np
import torch


class RandomGraspableObjectDetector(OneShotDetector):
    def __init__(
        self,
        search_box: OrientedBBox = None,
        max_side: int = None,
        n_box_proposals: int = 100,
        nms_iou_threshold: float = 0.7,
        inside_area_threshold: float = 0.7):
        '''
        Args:
            search_box (OrientedBBox): box to search in
            max_side (int): max side of bounding box to be considered as graspable object
            n_box_proposals (int): how many box proposals to consider
            nms_iou_threshold (float): threshhold for non max suppression
            inside_area_threshold (float): min box1 area fraction that should be inside box2 to filter box2
        '''

        super().__init__()

        self._model = make_mdef_detr_minus_language()

        self._search_box = search_box
        self._max_side = max_side
        self._n_box_proposals = n_box_proposals
        self._nms_iou_threshold = nms_iou_threshold
        self._inside_area_threshold = inside_area_threshold

    def to(self, destination):
        self._model.to(destination)
        return self

    def train(self, mode: bool = True):
        if mode:
            raise ValueError("Can't set RandomGraspableObjectDetector to train mode")
        self._model.eval()
        return self

    @torch.no_grad()
    def find(
        self,
        query_image: NDArray[np.uint8],
        target_image: NDArray[np.uint8],
        search_box: OrientedBBox = None) -> OrientedBBox:
        '''
        Look for graspable objects on an image and return random

        Args:
            image (np.array): image in RGB format to look for objects on

        Returns:
            box (OrientedBBox): random graspable object bbox
        '''

        boxes, scores = self._model(target_image)
        boxes = boxes.to_axis_aligned().to_xyxy()
        ind = np.argsort(scores)[-self._n_box_proposals:]

        boxes, _ = nms(AxisAlignedBBox(boxes[ind]), np.array(scores)[ind], self._nms_iou_threshold)
        xyxy = boxes.to_xyxy()

        xyxy = np.array(sorted(list(xyxy), key=lambda b: box_area(b)))

        if search_box is not None:
            global_box = search_box.to_axis_aligned().to_xyxy()
        else:
            if self._search_box is not None:
                global_box = self._search_box.to_axis_aligned().to_xyxy()
            else:
                global_box = (0, 0, target_image.shape[1], target_image.shape[0])

        keep = np.ones(xyxy.shape[0], dtype=bool)
        for i, box in enumerate(xyxy):
            if box_area(intersect_boxes(global_box, box)) / box_area(box) < self._inside_area_threshold:
                keep[i] = 0
                continue

            if self._max_side is not None and max(box[2] - box[0], box[3] - box[1]) > self._max_side:
                keep[i] = 0
                continue

            for box2 in xyxy[:i]:
                if box_area(intersect_boxes(box2, box)) / box_area(box2) > self._inside_area_threshold:
                    keep[i] = 0
                    continue

        xyxy = xyxy[keep]
        return AxisAlignedBBox(xyxy[np.random.choice(xyxy.shape[0])]).to_oriented()
