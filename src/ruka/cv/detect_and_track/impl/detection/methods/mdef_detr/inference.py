import numpy as np
import torch
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from numpy.typing import NDArray
from ruka.cv.util import AxisAlignedBBox, OrientedBBox
from typing import Tuple
from ruka.pytorch_util import TorchAware


class MDefDETRInference(TorchAware):
    IMAGE_SIZE = 800

    def __init__(self, model):
        super().__init__()

        self._model = model.eval()
        self._transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.IMAGE_SIZE),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def to(self, destination):
        self._model.to(destination)
        return self

    def train(self, mode: bool = True):
        if mode:
            raise ValueError("Can't set MDefDETRInference to train mode")
        self._model.eval()
        return self

    @torch.no_grad()
    def __call__(
        self,
        image: NDArray[np.uint8]) -> Tuple[OrientedBBox, NDArray[np.float]]:

        '''
        Find class-agnostic bounding boxes

        Args:
            image: RGB image to bind boxes on

        Returns:
            boxes (OrientedBBox): bounding boxes found
            scores (np.array): objectness score of each box in range [0; 1]
                shape: (n_boxes,)
        '''

        img = self._transform(image).unsqueeze(0).to(self.device)

        memory_cache = self._model(img, encode_and_save=True)
        outputs = self._model(img, encode_and_save=False, memory_cache=memory_cache)

        scores = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        bboxes = self._rescale_bboxes(
            outputs['pred_boxes'].cpu()[0, :],
            (image.shape[1], image.shape[0],)).numpy()
        bboxes = AxisAlignedBBox(bboxes).to_oriented()

        return bboxes, scores.numpy()

    def _rescale_bboxes(self, boxes, size):
        img_w, img_h = size
        boxes = self._box_cxcywh_to_xyxy(boxes)
        return boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @property
    def device(self):
        return next(iter(self._model.parameters())).device
