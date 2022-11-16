import torch
import torch.nn as nn
import numpy as np
from ruka.cv.methods.swintrack import build_swin_track
from .single_object_tracker import SingleObjectTracker
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import cv2
import dataclasses
from ruka.cv.util import cut_box_and_make_square


@dataclasses.dataclass
class SingleObjectSwinTrackConfig:
    swin_track_type: str = 'Tiny'
    template_image_size: int = 112
    search_image_size: int = 224
    min_search_region_size: int = 224
    template_expand_coefficient: float = 1.8
    search_region_expand_coefficient: float = 3.0


class SingleObjectSwinTrack(SingleObjectTracker):
    def __init__(self, config: SingleObjectSwinTrackConfig = SingleObjectSwinTrackConfig()):
        super().__init__()

        self._config = config
        self._swin_track = build_swin_track(self._config.swin_track_type)

        self._normalize = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))

    @torch.no_grad()
    def initialize(self, img, box):
        template, _ = cut_box_and_make_square(img, box, expand_k=self._config.template_expand_coefficient)

        sz = self._config.template_image_size
        self._template_features = self._swin_track.initialize(self._preprocess_image(template, (sz, sz,)))
        self._box = box

    @torch.no_grad()
    def forward(self, image, box=None):
        self._box = self._track(image, box)
        return self._box

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def _track(self, image, box=None):
        self._box = box or self._box
        search_region, (offset_x, offset_y) = cut_box_and_make_square(
            image,
            self._box,
            expand_k=self._config.search_region_expand_coefficient,
            min_side=self._config.min_search_region_size)

        sz = self._config.search_image_size
        out = self._swin_track.track(
            self._template_features,
            self._preprocess_image(search_region, (sz, sz,)))
        classes, boxes = out['class_score'][0, 0].cpu().numpy(), out['bbox'][0].cpu().numpy()

        center_x, center_y, w, h = boxes[np.unravel_index(np.argmax(classes), classes.shape)]

        box = (
            int((center_x - w / 2) * search_region.shape[1] + offset_x),
            int((center_y - h / 2) * search_region.shape[0] + offset_y),
            int((center_x + w / 2) * search_region.shape[1] + offset_x),
            int((center_y + h / 2) * search_region.shape[0] + offset_y),
        )
        return box

    def _preprocess_image(self, img, shape):
        return self._normalize(torch.tensor(cv2.resize(img, shape).transpose((2, 0, 1))).float()[None] / 255).to(self.device)
