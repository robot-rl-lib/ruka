import torch
import torch.nn as nn
import numpy as np
from .swintrack.builder import build_swin_track
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import cv2

class SingleObjectSwinTrack(nn.Module):
    def __init__(self):
        super().__init__()

        self.swin_track = build_swin_track('Tiny')

        self.template_features = None
        self.box = None
        self.normalize = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))

        self.n_frames = 0

    def set_template(self, img, box):
        template, _ = self._cut_square_box(img, *box, k=1.8)
        self.box = box
        with torch.no_grad():
            self.template_features = self.swin_track.initialize(self._preprocess_image(template, (112, 112,)))

    def forward(self, img_):
        img = img_.copy()
        search, roi = self._cut_square_box(img, *self.box, k=3.0)

        with torch.no_grad():
            res = self.swin_track.track(self.template_features, self._preprocess_image(search, (224, 224,)))
            classes, boxes = res['class_score'][0, 0].cpu().numpy(), res['bbox'][0].cpu().numpy()

        # classes = classes / np.amax(classes)

        pos = np.unravel_index(np.argmax(classes), classes.shape)
        box = boxes[pos]

        cell_h, cell_w = search.shape[0] // classes.shape[0], search.shape[1] // classes.shape[1]
        center_x, center_y = int(box[1] * search.shape[0]), int(box[0] * search.shape[1]) # int(pos[0] * cell_h + box[0] * cell_h), int(pos[1] * cell_w + box[1] * cell_w)
        box_h, box_w = int(search.shape[0] * box[3]), int(search.shape[1] * box[2])

        self.box = (
            center_x - box_h // 2 + roi[0],
            center_y - box_w // 2 + roi[1],
            center_x + box_h // 2 + roi[0],
            center_y + box_w // 2 + roi[1],
        )

        img = cv2.rectangle(img, (roi[1], roi[0],), (roi[3], roi[2],), (255, 0, 0,), 2)

        img = cv2.rectangle(img, (self.box[1], self.box[0]), (self.box[3], self.box[2]), (0, 0, 255,), 2)

        '''
        self.n_frames += 1
        if self.n_frames > 10:
            self.n_frames = 0
            self.set_template(img_, self.box)
        '''

        return img, self.box

    @property
    def device(self):
        return next(self.parameters()).device

    def _cut_square_box(self, img, x_min, y_min, x_max, y_max, k):
        c_x, c_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        s = max(int((max(x_max - x_min, y_max - y_min) // 2) * k), 50)

        x_min_, y_min_ = max(0, c_x - s), max(0, c_y - s)
        x_max_, y_max_ = min(img.shape[0], c_x + s), min(img.shape[1], c_y + s)

        pad_x, pad_y = (0, 0), (0, 0)
        if (x_max_ - x_min_) < (y_max_ - y_min_):
            pad_x = 0, -((x_max_ - x_min_) - (y_max_ - y_min_))
        elif (x_max_ - x_min_) > (y_max_ - y_min_):
            pad_y = 0, ((x_max_ - x_min_) - (y_max_ - y_min_))

        res = np.pad(img[x_min_:x_max_, y_min_:y_max_], (pad_x, pad_y, (0, 0,)), mode='edge')

        assert res.shape[0] == res.shape[1]

        return res, (x_min_, y_min_, x_max_, y_max_)

    def _preprocess_image(self, img, shape):
        return self.normalize(torch.tensor(cv2.resize(img[:, :, ::-1], shape).transpose((2, 0, 1))).float()[None] / 255).to(self.device)
