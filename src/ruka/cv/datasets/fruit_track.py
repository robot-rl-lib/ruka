import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from ruka.cv.datasets import SingleObjectTrackingEpisode


class FruitTrack(Dataset):
    def __init__(self, path):
        super().__init__()

        self.path = path

        with open(os.path.join(self.path, 'info.json'), 'r') as f:
            meta = json.load(f)

        self.episodes = []
        for ep in meta['episodes']:
            self.episodes.append({
                'frames': meta['runs'][ep['run']],
                'boxes': ep['boxes'],
                'object': ep['object'],
            })

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        obj = self.episodes[idx]['object']

        reference = cv2.imread(os.path.join(self.path, 'reference', f'{obj}.jpeg'))
        assert reference is not None
        reference = np.ascontiguousarray(reference[:, :, ::-1])

        return {
            'episode': SingleObjectTrackingEpisode(self.path, self.episodes[idx]['frames'], self.episodes[idx]['boxes']),
            'object': obj,
            'reference': reference,
        }
