import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np


class FruitTrackEpisode(Dataset):
    def __init__(self, episode_path):
        super().__init__()

        def read_boxes(path):
            res = []
            with open(path, 'r') as f:
                for l in f:
                    t = tuple(map(int, l.split()))
                    res.append((t[1], t[0], t[3], t[2],))
            return res

        self.rgb = []
        self.boxes = []

        frame = 0
        while True:
            frame_str = str(frame).zfill(4)
            if not os.path.exists(os.path.join(episode_path, frame_str + '_rgb.jpg')):
                break

            self.rgb.append(cv2.imread(os.path.join(episode_path, frame_str + '_rgb.jpg')))
            self.boxes.append(read_boxes(os.path.join(episode_path, frame_str + '_boxes.txt')))

            frame += 1

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        return {
            'rgb': self.rgb[idx],
            'boxes': self.boxes[idx],
        }
