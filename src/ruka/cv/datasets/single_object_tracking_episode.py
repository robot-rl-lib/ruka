import cv2
import torch
import os
from typing import List
import numpy as np


class SingleObjectTrackingEpisode(torch.utils.data.IterableDataset):
    def __init__(self, base_path: str, frames: List[str], boxes: List[List[int]]):
        """
        Args:
            base_path: path to dataset
            frames: list of paths to frames of the episode relative to base_path
            boxes: list of ground truth bounding boxes of the object in XYXY format
        """

        assert len(frames) == len(boxes)

        self.base_path = base_path
        self.frames = frames
        self.boxes = boxes

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns:
            Dict with the next frame.
            frame: np.array containing image in RGB format
            box: ground truth bounding box in XYXY format
        """

        assert torch.utils.data.get_worker_info() is None

        if self.idx == len(self.frames):
            raise StopIteration

        idx = self.idx
        img = cv2.imread(os.path.join(self.base_path, self.frames[idx]))
        assert img is not None
        img = np.ascontiguousarray(img[:, :, ::-1])

        self.idx += 1
        return {
            'frame': img,
            'box': tuple(self.boxes[idx]),
        }

    def visualize(self, out_file, predicted_boxes=None):
        frame_shape = next(iter(self))['frame'].shape[:2][::-1]
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, frame_shape)

        for i, frame in enumerate(self):
            img = cv2.rectangle(frame['frame'], frame['box'][:2], frame['box'][2:], (0, 255, 0,), 2)
            if predicted_boxes is not None:
                box = predicted_boxes[i]
                img = cv2.rectangle(frame['frame'], box[:2], box[2:], (255, 0, 0,), 2)

            writer.write(img[:, :, ::-1])

        writer.release()
