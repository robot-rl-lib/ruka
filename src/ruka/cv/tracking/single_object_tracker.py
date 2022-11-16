import torch
from ruka.cv.datasets import SingleObjectTrackingEpisode


class SingleObjectTracker(torch.nn.Module):
    def initialize(self, image, box):
        """
        Initialize tracker with target

        Args:
            image (np.array): image in RGB format
            box: target object bounding box coordinates in XYXY format

        Returns:
            None
        """

        raise NotImplementedError()

    def forward(self, image, box = None):
        """
        Perform tracking on next frame

        Args:
            image (np.array): image in RGB format
            box: optional search bbox in XYXY format
        Returns:
            box: target object bounding box coordinates in XYXY format
        """

        raise NotImplementedError()

    def process_episode(self, episode: SingleObjectTrackingEpisode, first_box=None):
        """
        Run tracker inference on tracking episode

        Args:
            episode (SingleObjectTrackingEpisode): episode
            first_box: bounding box to init with in XYXY format. If None, ground truth box will be taken

        Returns:
            boxes: predicted boxes for each frame in XYXY format
        """

        result = []
        for i, d in enumerate(episode):
            if i == 0:
                self.initialize(d['frame'], d['box'] if first_box is None else first_box)

            result.append(self(d['frame']))

        return result
