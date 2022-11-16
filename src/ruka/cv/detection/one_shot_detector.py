import torch
from ruka.cv.datasets import SingleObjectTrackingEpisode


class OneShotDetector(torch.nn.Module):
    def forward(self, query_image, target_image, search_box=None):
        """
        Initialize tracker with target

        Args:
            query_image (np.array): reference image in RGB format
            target_image (np.array): image to search on in RGB format
            search_box: bounding box to crop target image. if None, whole image will be used

        Returns:
            box: detected object bounding box in XYXY format
        """

        raise NotImplementedError()
