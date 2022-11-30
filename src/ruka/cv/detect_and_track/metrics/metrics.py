import numpy as np

from ruka.cv.detect_and_track import DetectAndTrack
from ruka.cv.detect_and_track.datasets import DetectAndTrackEpisode
from ruka.cv.util import box_iou
from typing import Dict, Iterable


class MetricsComputer: # TODO: store in proper place for the whole project
    def compute_metrics(self) -> Dict[str, float]:
        raise NotImplementedError()


class DetectAndTrackMetricsComputer(MetricsComputer):
    def __init__(
        self,
        detect_and_track: DetectAndTrack,
        dataset: Iterable[DetectAndTrackEpisode],
        detection_iou_threshold: float = 0.5,
        tracking_iou_threshold: float = 0.3):

        self._detect_and_track = detect_and_track
        self._dataset = dataset

        self._detection_iou_threshold = detection_iou_threshold
        self._tracking_iou_threshold = tracking_iou_threshold

    def compute_metrics(self) -> Dict[str, float]:
        detection_success = []
        tracking_statistics = []
        for episode in self._dataset:
            boxes, gt_boxes = self._process_episode(episode)
            detection_success.append(self._detection_successful(boxes, gt_boxes))
            tracking_statistics.append(self._tracking_frame_success_statistics(boxes, gt_boxes))

        detection_success = np.array(detection_success)
        tracking_statistics = np.array(tracking_statistics)
        return {
            'detection_success_rate': np.mean(detection_success),
            'detect_and_track_success_rate': np.sum(tracking_statistics[:, 1]) / np.sum(tracking_statistics[:, 0]),
            'tracking_success_rate_where_detected':
                np.sum(tracking_statistics[detection_success, 1]) / np.sum(tracking_statistics[detection_success, 0]),
        }

    def _process_episode(self, episode):
        self._detect_and_track.reset(episode.target)

        boxes, gt_boxes = [], []
        for frame in episode.episode:
            boxes.append(self._detect_and_track.find(frame.frame))
            gt_boxes.append(frame.box)

        return boxes, gt_boxes

    def _detection_successful(self, boxes, gt_boxes):
        iou = box_iou(
            boxes[0].to_axis_aligned().to_xyxy(),
            gt_boxes[0].to_axis_aligned().to_xyxy())
        return iou > self._detection_iou_threshold

    def _tracking_frame_success_statistics(self, boxes, gt_boxes):
        total, good = 0, 0
        for box, gt_box in zip(boxes, gt_boxes):
            iou = box_iou(
                box.to_axis_aligned().to_xyxy(),
                gt_box.to_axis_aligned().to_xyxy())

            if iou > self._tracking_iou_threshold:
                good += 1
            total += 1

        return total, good
