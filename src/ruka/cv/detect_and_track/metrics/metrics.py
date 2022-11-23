from ruka.cv.detect_and_track import DetectAndTrack
from ruka.cv.detect_and_track.datasets import DetectAndTrackEpisode
from collections.abc import Iterable
from typing import Dict


class MetricsComputer: # TODO: store in proper place for the whole project
    def compute_metrics(self) -> Dict[str, float]:
        raise NotImplementedError()


class DetectAndTrackMetricsComputer(MetricsComputer):
    def __init__(
        self,
        detect_and_track: DetectAndTrack,
        dataset: Iterable[DetectAndTrackEpisode]):

        raise NotImplementedError()

    def compute_metrics(self) -> Dict[str, float]:
        ...
