from .impl.tracking.single_object_tracker import SingleObjectTracker
from .impl.detection.one_shot_detector import OneShotDetector
from .detect_and_track import DetectAndTrack


def create_generic_detect_and_track(
    detector: OneShotDetector,
    tracker: SingleObjectTracker) -> DetectAndTrack:

    from .impl.generic_detect_and_track import GenericDetectAndTrack
    return GenericDetectAndTrack(detector, tracker)
