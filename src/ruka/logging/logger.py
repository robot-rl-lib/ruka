from dataclasses import dataclass
from numpy.typing import ArrayLike
from ruka.util.json import JSONSerializable
from typing import Any, Collection, Optional


# ------------------------------------------------------------------- Logger --


class Logger:
    """
    Tensorboard-style logger.
    """

    def add_scalar(self, key: str, value: float, step_no: int = -1):
        """
        Store scalar.
        """
        raise NotImplementedError()

    def add_text(self, key: str, value: str, step_no: int = -1):
        """
        Store a text.
        """
        raise NotImplementedError()

    def add_video_frame(self, key: str, value: ArrayLike, step_no: int = -1):
        """
        Store a video frame. We may combine them into a video.

        'value': [height, width, 1] or [height, width, 3], dtype=uint8.

        Raise an exception if frame dimensions differ between steps.
        """
        raise NotImplementedError()

    def add_data(self, key: str, value: Any, step_no: int = -1):
        """
        Add data that is to be serialized (pickled) without any changes. Later
        it could be restored exactly the same way from the archive.
        """
        raise NotImplementedError()

    def add_metadata(self, key: str, value: JSONSerializable):
        """
        Add some metadata.
        """
        raise NotImplementedError()

    def set_video_fps(self, video_fps: 'FPSParams'):
        """
        If you have videos, you MUST call this function, otherwise close()
        will return an error.

        You can call this function any time before close().
        """
        raise NotImplementedError()

    def step(self, step_no: int = -1):
        """
        Set current step_no.
        Or switch to next step if passed step_no == -1.
        """
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def __enter__(self) -> 'Logger':
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_value, tb):
        raise NotImplementedError()


def create_ruka_logger(local_path: str) -> Logger:
    """
    local_path shouldn't exist.
    A folder, located at local_path, will be created.
    """
    from .impl.logger_ruka import RukaLogger
    return RukaLogger(local_path)


def create_tensorboard_logger(local_path: str) -> Logger:
    """
    A folder, located at local_path, will be created, if not exists.
    local_path can contain data from previous tensorboard logger.
    """
    from .impl.logger_tensorboard import TensorboardLogger
    return TensorboardLogger(local_path)


def create_dummy_logger() -> Logger:
    from .impl.logger_dummy import DummyLogger
    return DummyLogger()


@dataclass
class FPSParams:
    """
    There are multiple methods of setting fps, but only one can be used
    at the same time:

    - You can set fps explicitly;
    - You can pass dt, an inverse of fps;
    - You can pass total_time, and fps will be computed automatically
        from the largest step_no.
    """

    fps: float = -1
    dt: float = -1
    total_time: float = -1

    def __post_init__(self):
        has_fps = (self.fps > 0)
        has_dt = (self.dt > 0)
        has_total_time = (self.total_time > 0)
        assert int(has_fps) + int(has_dt) + int(has_total_time) == 1

    def get_fps(self, num_steps: int) -> int:
        if self.fps > 0:
            return self.fps
        if self.dt > 0:
            return round(1. / self.dt)
        if self.total_time > 0:
            return round(num_steps / self.total_time)
        assert 0


# ---------------------------------------------------------------- LogReader --


class DataType:
    SCALAR = 'scalar'
    TEXT = 'text'
    VIDEO = 'video'
    DATA = 'data'
    METADATA = 'metadata'


class LogReader:
    def get_keys(self, datatype: Optional[DataType] = None) -> Collection[str]:
        """
        If datatype=None, return all keys.
        If datatype=<some datatype>, return all keys of specific datatype.
        """
        raise NotImplementedError()

    def get_datatype(self, key: str) -> DataType:
        raise NotImplementedError()

    def get_value(self, key: str) -> Any:
        """
        For 'scalar', return Dict[int, float]
        For 'text', return Dict[int, str]
        For 'video', return path to a MP4 video file
        For 'data', return Dict[int, Any]
        For 'metadata', return JSONSerializable

        Raise KeyError if key doesn't exist.
        """
        raise NotImplementedError()

    def get_max_step_no(self) -> int:
        raise NotImplementedError()

    def get_video_info(self, key: str) -> 'VideoInfo':
        raise NotImplementedError()


def create_ruka_log_reader(local_path: str) -> LogReader:
    """
    local_path must exist and be a directory.

    local_path must exist while the LogReader is used.
    """
    from .impl.logger_ruka import RukaLogReader
    return RukaLogReader(local_path)


@dataclass
class VideoInfo:
    height: int
    width: int
    channels: int
    fps: int