from .robot import Camera

import numpy as np

from pyrealsense2 import pyrealsense2 as rs
from dataclasses import dataclass


@dataclass
class RealsenseConfig:
    width: int
    height: int
    fps: int
    serial_number: str = '146222252031'
    enable_color: bool = True
    enable_infrared: bool = False
    warmup_frames: int = 30  # first captured frames are dark, skip them


class RealsenseCamera(Camera):
    def __init__(self, config: RealsenseConfig):
        self._width = config.width
        self._height = config.height
        self._enable_color = config.enable_color
        self._enable_infrared = config.enable_infrared
        self._warmup_frames = config.warmup_frames

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(config.serial_number)
        self._config.enable_stream(rs.stream.depth, config.width, config.height, rs.format.z16, config.fps)
        if self._enable_color:
            self._config.enable_stream(rs.stream.color, config.width, config.height, rs.format.rgb8, config.fps)
        if self._enable_infrared:
            self._config.enable_stream(rs.stream.infrared, config.width, config.height, rs.format.y8, config.fps)

    def start(self):
        self._pipeline.start(self._config)
        for _ in range(self._warmup_frames):
            self.capture()

    def stop(self):
        self._pipeline.stop()

    def capture(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()

        result = []
        if self._enable_color:
            result.append(np.asanyarray(frames.get_color_frame().get_data()))

        result.append(np.asanyarray(frames.get_depth_frame().get_data()))

        if self._enable_infrared:
            result.append(np.asanyarray(frames.get_infrared_frame().get_data()))

        return np.dstack(result)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
