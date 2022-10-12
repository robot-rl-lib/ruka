from . import Camera

import numpy as np
from pyrealsense2 import pyrealsense2 as rs


class RealsenseCamera(Camera):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    def start(self):
        self._pipeline.start(self._config)

    def stop(self):
        self._pipeline.stop()

    def capture(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return np.dstack([color, depth])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height