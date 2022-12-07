import copy
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from pyrealsense2 import pyrealsense2 as rs
from typing import Dict, List

from .sensor_system import SensorSystem, SensorId


@dataclass
class RGBDCameraFrame:
    rgb: NDArray[np.uint8] = None
    depth: NDArray[np.uint16] = None
    infrared: NDArray[np.uint8] = None


@dataclass
class RealsenseMultiCameraSystemConfig:
    @dataclass
    class CameraConfig:
        width: int
        height: int
        streams: List[str] # support 'depth', 'color', 'infrared'
        fps: int

    camera_serials: List[str]
    warmup_frames: int
    default_camera: CameraConfig = None
    serial_to_camera: Dict[str, CameraConfig] = None


class RealsenseMultiCameraSystem(SensorSystem):
    STREAM_TO_RS = {
        'depth': (rs.stream.depth, rs.format.z16),
        'color': (rs.stream.color, rs.format.rgb8),
        'infrared': (rs.stream.infrared, rs.format.y8),
    }

    def __init__(self, config):
        self._config = config
        self._cameras = dict()
        self._frames = dict()

        self._initialize_cameras()
        self._warmup()

    def capture(self) -> Dict[SensorId, RGBDCameraFrame]:
        '''
        Immidiately returns the last set of frames captured.

        Returns:
            frames: dict, mapping sensor id to frame
                realsense sensor id is composed like rs_{serial}
        '''

        for id, pipe in self._cameras.items():
            frames = RealsenseMultiCameraSystem._try_capture_one_camera(pipe)
            if frames is not None:
                self._frames[id] = frames

        return copy.deepcopy(self._frames)

    def _initialize_cameras(self):
        for serial in self._config.camera_serials:
            if self._config.serial_to_camera is not None and serial in self._config.serial_to_camera:
                config = self._config.serial_to_camera[serial]
            else:
                config = self._config.default_camera

            rs_config = rs.config()
            rs_config.enable_device(serial)

            for stream in config.streams:
                rs_stream, rs_format = self.STREAM_TO_RS[stream]
                rs_config.enable_stream(
                    rs_stream,
                    config.width,
                    config.height,
                    rs_format,
                    config.fps)

            pipeline = rs.pipeline()
            pipeline.start(rs_config)
            self._cameras['rs_' + serial] = pipeline

    def _warmup(self):
        for _ in range(self._config.warmup_frames):
            for id, pipe in self._cameras.items():
                frames = pipe.wait_for_frames()
                self._frames[id] = self._parse_frameset(frames)

    @staticmethod
    def _try_capture_one_camera(pipe):
        frames = pipe.poll_for_frames()
        if not frames.is_frameset():
            return None
        return RealsenseMultiCameraSystem._parse_frameset(frames.as_frameset())

    @staticmethod
    def _parse_frameset(frames):
        result = RGBDCameraFrame()

        if frames.get_color_frame().is_frame():
            result.rgb = np.asanyarray(frames.get_color_frame().get_data())
        if frames.get_depth_frame().is_frame():
            result.depth = np.asanyarray(frames.get_depth_frame().get_data())
        if frames.get_infrared_frame().is_frame():
            result.infrared = np.asanyarray(frames.get_infrared_frame().get_data())

        return result
