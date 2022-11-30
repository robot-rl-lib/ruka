import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.util.migrating import Migrating
from ruka.util.distributed_fs import download_pickle, upload_pickle
from ruka.util.saved_by_remote_path import SavedByRemotePath


@dataclass
class RGBDCameraCalibration(Migrating):
    depth_intrinsics: NDArray[np.float]  # (3, 3,) intrinsics matrix
    color_intrinsics: NDArray[np.float]  # (3, 3,) intrinsics matrix

    depth_to_color: NDArray[np.float]  # (4, 4,) transformation matrix


@dataclass
class RobotCalibration(Migrating):
    gripper_camera: RGBDCameraCalibration

    gripper_camera_to_tcp: NDArray[np.float]  # (4, 4,) transformation matrix


@dataclass
class SavedRobotCalibration(SavedByRemotePath):
    calibration: RobotCalibration
    remote_path: str

    def save(self):
        upload_pickle(self.calibration, self.remote_path)

    @staticmethod
    def load(remote_path: str) -> 'SavedRobotCalibration':
        calibration = download_pickle(remote_path)
        return SavedRobotCalibration(calibration=calibration, remote_path=remote_path)
