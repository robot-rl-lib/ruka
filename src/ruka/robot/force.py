from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ruka.util.x3d import Vec3
from ruka.robot.collision import CollisionDetector


class ForceInfo(ABC):
    @property
    @abstractmethod
    def external_force(self) -> List[float]:
        ...


def default_force_filter(force: float, min_force=1.0) -> float:
    if force < min_force:
        return 0
    return 1


class ForceTorqueCollisionDetector(CollisionDetector):
    def __init__(self, force_info: ForceInfo, kp:float, kd: float) -> None:
        self._force_info = force_info
        self._kp = kp
        self._kd = kd

    def test_vel(self, vel: Vec3) -> Vec3:
        external_force = np.array(self._force_info.external_force[:3], dtype=np.float32)
        external_force = default_force_filter(np.linalg.norm(external_force)) * external_force
        elastic_force = np.zeros((3,))  # no stiffness
        return list(np.array(vel) + (external_force - elastic_force) / self._kd)


