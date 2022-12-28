from abc import ABC, abstractmethod
from typing import Callable, List

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
    def __init__(self, force_info: ForceInfo, dt: float, kf: float, force_filter: Callable[[float], float] = default_force_filter) -> None:
        self._force_info = force_info
        self._dt = dt
        self._kf = kf
        self._force_filter = force_filter

    def test_vel(self, vel: Vec3) -> Vec3:
        ext_force = np.array(self._force_info.external_force[:3], dtype=np.float32)
        print(ext_force)
        vel = np.array(vel, dtype=np.float32) + self._force_filter(np.linalg.norm(ext_force)) * self._dt / self._kf * ext_force
        print(vel)
        return list(vel)

