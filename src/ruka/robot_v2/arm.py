from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple


class ArmStatus(Enum):
    pass


class ArmControlMode(Enum):
    pass


class AbstractArm(ABC):
    @property
    @abstractmethod
    def sn(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def urdf_path(self) -> str:
        ...

    @property
    @abstractmethod
    def limits(self) -> List[Tuple[float, float]]:
        ...

    @property
    @abstractmethod
    def status(self) -> ArmStatus:
        ...

    @property
    @abstractmethod
    def joints_pos(self) -> List[float]:
        ...

    @abstractmethod
    def steady(self, mode: ArmControlMode):
        ...

    @abstractmethod
    def hold(self):
        ...

    @abstractmethod
    def relax(self):
        ...

    @abstractmethod
    def park(self):
        ...

    @abstractmethod
    def go_home(self):
        ...

    @abstractmethod
    def hw_reset(self):
        ...

    @abstractmethod
    def raise_for_error(self):
        ...

    @abstractmethod
    def set_tcp_vel(self, vel: List[float], angular_vel: List[float]):
        ...

    @abstractmethod
    def set_tcp_pos(self, pos: List[float], angles: List[float]):
        ...

    @abstractmethod
    def set_joints_vel(self, vel: List[float]):
        ...

    @abstractmethod
    def set_joints_pos(self, pos: List[float]):
        ...


