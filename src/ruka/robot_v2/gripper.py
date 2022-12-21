from abc import ABC, abstractmethod
from typing import Tuple


class AbstractGripper(ABC):
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
    def limits(self) -> Tuple[float, float]:
        ...

    @property
    @abstractmethod
    def pos(self) -> float:
        ...

    @abstractmethod
    def raise_for_error(self):
        ...

    @abstractmethod
    def set_pos(self, pos: float):
        ...

    @abstractmethod
    def go_home(self):
        ...

    @abstractmethod
    def hw_reset(self):
        ...
