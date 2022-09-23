import enum
import torch

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Union


Vec3 = List
Quat = List


# ---------------------------------------------------------------- Simulator --


class Simulator:
    def reset(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def run_for(self, num_seconds: int):
        """
        Number of steps is rounded down.
        """
        raise NotImplementedError()
    
    def add_object(self, 
            config: 'ObjectConfig',
            position: Vec3, 
            orientation: Quat
        ) -> 'Object':
        raise NotImplementedError()

    @property
    def objects(self) -> Iterable['Object']:
        raise NotImplementedError()

    def add_camera(
            self, 
            config: 'CameraConfig', 
            position: Vec3, 
            orientation: Quat
        ) -> 'Camera':
        raise NotImplementedError()

    @property
    def cameras(self) -> Iterable['Camera']:
        raise NotImplementedError()

    def poll(
            self,
            entities: Iterable[Union['Link', 'Joint']], 
            *properties: Union['LinkProperty', 'JointProperty']
        ):
        pass

    def close(self):
        raise NotImplementedError()


# ------------------------------------------------------------------- Object --


class Object:
    @property
    def simulator(self) -> Simulator:
        raise NotImplementedError()

    @property
    def config(self) -> 'ObjectConfig':
        raise NotImplementedError()

    @property
    def root(self) -> 'Link':
        raise NotImplementedError()

    @property
    def links(self) -> Mapping[str, 'Link']:
        raise NotImplementedError()

    @property
    def joints(self) -> Mapping[str, 'Joint']:
        raise NotImplementedError()

    def remove(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class ObjectConfig:
    """
    List of formats that must be supported:

    - URDF
    - SDF
    """

    definition_file_path: str
    static: bool = False
    scale: float = 1.


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  Link -


class Link:
    @property
    def simulator(self) -> Simulator:
        return self.object.simulator

    @property
    def object(self) -> Object:
        raise NotImplementedError()

    @property
    def config(self) -> 'LinkConfig':
        raise NotImplementedError()
    
    @property
    def position(self) -> Vec3:
        raise NotImplementedError()

    @property
    def orientation(self) -> Quat:
        raise NotImplementedError()

    def poll(self, *properties: 'LinkProperty'):
        pass


@dataclass(frozen=True)
class LinkConfig:
    name: str


class LinkProperty(enum.Enum):
    POSITION = 'position'
    ORIENTATION = 'orientation'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Joint -


class Joint:
    @property
    def simulator(self) -> Simulator:
        return self.object.simulator

    @property
    def object(self) -> Object:
        raise NotImplementedError()

    @property
    def config(self) -> 'JointConfig':
        raise NotImplementedError()

    @property
    def position(self) -> Vec3:
        raise NotImplementedError()

    @property
    def control_mode(self) -> 'ControlMode':
        """
        Returns ControlMode.DISABLE for fixed joints.
        """
        raise NotImplementedError()

    @control_mode.setter
    def control_mode(self, value: 'ControlMode'):
        """
        Exception is raised for JointType.FIXED joints.

        Every time control_mode is set (regardless of whether the actual 
        value changed), control_target is set to None.
        """
        raise NotImplementedError()

    @property
    def control_target(self) -> Optional[float]:
        """
        Initially, control_target is None.

        Returns None for fixed joints.
        """
        raise NotImplementedError()

    @control_target.setter
    def control_target(self, value: Optional[float]):
        """
        If set to None, joint motors are disabled, regardless of control_mode.

        Exception is raised for JointType.FIXED joints.

        If control_mode is set to DISABLE, any value except None is invalid.
        """
        raise NotImplementedError()

    @property
    def max_effort(self) -> float:
        """
        Initial value is taken from URDF/SDF/... and is the same as in 
        self.config.max_effort. 
        
        Can be later overridden through assignment.
        """
        raise NotImplementedError()

    @max_effort.setter
    def max_effort(self, value: float):
        raise NotImplementedError()

    def poll(self, *properties: 'JointProperty'):
        pass


@dataclass(frozen=True)
class JointConfig:
    name: str
    type: 'JointType'
    lower_limit: float
    upper_limit: float
    max_effort: float


class JointType(enum.Enum):
    REVOLUTE = 'revolute'
    PRISMATIC = 'prismatic'
    FIXED = 'fixed'


class JointProperty(enum.Enum):
    POSITION = 'position'


class ControlMode(enum.Enum):
    DISABLE = 'disable'
    POSITION_PD = 'position'


# ------------------------------------------------------------------- Camera --


class Camera:
    @property
    def simulator(self) -> Simulator:
        raise NotImplementedError()

    @property
    def config(self) -> 'CameraConfig':
        raise NotImplementedError()

    @property
    def position(self) -> Vec3:
        raise NotImplementedError()

    @position.setter
    def position(self, value: Vec3):
        raise NotImplementedError()

    @property
    def orientation(self) -> Quat:
        raise NotImplementedError()

    @orientation.setter
    def orientation(self, value: Quat):
        raise NotImplementedError()

    def render(self) -> torch.Tensor:
        raise NotImplementedError()

    def remove(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class CameraConfig:
    w: int
    h: int
    hfov_deg: float
    near: float
    far: float
    image_type: 'ImageType'
    attach_to: Optional[Link] = None


class ImageType(enum.Enum):
    RGB = 'rgb'
    DEPTH = 'depth'