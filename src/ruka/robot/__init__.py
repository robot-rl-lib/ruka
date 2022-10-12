import abc
from typing import List, Tuple

import numpy as np
from math import sin, cos, pi


class RobotError(Exception):
    pass


class Robot(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def disable(self):
        pass

    def isa(self, type):
        return isinstance(self, type)


class Camera(abc.ABC):
    @abc.abstractmethod
    def start(self):
        """
        Start frames capturing pipeline
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """
        Stop frames capturing pipeline
        """
        pass

    @abc.abstractmethod
    def capture(self) -> np.ndarray:
        """
        Capture single multichannel frame
        """
        pass


class CameraError(RobotError):
    pass


class Arm(abc.ABC):
    @property
    @abc.abstractmethod
    def speed(self) -> float:
        """
        Current tool center point cartesian speed in mm/s
        """
        pass

    @property
    @abc.abstractmethod
    def position(self) -> Tuple[float, float, float, float, float, float]:
        """
        Current tool center point cartesian position (x, y, z, roll, pitch, yaw)

        Cartesian position is in mm, rotation is in degrees.
        """
        pass

    @property
    @abc.abstractmethod
    def angles(self) -> List[float]:
        """
        Current arm joints' angles in degrees
        """
        pass


class ArmError(RobotError):
    pass


class ArmVelocityController(Arm, abc.ABC):
    @abc.abstractmethod
    def set_velocity(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """
        Set tool center point cartesian velocity

        :param x: cartesian velocity along X axis in mm/s
        :param y: cartesian velocity along Y axis in mm/s
        :param z: cartesian velocity along Z axis in mm/s
        :param roll: rotation velocity around X axis in degrees/s
        :param pitch: rotation velocity around Y axis in degrees/s
        :param yaw: rotation velocity around Z axis in degrees/s
        """
        pass


class ArmPositionController(Arm, abc.ABC):
    @abc.abstractmethod
    def set_position(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """
        Perform tool center point linear motion to desired position

        Method doesn't wait movement completion. Movement will be interrupted with next method call.

        :param x: cartesian position x in mm
        :param y: cartesian position y in mm
        :param z: cartesian position z in mm
        :param roll: rotation around X axis in degrees
        :param pitch: rotation around Y axis in degrees
        :param yaw: rotation around Z axis in degrees
        """
        pass


class RestrictedPositionMixin:
    def __init__(self):
        self.__xmax = float("inf")
        self.__xmin = float("-inf")
        self.__ymax = float("inf")
        self.__ymin = float("-inf")
        self.__zmax = float("inf")
        self.__zmin = float("-inf")
        self.__bb_xmax = 0
        self.__bb_xmin = 0
        self.__bb_ymax = 0
        self.__bb_ymin = 0
        self.__bb_zmax = 0
        self.__bb_zmin = 0

    @property
    def boundaries(self) -> Tuple[float, float, float, float, float, float]:
        """
        Arm workspace boundaries
        """
        return (self._xmax, self._xmin, self._ymax, self._ymin, self._zmax, self._zmin)

    def set_boundaries(self, xmax: float = ..., xmin: float = ...,
                       ymax: float = ..., ymin: float = ...,
                       zmax: float = ..., zmin: float = ...):
        """
        Restrict arm workspace with boundaries
        """
        if xmax is not Ellipsis:
            self._xmax = xmax
        if xmin is not Ellipsis:
            self._xmin = xmin
        if ymax is not Ellipsis:
            self._ymax = ymax
        if ymin is not Ellipsis:
            self._ymin = ymin
        if zmax is not Ellipsis:
            self._zmax = zmax
        if zmin is not Ellipsis:
            self._zmin = zmin

    def set_bounding_box(self, xmax: float = ..., xmin: float = ...,
                         ymax: float = ..., ymin: float = ...,
                         zmax: float = ..., zmin: float = ...):
        """
        Define tool bounding box
        """
        if xmax is not Ellipsis:
            self._bb_xmax = xmax
        if xmin is not Ellipsis:
            self._bb_xmin = xmin
        if ymax is not Ellipsis:
            self._bb_ymax = ymax
        if ymin is not Ellipsis:
            self._bb_ymin = ymin
        if zmax is not Ellipsis:
            self._bb_zmax = zmax
        if zmin is not Ellipsis:
            self._bb_zmin = zmin

    def _check_boundaries(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        roll = pi * roll / 180
        pitch = pi * pitch / 180
        yaw = pi * yaw / 180
        rx = np.array([
            [1, 0, 0, 0],
            [0, cos(roll), sin(roll), 0],
            [0, -sin(roll), cos(roll), 0],
            [0, 0, 0, 1]
        ])
        ry = np.array([
            [cos(pitch), 0, -sin(pitch), 0],
            [0, 1, 0, 0],
            [sin(pitch), 0, cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        rz = np.array([
            [cos(yaw), sin(yaw), 0, 0],
            [-sin(yaw), cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        t = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        m = t @ rx @ ry @ rz
        bb = [
            np.array([self._bb_xmin, self._bb_ymin, self._bb_zmin, 1]),
            np.array([self._bb_xmax, self._bb_ymin, self._bb_zmin, 1]),
            np.array([self._bb_xmin, self._bb_ymax, self._bb_zmin, 1]),
            np.array([self._bb_xmax, self._bb_ymax, self._bb_zmin, 1]),
            np.array([self._bb_xmin, self._bb_ymin, self._bb_zmax, 1]),
            np.array([self._bb_xmax, self._bb_ymin, self._bb_zmax, 1]),
            np.array([self._bb_xmin, self._bb_ymax, self._bb_zmax, 1]),
            np.array([self._bb_xmax, self._bb_ymax, self._bb_zmax, 1]),
        ]
        for point in bb:
            px, py, pz = (m @ point)[:3]
            if px < self._xmin or px > self._xmax or py < self._ymin or py > self._ymax or pz < self._zmin or pz > self._zmax:
                return False
        return True

    def set_position(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        if not self._check_boundaries(x, y, z, roll, pitch, yaw):
            raise ArmError("Target position violates workspace boundaries")
        super().set_position(x, y, z, roll, pitch, yaw)


class Gripper(abc.ABC):
    @property
    @abc.abstractmethod
    def gripper_position(self) -> int:
        """
        Current gripper position in mm
        """
        pass

    @abc.abstractmethod
    def set_gripper_position(self, position: int):
        """
        Set gripper position

        :param position: gripper position in mm
        """
        pass


class GripperError(RobotError):
    pass
