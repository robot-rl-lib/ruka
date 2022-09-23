import abc
from typing import Tuple

import numpy as np
from math import sin, cos, pi


class RobotError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ArmError(RobotError):
    def __init__(self, message: str):
        super().__init__(message)


class GripperError(RobotError):
    def __init__(self, message: str):
        super().__init__(message)


class Arm(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        """
        Clear errors and prepare to move
        """
        raise NotImplementedError

    @abc.abstractmethod
    def disable(self):
        """
        Disable arm
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def speed(self) -> float:
        """
        Move speed in mm/s
        """
        raise NotImplementedError

    @speed.setter
    @abc.abstractmethod
    def speed(self, speed: float):
        """
        Set move speed

        :param speed: move speed in mm/s
        """
        raise NotImplementedError

    @abc.abstractmethod
    def move(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self) -> Tuple[float, float, float, float, float, float]:
        """
        Current tool center point cartesian position (x, y, z, roll, pitch, yaw)

        Cartesian position is in mm, rotation is in degrees.
        """
        raise NotImplementedError


class RestrictedArm(Arm):
    def __init__(self, arm: Arm):
        self._arm = arm
        self._xmax = float("inf")
        self._xmin = float("-inf")
        self._ymax = float("inf")
        self._ymin = float("-inf")
        self._zmax = float("inf")
        self._zmin = float("-inf")
        self._bb_xmax = 0
        self._bb_xmin = 0
        self._bb_ymax = 0
        self._bb_ymin = 0
        self._bb_zmax = 0
        self._bb_zmin = 0

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

    def reset(self):
        self._arm.reset()

    def disable(self):
        self._arm.disable()

    @property
    def speed(self) -> float:
        return self._arm.speed

    @speed.setter
    def speed(self, speed: float):
        self._arm.speed = speed

    @property
    def position(self) -> Tuple[float, float, float, float, float, float]:
        return self._arm.position

    def _check_boundaries(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        roll = pi * roll / 180
        pitch = pi * pitch / 180
        yaw = pi * yaw / 180
        rx = np.array([
            [1, 0, 0, 0],
            [0, cos(roll), -sin(roll), 0],
            [0, sin(roll), cos(roll), 0],
            [0, 0, 0, 1]
        ])
        ry = np.array([
            [cos(pitch), 0, sin(pitch), 0],
            [0, 1, 0, 0],
            [-sin(pitch), 0, cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        rz = np.array([
            [cos(yaw), -sin(yaw), 0, 0],
            [sin(yaw), cos(yaw), 0, 0],
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

    def move(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        if not self._check_boundaries(x, y, z, roll, pitch, yaw):
            raise ArmError("Target position violates workspace boundaries")
        self._arm.move(x, y, z, roll, pitch, yaw)


class Gripper(abc.ABC):
    @property
    @abc.abstractmethod
    def speed(self) -> int:
        """
        Grip speed in mm/s
        """
        raise NotImplementedError

    @speed.setter
    @abc.abstractmethod
    def speed(self, speed: int):
        """
        Set grip speed

        :param speed: grip speed in mm/s
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def force(self) -> float:
        """
        Grip force
        """
        raise NotImplementedError

    @force.setter
    @abc.abstractmethod
    def force(self, force: float):
        """
        Set grip force
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self) -> int:
        """
        Current gripper position in mm
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_position(self, position: int):
        """
        Set gripper position

        :param position: gripper position in mm
        """
        raise NotImplementedError


class Robot(object):
    def __init__(self, arm: Arm, gripper: Gripper):
        self._arm = arm
        self._gripper = gripper

    @property
    def arm(self) -> Arm:
        return self._arm

    @property
    def gripper(self) -> Gripper:
        return self._gripper

    def reset(self):
        self.arm.reset()

    def disable(self):
        self.arm.disable()

