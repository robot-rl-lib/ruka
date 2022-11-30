import abc
import enum
import numpy as np

from typing import Tuple
from ruka.util.x3d import Vec3


class ControlMode(enum.Enum):
    POS = 'position'
    VEL = 'velocity'


# -------------------------------------------------------------------- Robot --


class Robot:
    def steady(self, control_mode=False):
        """
        Stop the arm and get ready to move.

        This is the only mode in which the arm responds to movement commands.
        In all other modes, move commands are silently ignored.

        Has "wait=True" semantics: waits until the mode transition is complete
        before returning.
        """
        raise NotImplementedError()

    def hold(self):
        """
        Stop the arm and do not respond to commands.

        Has "wait=True" semantics: waits until the mode transition is complete
        before returning.
        """
        raise NotImplementedError()

    def relax(self):
        """
        Relax the arm so it can be manually moved by a human.

        Has "wait=True" semantics: waits until the mode transition is complete
        before returning.
        """
        raise NotImplementedError()

    def park(self):
        """
        Engage brakes and disable motors.

        Has "wait=True" semantics: waits until the mode transition is complete
        before returning.
        """
        raise NotImplementedError()

    def go_home(self):
        """
        Go to initial position and go to hold() mode after it is reached.

        Should be called after RobotRecoverableError has occurred.

        Has "wait=True" semantics: waits until the mode transition is complete
        before returning.
        """
        raise NotImplementedError()

    def check(self):
        """
        If an exception occurred while moving (e.g. due to collision), raise it.

        When exception occurs, the robot no longer accepts move commands until
        the mode is explicitly changed using the methods above. Furthermore, an
        attempt to issue a move commands MUST lead to an exception being raised,
        the same exception raised by check().

        The exception is reset when robot is told to go to some mode: steady(),
        hold(), etc.
        """
        raise NotImplementedError()


# ---------------------------------------------------------------------- Arm --


class ArmInfo:
    @property
    def pos(self) -> Vec3:
        raise NotImplementedError()

    @property
    def speed(self) -> float:
        raise NotImplementedError()

    @property
    def acc(self) -> float:
        raise NotImplementedError()

    @property
    def angles(self) -> Vec3:
        raise NotImplementedError()

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        """
        Return (low, high).

            low = [x_min, y_min, z_min]
            high = [x_max, y_max, z_max]

        min == max = 0 is a valid limit.
        """
        raise NotImplementedError()

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        """
        Return (low, high).

            low = [roll_min, pitch_min, yaw_min]
            high = [roll_max, pitch_max, yaw_max]

        min == max = 0 is a valid limit.
        """


class ArmPosControlled(Robot):
    def __init__(self):
        self._target_pos = None
        self._target_angles = None

    def set_pos(self, pos: Vec3, angles: Vec3):
        """
        Set target cartesian position for the tool center point.

        - 'pos': xyz in mm
        - 'angles': rpy in deg

        The target position is reset when the robot leaves ready-to-move mode.

        Has "wait=False" semantics: does NOT wait until the movement is
        complete before returning.
        """
        raise NotImplementedError()

    def is_target_reached(self, pos_tolerance=10, angles_tolerance=10):
        """
        Check if target position is reached.

        - 'pos_tolerance': max L2 norm of difference between target and real positions
        - 'angles': max L2 norm of angle difference vector between target and real angles
        """

        raise NotImplementedError()


class ArmVelControlled(Robot):
    def set_vel(self, vel: Vec3, angular_vel: Vec3):
        """
        Set tool center point cartesian velocity.

        - 'vel': xyz in mm/s
        - 'angular_vel': rpy in deg/s

        The target velocity is reset when the robot leaves ready-to-move mode.

        Has "wait=False" semantics: does NOT wait until the movement is
        complete before returning.
        """
        raise NotImplementedError()


class ArmPosVelControlled(ArmPosControlled,ArmVelControlled):
    pass

# ------------------------------------------------------------------ Gripper --


class GripperInfo:
    @property
    def gripper_pos(self) -> float:
        raise NotImplementedError()

    @property
    def gripper_speed(self) -> float:
        raise NotImplementedError()

    @property
    def gripper_acc(self) -> float:
        raise NotImplementedError()

    @property
    def gripper_pos_limits(self) -> Tuple[float, float]:
        """
        Return (low, high).
        """
        raise NotImplementedError()


class GripperPosControlled(Robot):
    def set_gripper_pos(self, pos: float):
        """
        Set the gripper target position (in mm).

        The target position is reset when the robot leaves ready-to-move mode.

        Has "wait=False" semantics: does NOT wait until the movement is
        complete before returning.
        """
        raise NotImplementedError()


# --------------------------------------------------------------- Camera ------

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

    @property
    @abc.abstractmethod
    def width(self):
        """
        Return width of the frame
        """
        pass

    @property
    @abc.abstractmethod
    def height(self):
        """
        Return height of the frame
        """
        pass


# --------------------------------------------------------------- Exceptions --


class RobotError(Exception):
    """
    Generic robot error
    """
    pass


class RobotRecoverableError(RobotError):
    """
    Recoverable robot error

    To perform robot recovery consider calling go_home method
    """
    pass


class RobotUnrecoverableError(RobotError):
    """
    Unrecoverable robot error

    Manual robot operation and restart are required
    """
    pass

