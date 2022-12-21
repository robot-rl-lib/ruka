from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from urdfpy import URDF

from ruka.robot_v2.sensor import SensorSystem
from ruka.robot_v2.arm import AbstractArm, ArmControlMode, ArmStatus
from ruka.robot_v2.gripper import AbstractGripper


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


# -------------------------------------------------------------------- Robot --


@dataclass
class RobotConfig:
    pass


@dataclass
class RobotState:
    status: ArmStatus
    joints_pos: List[float]
    gripper_pos: float
    sensors: Dict[str, Any]


class Robot:
    def __init__(self, arm: AbstractArm, gripper: AbstractGripper, sensors: SensorSystem):
        self._arm = arm
        self._gripper = gripper
        self._sensors = sensors

    @property
    def sn(self) -> Dict[str, str]:
        return {
            'arm': self._arm.sn,
            'gripper': self._gripper.sn,
        }

    @property
    def model_name(self) -> Dict[str, str]:
        return {
            'arm': self._arm.model_name,
            'gripper': self._gripper.model_name,
        }

    @property
    def urdf(self) -> URDF:
        arm_urdf = URDF.load(self._arm.urdf_path)
        gripper_urdf = URDF.load(self._gripper.urdf_path)
        return URDF(
            links=arm_urdf.links + gripper_urdf.links,
            joints=arm_urdf.joints + gripper_urdf.joints,
            transmissions=arm_urdf.transmissions + gripper_urdf.transmissions,
        )

    @property
    def joints_limits(self) -> List[Tuple[float, float]]:
        return self._arm.limits

    @property
    def gripper_pos_limits(self) -> Tuple[float, float]:
        return self._gripper.limits

    def get_state(self, raise_for_error=True) -> RobotState:
        if raise_for_error:
            self._arm.raise_for_error()
            self._gripper.raise_for_error()

        return RobotState(
            sensors = self._sensors.capture(),
            joints_pos = self._arm.joints_pos,
            gripper_pos = self._gripper.pos,
            status = self._arm.status,
        )

    def register_state_callback(self, cb: Callable[[RobotState], None]):
        ...

    def register_error_callback(self, cb: Callable[[RobotError], None]):
        ...

    @staticmethod
    def from_config(config: RobotConfig) -> Robot:
        ...


class RobotControl(Robot):
    def __init__(self, arm: AbstractArm, gripper: AbstractGripper, sensors: SensorSystem):
        # TODO: Make access exclusive
        super().__init__(arm, gripper, sensors)

    def steady(self, mode: ArmControlMode):
        self._arm.steady(mode)

    def hold(self):
        self._arm.hold()

    def relax(self):
        self._arm.relax()

    def park(self):
        self._arm.park()

    def go_home(self):
        self._arm.go_home()
        self._gripper.go_home()

    def hw_reset(self):
        self._arm.hw_reset()
        self._gripper.hw_reset()

    def set_tcp_vel(self, vel: List[float], angular_vel: List[float]):
        # TODO: Remove this as soon as we have IK
        self._arm.set_tcp_vel(vel, angular_vel)

    def set_tcp_pos(self, pos: List[float], angles: List[float]):
        # TODO: Remove this as soon as we have IK
        self._arm.set_tcp_pos(pos, angles)

    def set_joints_vel(self, vel: List[float]):
        self._arm.set_joints_vel(vel)

    def set_joints_pos(self, pos: List[float]):
        self._arm.set_joints_pos(pos)

    def set_gripper_pos(self, pos: float):
        self._gripper.set_pos(pos)

    @staticmethod
    def from_config(config: RobotConfig) -> RobotControl:
        ...

