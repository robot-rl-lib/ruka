import inspect
import logging
import numpy as np
import random
import time
import traceback

from dataclasses import dataclass
from xarm.wrapper import XArmAPI
from xarm.x3.code import APIState

from ruka.util.x3d import Vec3

from .robot import \
    ArmError, ArmInfo, ArmPosControlled, ArmVelControlled, Robot, \
    GripperInfo, GripperPosControlled, not_moving

logging.basicConfig()


# ---------------------------------------------------------------------- API --


@dataclass
class XArmConfig:
    ip: str

    max_speed: float          # L2
    max_angular_speed: float  # Linf
    max_acc: float
    max_jerk: float
    gripper_max_speed: float

    home_pos: Vec3
    home_angles: Vec3
    collision_sensitivity: int
    max_reset_retries: int


class _XArm(ArmInfo, GripperInfo, GripperPosControlled):
    def __init__(self, config: XArmConfig):
        self._config = config
        self._api = _with_xarm_error_handling(XArmAPI(config.ip))         # Handle errors

        self._api.clean_error()                                           # Clean all previous errors
        self._api.clean_warn()                                            # Clean all previous warnings
        self._api.clean_gripper_error()                                   # Clean all previous gripper errors
        self._api.set_allow_approx_motion(True)                           # Allow to avoid overspeed near some singularities using approximate solutions
        self._api.set_collision_rebound(False)                            # Disable rebound after collision
        self._api.set_collision_sensitivity(config.collision_senstivity)  # Set collision sensitivity
        self._api.set_self_collision_detection(True)                      # Enable self collision detection
        self._api.set_collision_tool_model(1)                             # Set XArm Gripper collision tool model
        self._api.set_safe_level(4)                                       # Default safe level
        self._api.set_fence_mode(False)                                   # Disable xarm safety boundaries
        self._api.set_reduced_mode(False)                                 # Disabled joints restrictions
        self._api.set_gripper_speed(config.gripper_max_speed * 10)        # Set current gripper speed

        self._api.set_tcp_maxacc(config.max_acc)
        self._api.set_tcp_jerk(config.max_jerk)
        self._accept_move_commands = False

    # - Robot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def steady(self):
        self.hold()
        self._api.reset()
        self._accept_move_commands = True

    def hold(self):
        raise NotImplementedError()

    def relax(self):
        self._api.reset()
        self._api.set_mode(2)
        self._api.set_state(0)
        self._accept_move_commands = False

    def park(self):
        self._api.reset()
        self._api.set_state(4)
        self._api.motion_enable(False)
        self._accept_move_commands = False

    def go_home(self):
        self._api.motion_enable(True)
        self._api.reset()
        self._api.set_mode(0)
        self._api.set_state(0)
        self._api.set_collision_sensitivity(0)  # Disable collision detection
        retries = 0
        sleep = 0.1
        while True:
            self._reset()
            try:
                x, y, z = self._config.home_pos
                roll, pitch, yaw = self._config.home_angles
                self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)
                self._api.set_collision_sensitivity(self._collision_senstivity)
                self._api.reset()
                break
            except XArmCollisionError:
                if retries >= self._max_reset_retries:
                    raise
                if retries % 5 == 0:
                    sleep *= 2
                logging.error("Raised error during reset")
                traceback.print_exc()
                time.sleep(min(random.expovariate(1.0 / sleep), 0.5))
                retries += 1
        self.hold()

    def check(self):
        if self._api.has_error:
            _raise_for_code(self._api, self._api.error_code)

    # - ArmInfo - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def pos(self) -> Vec3:
        return list(self._api.position)[:3]

    @property
    def speed(self) -> float:
        return self._api.realtime_tcp_speed

    @property
    def acc(self) -> float:
        ...

    @property
    def angles(self) -> Vec3:
        return list(self._api.position)[3:]

    # - GripperInfo - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def gripper_pos(self) -> float:
        return self._api.get_gripper_position()[1] / 10

    @property
    def gripper_speed(self) -> float:
        ...

    @property
    def gripper_acc(self) -> float:
        ...

    # - GripperPosControlled  - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_gripper_pos(self, pos: float):
        if not self._accept_move_commands:
            return
        self._api.set_gripper_position(pos * 10, auto_enable=True, wait=False)


# -------------------------------------------------------------- Controllers --


class XArmPosControlled(_XArm, ArmPosControlled):
    def steady(self):
        super().steady()
        self._api.set_mode(7)   # Enable online trajectory planning
        self._api.set_state(0)  # Prepare to move

    def set_pos(self, pos: Vec3, angles: Vec3):
        if not self._accept_move_commands:
            return
        self._api.set_position(
            x=pos[0], y=pos[1], z=pos[2],
            roll=angles[0], pitch=angles[1], yaw=angles[2],
            speed=self._config.max_speed, is_radian=False, wait=False
        )


class XArmVelControlled(_XArm, ArmVelControlled):
    def reset(self):
        super().steady()
        self._api.set_mode(5)   # Enable velocity control
        self._api.set_state(0)  # Prepare to move

    def set_vel(self, vel: Vec3, angular_vel: Vec3):
        if not self._accept_move_commands:
            return

        # Clip velocity.
        vel = np.array(vel)
        speed = np.linalg.norm(vel)
        if speed > self._config.max_speed:
            vel *= self._max_speed / speed

        # Clip angular velocity.
        angular_vel = np.array(angular_vel)
        angular_speed = np.linalg.norm(angular_vel, np.inf)
        if angular_speed > self._config.max_angular_speed:
            angular_vel *= self._max_angular_vel / vel

        # Set.
        x, y, z = vel
        roll, pitch, yaw = angular_vel
        self._api.vc_set_cartesian_velocity(
            [x, y, z, roll, pitch, yaw], is_radian=False)


# --------------------------------------------------------------- Exceptions --


class XArmError(ArmError):
    def __init__(self, message: str):
        super().__init__(message)


class XArmAPIError(XArmError):
    def __init__(self, code: APIState, message: str = ""):
        super().__init__(f"API error {code}: {message}")
        self.code = code


class XArmControllerError(XArmError):
    def __init__(self, code, message: str = ""):
        super().__init__(f"Controller error {code}: {message}")
        self.code = code


class XArmCollisionError(XArmControllerError):
    def __init__(self, code: int, message: str = ""):
        super().__init__(code, message)


def _with_xarm_error_handling(
        xarm: XArmAPI,
        min_timeout: int = 1,
        max_timeout: int = 8,
        max_retries: int = 5
    ):

    def handle_xarm_errors(method):
        def wrapper(*args, **kwargs):
            retries = 0
            timeout = min_timeout
            while True:
                # Call method.
                res = method(*args, **kwargs)
                if not res:
                    return
                if isinstance(res, tuple) or isinstance(res, list):
                    code = res[0]
                else:
                    code = res

                # Parse error code.
                if code == APIState.NOT_CONNECTED:
                    logging.warning("Not connected")
                    try:
                        xarm.connect()
                        logging.info("Connected. Retrying...")
                    except:
                        logging.warning("Connect failed. Retrying...")
                elif code == APIState.HAS_WARN:
                    logging.warning(f"Controller warning code {xarm.warn_code}")
                    return res
                elif code == APIState.RES_TIMEOUT:
                    logging.warning("Control box response timeout. Retrying...")
                elif code == APIState.NO_TCP:
                    logging.warning("Failed to send command. Retrying...")
                elif code == APIState.NORMAL:
                    logging.debug(f"{method.__name__} successfully executed")
                    return res
                else:
                    _raise_for_code(xarm, code)

                # Sleep and try again
                time.sleep(timeout)
                if timeout < max_timeout:
                    timeout *= 2
                retries += 1
                if retries > max_retries:
                    raise XArmAPIError(code, "Max retries exceeded")

        return wrapper

    for name, method in inspect.getmembers(xarm, inspect.isroutine):
        setattr(xarm, name, handle_xarm_errors(method))

    return xarm


def _raise_for_code(xarm, code):
    if code == APIState.NOT_CONNECTED:
        raise XArmAPIError(code, "API not connected")
    elif code == APIState.RES_TIMEOUT:
        raise XArmAPIError(code, "Control box response timeout")
    elif code == APIState.NO_TCP:
        raise XArmAPIError(code, "Failed to send command")
    elif code == APIState.NOT_READY:
        raise XArmAPIError(code, "Arm is not ready to move")
    elif code == APIState.CMD_NOT_EXIST:
        raise XArmAPIError(code, "Unknown command")
    elif code == APIState.TCP_LIMIT:
        raise XArmAPIError(code, "Tool cartesian position exceeds limit")
    elif code == APIState.JOINT_LIMIT:
        raise XArmAPIError(code, "Joint angle exceeds limit")
    elif code == APIState.OUT_OF_RANGE:
        raise XArmAPIError(code, "Out of range")
    elif code == APIState.EMERGENCY_STOP:
        raise XArmAPIError(code, "Emergency stop occured")
    elif code == APIState.HAS_ERROR:
        error_code = xarm.error_code
        if error_code >= 1 and error_code <= 3:
            raise XArmControllerError(error_code, "Emergency stop occured")
        elif error_code >= 10 and error_code <= 17:
            raise XArmControllerError(error_code, "Servo motor error")
        elif error_code == 18:
            raise XArmControllerError(error_code, "Force torque sensor communication error")
        elif error_code == 19 or error_code == 28:
            raise XArmControllerError(error_code, "End module communication error")
        elif error_code == 21:
            raise XArmControllerError(error_code, "Kinematic error")
        elif error_code == 22:
            raise XArmCollisionError(error_code, "Self collision detected")
        elif error_code == 23:
            raise XArmControllerError(error_code, "Joint angle exceeds limit")
        elif error_code == 24:
            raise XArmControllerError(error_code, "Speed exceeds limit")
        elif error_code == 25:
            raise XArmControllerError(error_code, "Planning error")
        elif error_code == 31:
            raise XArmCollisionError(error_code, "Collision caused abnormal current")
        elif error_code == 35:
            raise XArmControllerError(error_code, "Safety boundary limit")
        else:
            raise XArmControllerError(error_code, "Other error")
    elif code == APIState.CMD_NUM_ERROR:
        raise XArmAPIError(code, "TCP reply number error")
    elif code == APIState.CMD_PROT_ERROR:
        raise XArmAPIError(code, "TCP protocol flag error")
    elif code == APIState.FUN_ERROR:
        raise XArmAPIError(code, "TCP reply command and send command do not match")
    elif code == APIState.RES_LENGTH_ERROR:
        raise XArmAPIError(code, "TCP reply length error")
    else:
        raise XArmAPIError(code, "Other error")