import inspect
import logging
import numpy as np
import random
import time
import traceback
from typing import List, Optional, Tuple
from . import ArmPositionController, ArmVelocityController, RestrictedPositionMixin, Robot, ArmError, Arm, Gripper

from xarm.wrapper import XArmAPI
from xarm.x3.code import APIState

logging.basicConfig()


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


def with_xarm_error_handling(xarm: XArmAPI, min_timeout: int = 1, max_timeout: int = 8, max_retries: int = 5):
    def handle_xarm_errors(method):
        def wrapper(*args, **kwargs):
            retries = 0
            timeout = min_timeout
            while True:
                res = method(*args, **kwargs)
                if not res:
                    return
                if isinstance(res, tuple) or isinstance(res, list):
                    code = res[0]
                else:
                    code = res

                if code == APIState.NOT_CONNECTED:
                    logging.warning("Not connected")
                    # Try reconnect
                    try:
                        xarm.connect()
                        logging.info("Connected. Retrying...")
                    except:
                        logging.warning("Connect failed. Retrying...")
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
                elif code == APIState.HAS_WARN:
                    logging.warning(f"Controller warning code {xarm.warn_code}")
                    return res
                elif code == APIState.RES_TIMEOUT:
                    logging.warning("Control box response timeout. Retrying...")
                elif code == APIState.NO_TCP:
                    logging.warning("Failed to send command. Retrying...")
                elif code == APIState.CMD_NUM_ERROR:
                    raise XArmAPIError(code, "TCP reply number error")
                elif code == APIState.CMD_PROT_ERROR:
                    raise XArmAPIError(code, "TCP protocol flag error")
                elif code == APIState.FUN_ERROR:
                    raise XArmAPIError(code, "TCP reply command and send command do not match")
                elif code == APIState.RES_LENGTH_ERROR:
                    raise XArmAPIError(code, "TCP reply length error")
                elif code == APIState.NORMAL:
                    logging.debug(f"{method.__name__} successfully executed")
                    return res
                else:
                    raise XArmAPIError(code, "Other error")

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


class XArm(Robot, Arm, Gripper):
    def __init__(self, ip: str, gripper_speed: int = 50, collision_sensitivity: int = 3, max_reset_retries: int = 10, home: Optional[Tuple[float, float, float, float, float, float]] = None):
        self._home = home
        self._gripper_speed = gripper_speed
        self._collision_sensitivity = collision_sensitivity
        self._max_reset_retries = max_reset_retries
        self._api = with_xarm_error_handling(XArmAPI(ip))                # Handle errors
        self._api.clean_error()                                          # Clean all previous errors
        self._api.clean_warn()                                           # Clean all previous warnings
        self._api.clean_gripper_error()                                  # Clean all previous gripper errors
        self._api.set_allow_approx_motion(True)                          # Allow to avoid overspeed near some singularities using approximate solutions
        self._api.set_collision_rebound(False)                           # Disable rebound after collision
        self._api.set_collision_sensitivity(self._collision_sensitivity)  # Set collision sensitivity
        self._api.set_self_collision_detection(True)                     # Enable self collision detection
        self._api.set_collision_tool_model(1)                            # Set XArm Gripper collision tool model
        self._api.set_safe_level(4)                                      # Default safe level
        self._api.set_fence_mode(False)                                  # Disable xarm safety boundaries
        self._api.set_reduced_mode(False)                                # Disabled joints restrictions
        self._api.set_gripper_speed(self._gripper_speed * 10)            # Set current gripper speed

    def reset(self):
        self._api.motion_enable(True)
        self._reset_and_go_home()

    def disable(self):
        self._api.set_state(4)
        self._api.motion_enable(False)
        self._api.disconnect()

    @property
    def position(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(self._api.position)

    @property
    def speed(self) -> int:
        return self._api.realtime_tcp_speed

    @property
    def angles(self) -> List[float]:
        return self._api.angles

    @property
    def collision_sensitivity(self) -> int:
        return self._collision_sensitivity

    @collision_sensitivity.setter
    def collision_sensitivity(self, collision_sensitivity: int):
        self._collision_sensitivity = collision_sensitivity
        self._api.set_collision_sensitivity(self._collision_sensitivity)

    @property
    def gripper_speed(self) -> int:
        return self._gripper_speed

    @gripper_speed.setter
    def gripper_speed(self, speed: int):
        self._gripper_speed = speed
        self._api.set_gripper_speed(self._gripper_speed * 10)

    @property
    def gripper_position(self) -> int:
        return self._api.get_gripper_position()[1] / 10

    def set_gripper_position(self, position: int):
        self._api.set_gripper_position(position * 10, auto_enable=True, wait=False)

    def relax_arm(self):
        self._api.reset(wait=True)
        self._api.set_mode(2)
        self._api.set_state(0)

    def _reset_and_go_home(self):
        self._api.reset(wait=True)
        if not self._home:
            return
        self._api.set_collision_sensitivity(0)  # Disable collision detection
        self._api.set_mode(0)
        self._api.set_state(0)
        retries = 0
        sleep = 0.1
        while True:
            self._api.reset(wait=True)
            try:
                x, y, z, roll, pitch, yaw = self._home
                self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)

                while np.sqrt(np.sum((np.array(self._home[:3]) - np.array(self.position[:3])) ** 2)) > 0.05:
                    time.sleep(1)

                self._api.set_collision_sensitivity(self._collision_sensitivity)
                self._api.reset(wait=True)
                return
            except XArmError:
                if retries >= self._max_reset_retries:
                    raise
                if retries % 5 == 0:
                    sleep *= 2
                logging.error("Raised error during reset")
                traceback.print_exc()
                time.sleep(min(random.expovariate(1.0 / sleep), 0.5))
                retries += 1

    def go_to(self, x, y, z, roll, pitch, yaw, max_retries = 3):
        self._api.set_mode(0)
        self._api.set_state(0)
        retries = 0
        sleep = 0.1
        while True:
            self._api.reset(wait=True)
            try:
                self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)

                while np.sqrt(np.sum((np.array([x,y,z]) - np.array(self.position[:3])) ** 2)) > 0.05:
                    time.sleep(1)
                self._api.reset(wait=True)
                return
            except XArmError:
                if retries >= max_retries:
                    raise
                if retries % 5 == 0:
                    sleep *= 2
                logging.error("Raised error during reset")
                traceback.print_exc()
                time.sleep(min(random.expovariate(1.0 / sleep), 0.5))
                retries += 1



class XArmPositionController(XArm, ArmPositionController):
    def __init__(self, *args, speed: int = 50, **kwargs):
       self._speed = speed
       super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._api.set_mode(7)   # Enable online trajectory planning
        self._api.set_state(0)  # Prepare to move

    def set_position(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=self._speed, is_radian=False, wait=False)


class RestrictedXArmPositionController(RestrictedPositionMixin, XArmPositionController):
    pass


class XArmVelocityController(XArm, ArmVelocityController):
    def __init__(self, *args, max_speed: float = 100.0, max_acc: float = 1000.0, **kwargs):
        self._max_speed = max_speed
        super().__init__(*args, **kwargs)
        self._api.set_tcp_maxacc(max_acc)
        self._api.set_tcp_jerk(10000.)

    def reset(self):
        super().reset()
        self._api.set_mode(5)   # Enable velocity control
        self._api.set_state(0)  # Prepare to move

    def set_velocity(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        velocity = np.array([x, y, z])
        speed = np.linalg.norm(velocity)
        if speed > self._max_speed:
            x, y, z = velocity * self._max_speed / speed
        self._api.vc_set_cartesian_velocity([x, y, z, roll, pitch, yaw], is_radian=False)
