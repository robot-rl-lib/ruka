import inspect
import logging
import time
from typing import Tuple
from . import ArmError, Arm, Gripper

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
    

class XArm(Arm):
    def __init__(self, ip: str, speed: int = 50, collision_senstivity: int = 3):
        self._speed = speed
        self._collision_senstivity = collision_senstivity
        self._api = with_xarm_error_handling(XArmAPI(ip))                # Handle errors
        self._api.clean_error()                                          # Clean all previous errors
        self._api.clean_warn()                                           # Clean all previous warnings
        self._api.set_allow_approx_motion(True)                          # Allow to avoid overspeed near some singularities using approximate solutions
        self._api.set_collision_rebound(False)                           # Disable rebound after collision
        self._api.set_collision_sensitivity(self._collision_senstivity)  # Set collision sensitivity
        self._api.set_self_collision_detection(True)                     # Enable self collision detection
        self._api.set_safe_level(4)                                      # Default safe level
        self._api.set_fence_mode(False)                                  # Disable xarm safety boundaries
        self._api.set_reduced_mode(False)                                # Disabled joints restrictions
        self._api.set_mode(7)                                            # Enable online trajectory planning
        self._api.set_state(0)                                           # Prepare to move

    def reset(self):
        self._api.motion_enable(True)
        self._api.reset(wait=True)

    def disable(self):
        self._api.set_state(4)
        self._api.motion_enable(False)

    @property
    def position(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(self._api.position)

    def move(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=self._speed, wait=False)

    @property
    def speed(self) -> int:
        return self._speed

    @speed.setter
    def speed(self, speed: int):
        self._speed = speed

    @property
    def collision_senstivity(self) -> int:
        return self._collision_senstivity

    @collision_senstivity.setter
    def collision_senstivity(self, collision_senstivity: int):
        self._collision_senstivity = collision_senstivity
        self._api.set_collision_sensitivity(self._collision_senstivity)


class XArmGripper(Gripper):
    def __init__(self, ip: str, speed: int = 50):
        self._speed = speed
        self._api = with_xarm_error_handling(XArmAPI(ip))  # Handle errors
        self._api.clean_gripper_error()                    # Clean all previous errors
        self._api.set_collision_tool_model(1)              # Set XArm Gripper collision tool model
        self._api.set_gripper_speed(self._speed * 10)      # Set current gripper speed

    @property
    def speed(self) -> int:
        return self._speed

    @speed.setter
    def speed(self, speed: int):
        self._speed = speed
        self._api.set_gripper_speed(self._speed * 10)

    @property
    def force(self) -> float:
        raise NotImplementedError

    @force.setter
    def force(self, force: float):
        raise NotImplementedError

    @property
    def position(self) -> int:
        return self._api.get_gripper_position()[1] / 10

    def set_position(self, position: int):
        self._api.set_gripper_position(position * 10, auto_enable=True, wait=False)

