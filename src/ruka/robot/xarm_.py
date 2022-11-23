import inspect
import logging
from typing import Tuple
import numpy as np
import multiprocessing as mp
import random
import time
import traceback

from dataclasses import dataclass
from xarm.wrapper import XArmAPI
from xarm.x3.code import APIState

from ruka.util.reporter import store_live_robot_params
from ruka.util.x3d import Vec3, chain, conventional_rotation


from .robot import \
    ArmError, ArmInfo, ArmPosControlled, ArmVelControlled, ArmPosVelControlled, \
    GripperInfo, GripperPosControlled, ControlMode

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

    report_info: bool = False


class _XArm(ArmInfo, GripperInfo, GripperPosControlled):
    def __init__(self, config: XArmConfig):
        self._config = config
        self._api = _with_xarm_error_handling(XArmAPI(config.ip))         # Handle errors
                        

        self._api.clean_error()                                           # Clean all previous errors
        self._api.clean_warn()                                            # Clean all previous warnings
        self._api.clean_gripper_error()                                   # Clean all previous gripper errors
        self._api.set_allow_approx_motion(True)                           # Allow to avoid overspeed near some singularities using approximate solutions
        self._api.set_collision_rebound(False)                            # Disable rebound after collision
        self._api.set_collision_sensitivity(config.collision_sensitivity)  # Set collision sensitivity
        self._api.set_self_collision_detection(True)                      # Enable self collision detection
        self._api.set_collision_tool_model(1)                             # Set XArm Gripper collision tool model
        self._api.set_safe_level(4)                                       # Default safe level
        self._api.set_fence_mode(False)                                   # Disable xarm safety boundaries
        self._api.set_reduced_mode(False)                                 # Disabled joints restrictions
        self._api.set_gripper_speed(config.gripper_max_speed * 10)        # Set current gripper speed

        self._api.set_tcp_maxacc(config.max_acc)
        self._api.set_tcp_jerk(config.max_jerk)
        self._accept_move_commands = False

        if self._config.report_info:
            self._api.register_report_callback(self.report_info, report_joints=True)

    # - Robot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def steady(self, control_mode=False):
        self.hold()
        self._api.reset()
        self._accept_move_commands = True

    def hold(self):
        self._api.reset()
        time.sleep(1)

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
        retries = 0
        sleep = 0.1
        while True:
            self._api.reset(wait=True)
            self._api.set_collision_sensitivity(0)  # Disable collision detection
            self._api.set_mode(0)
            self._api.set_state(0)
            time.sleep(.5) # Arm needs to sleep for a sec
            try:
                x, y, z = self._config.home_pos
                roll, pitch, yaw = self._config.home_angles
                ang = ruka_to_xarm([roll,pitch,yaw])
                roll = ang[0]
                pitch = ang[1]
                yaw = ang[2]
                self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)
                self._api.set_collision_sensitivity(self._config.collision_sensitivity)
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
    def angles(self) -> Vec3:
        return xarm_to_ruka(list(self._api.position)[3:])

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        ...

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        ...

    # - GripperInfo - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def gripper_pos(self) -> float:
        return self._api.get_gripper_position()[1] / 10

    @property
    def gripper_pos_limits(self) -> Tuple[float, float]:
        return (0., 79.)

    # - GripperPosControlled  - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_gripper_pos(self, pos: int):
        if not self._accept_move_commands:
            return
        self._api.set_gripper_position(pos * 10, auto_enable=True, wait=False)

    # - Misc - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def report_info(self, data):
        """
        Running through a callback method of the ARM which reports
        new joints positions.
        The joints array should match the names with the URDF model!

        data - array provided by the callback
        """
        if not data.get('joints'):
            return
        ds = {}

        # store joints
        joints = {}
        num = 1
        for x in data['joints']:
            jname = f'joint{num}'
            joints[jname] = np.deg2rad(x)
            num += 1        
        ds['joints'] = joints

        store_live_robot_params('xarm', ds)
        
    def force_report_info(self):
        """
        Fetch joints params from the API by a request
        and call the upper function
        """
        js = self._api.get_joint_states(is_radian=False)
        data = {'joints': js[1][0]}
        self.report_info(data)

# Translate XARM reported angles to RUKA conventional angles
def xarm_to_ruka(angles: Vec3) -> Vec3:
    a = conventional_rotation(angles, 2, 90)
    return [a[0],-a[1],-a[2]]


# Translate RUKA conventional angles to XARM API angles
def ruka_to_xarm(angles: Vec3) -> Vec3:
    a = [angles[0],-angles[1],-angles[2]]
    return conventional_rotation(a, 2, -90)
    
# Translate RUKA conventional angular velocity to XARM API velocity
def ruka_to_xarm_vel(angles_vel: Vec3) -> Vec3:
    a = [angles_vel[0],-angles_vel[1],-angles_vel[2]]
    return a

# -------------------------------------------------------------- Controllers --


class XArmPosControlled(_XArm, ArmPosControlled):
    def steady(self, control_mode=False):
        if control_mode is False:
            control_mode = ControlMode.POS
        if control_mode != ControlMode.POS:
            raise XArmControllerError(APIState.HAS_ERROR, "Cannot control non Pos in PosControlled")
        _XArm.steady(self, control_mode)
        self._api.set_mode(0)
        self._api.set_state(0)  # Prepare to move
        time.sleep(.5) # without sleep not work 

    def set_pos(self, pos: Vec3, angles: Vec3):
        self._target_pos = pos
        self._target_angles = angles

        self.check()
        if not self._accept_move_commands:
            return

        angles = ruka_to_xarm(angles)
        self._target_pos = pos
        self._target_angles = angles
        self._api.set_position(
            x=pos[0], y=pos[1], z=pos[2],
            roll=angles[0], pitch=angles[1], yaw=angles[2],
            speed=self._config.max_speed, is_radian=False, wait=True
        )

    def is_target_reached(self, pos_tolerance=3, angles_tolerance=3):
        pos_diff = np.array(self._target_pos) - np.array(self.pos)
        angle_diff = (np.array(self._target_angles).squeeze() - np.array(self.angles))

        angle_diff = angle_diff % 360
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)

        return (np.sqrt(np.sum(pos_diff ** 2)) < pos_tolerance) and (np.sqrt(np.sum(angle_diff ** 2)) < angles_tolerance)


class XArmVelOverPosControlled(ArmInfo, GripperInfo, GripperPosControlled, ArmVelControlled):
    def __init__(self, config: XArmConfig, dt: float = 0.04):
        self._conn, child_conn = mp.Pipe()
        self._control_loop_process = mp.Process(
            target=XArmVelOverPosControlled._control_loop,
            args=(config, dt, child_conn),
            daemon=True
        )
        self._control_loop_process.start()

    def set_vel(self, vel: Vec3, angular_vel: Vec3):
        self._rpc('set_vel', vel, angular_vel)

    def park(self):
        self._rpc('park')

    def hold(self):
        self._rpc('hold')

    def steady(self, control_mode):
        self._rpc('steady', control_mode)

    def relax(self):
        self._rpc('relax')

    @property
    def pos(self) -> Vec3:
        return self._rpc('pos')

    @property
    def speed(self) -> float:
        return self._rpc('speed')

    @property
    def angles(self) -> Vec3:
        return self._rpc('angles')

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        ...

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        ...

    @property
    def gripper_pos(self) -> float:
        return self._rpc('gripper_pos')

    @property
    def gripper_pos_limits(self) -> Tuple[float, float]:
        return self._rpc('gripper_pos_limits')

    def set_gripper_pos(self, pos: int):
        self._rpc('set_gripper_pos', pos)

    def _rpc(self, action, *args):
        self._conn.send((action, args))
        res = self._conn.recv()
        if isinstance(res, Exception):
            raise res
        return res

    @staticmethod
    def _control_loop(config, dt, conn):
        class _XArmPosControlledWithSpeed(XArmPosControlled):
            def set_pos(self, pos: Vec3, angles: Vec3, speed: float):
                self.check()
                if not self._accept_move_commands:
                    return
                angles = ruka_to_xarm(angles)
                self._api.set_position(
                    x=pos[0], y=pos[1], z=pos[2],
                    roll=angles[0], pitch=angles[1], yaw=angles[2],
                    speed=speed, is_radian=False, wait=False
                )

            @property
            def accept_move_commands(self):
                return self._accept_move_commands


        pos_control = _XArmPosControlledWithSpeed(config)
        vel = np.zeros((3,))
        angular_vel = np.zeros((3,))
        try:
            while True:
                if conn.poll():
                    try:
                        action, args = conn.recv()
                        if action == 'set_vel':
                            vel = np.array(args[0])
                            angular_vel = np.array(args[1])
                            conn.send(None)
                        elif action == 'park':
                            pos_control.park()
                            conn.send(None)
                        elif action == 'hold':
                            pos_control.hold()
                            conn.send(None)
                        elif action == 'steady':
                            control_mode = np.array(args[0])
                            pos_control.steady(control_mode)
                            conn.send(None)
                        elif action == 'relax':
                            pos_control.relax()
                            conn.send(None)
                        elif action == 'go_home':
                            pos_control.go_home()
                            conn.send(None)
                        elif action == 'check':
                            pos_control.check()
                            conn.send(None)
                        elif action == 'pos':
                            conn.send(pos_control.pos)
                        elif action == 'speed':
                            conn.send(pos_control.speed)
                        elif action == 'angles':
                            conn.send(pos_control.angles)
                        elif action == 'gripper_pos':
                            conn.send(pos_control.gripper_pos)
                        elif action == 'gripper_pos_limits':
                            conn.send(pos_control.gripper_pos_limits)
                        elif action == 'set_gripper_pos':
                            pos_control.set_gripper_pos(args[0])
                            conn.send(None)
                        else:
                            conn.send(None)
                    except Exception as e:
                        traceback.print_exc()
                        conn.send(e)
                if pos_control.accept_move_commands:
                    speed = np.linalg.norm(vel)
                    pos = np.array(pos_control.pos)
                    angles = np.array(pos_control.angles)
                    pos_control.set_pos(list(pos + vel), list(angles + angular_vel), speed)
                time.sleep(dt)
        except:
            traceback.print_exc()
        finally:
            pos_control.hold()


class XArmVelControlled(_XArm, ArmVelControlled):
    def steady(self, control_mode):
        if control_mode is False:
            control_mode = ControlMode.VEL
        if control_mode != ControlMode.VEL:
            raise XArmControllerError(APIState.HAS_ERROR, "Cannot control non Vel in VelControlled")
        _XArm.steady(self,control_mode)
        self._api.set_mode(5)   # Enable velocity control
        self._api.set_state(0)  # Prepare to move
        time.sleep(.5) # without sleep not work

    def set_vel(self, vel: Vec3, angular_vel: Vec3):
        self.check()
        if not self._accept_move_commands:
            return

        # Clip velocity.
        vel = np.array(vel)
        speed = np.linalg.norm(vel)
        if speed > self._config.max_speed:
            vel *= self._config.max_speed / speed

        # Clip angular velocity.
        angular_vel = np.array(angular_vel)
        angular_speed = np.linalg.norm(angular_vel, np.inf)
        if angular_speed > self._config.max_angular_speed:
            angular_vel *= self._config.max_angular_speed / angular_speed

        # Set.
        x, y, z = vel
        ang = ruka_to_xarm_vel(angular_vel)
        roll = ang[0]
        pitch = ang[1]
        yaw = ang[2]
        self._api.vc_set_cartesian_velocity(
            [x, y, z, roll, pitch, yaw], is_radian=False)


class XArmPosVelControlled(ArmPosVelControlled, XArmPosControlled, XArmVelControlled):
    def steady(self, control_mode):
        if control_mode == ControlMode.POS:
            return XArmPosControlled.steady(self, control_mode)
        if control_mode == ControlMode.VEL:
            return XArmVelControlled.steady(self, control_mode)
        raise NotImplementedError()


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
                if res is True:
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
