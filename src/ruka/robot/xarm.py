import inspect
import logging
from typing import Dict, List, Tuple
import numpy as np
import multiprocessing as mp
import os
import random
import select
import time
import traceback
import math
import itertools
from functools import wraps
from threading import Event

from dataclasses import dataclass
from xarm.wrapper import XArmAPI
from xarm.x3.code import APIState

from ruka.robot.force import ForceInfo
from ruka.robot.realtime import WatchHoundPiped, WatchdogParams, WatchHoundOSPiped
from ruka.util.reporter import store_live_robot_params
from ruka.util.x3d import Vec3, conventional_rotation, tool_to_world
from ruka_os.globals import WATCHDOG_AGGRESSIVE_REALTIME, USE_VEL_WATCHDOG


from .robot import \
    RobotError, RobotRecoverableError, RobotUnrecoverableError, ArmInfo, ArmPosControlled, ArmVelControlled, ArmPosVelControlled, \
    GripperInfo, GripperPosControlled, ControlMode

logging.basicConfig()

FALLBACK_TO_SLEEP_IN_API_CALLS = False
FALLBACK_TO_NORMAL_SPEED_FOR_INTENSE = False

def raise_for_not_finite(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        def _raise_for_not_finite(arg):
            if isinstance(arg, str) or isinstance(arg, bytes):
                return
            if isinstance(arg, float) and not math.isfinite(arg):
                raise ValueError("Infinite or NaN are not valid arguments")
            if np.isscalar(arg) and not np.isfinite(arg):
                raise ValueError("Infinite or NaN are not valid arguments")
        for arg in itertools.chain(args, kwargs.values()):
            _raise_for_not_finite(arg)
            try:
                for inner in arg:
                    _raise_for_not_finite(inner)
            except TypeError:
                continue
        return f(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------- API --


@dataclass
class XArmConfig:
    ip: str

    max_speed: float          # L2
    max_angular_speed: float  # Linf
    max_acc: float
    max_jerk: float
    gripper_max_speed: float
    max_joint_speed: float
    joints_limits: Dict[int, Tuple[float, float]]

    home_pos: Vec3
    home_angles: Vec3
    collision_sensitivity: int
    max_reset_retries: int
    max_intense_speed: float  # L2
                              # max speed for the trajectories we are sure enough
                              # will not cause problems (e.g. go_home)
    report_info: bool = True


class _XArm(ArmInfo, ForceInfo, GripperInfo, GripperPosControlled):
    def __init__(self, config: XArmConfig):
        self._config = config
        self._joints_hard_limits = {
            1: (-360, 360),                                                       # Joint 1
            2: (-118, 120),                                                       # Joint 2
            3: (-360, 360),                                                       # Joint 3
            4: (-11, 225),                                                        # Joint 4
            5: (-360, 360),                                                       # Joint 5
            6: (-97, 180),                                                        # Joint 6
            7: (-360, 360),                                                       # Joint 7
        }
        self._joints_soft_limits = {
            i: (
                max(hard_limit[0], soft_limit[0]),
                min(hard_limit[1], soft_limit[1])
            )
            for i, hard_limit in self._joints_hard_limits.items()
            if (soft_limit := self._config.joints_limits.get(i, hard_limit))
        }
        self._reduced_limits_list = []
        for i in sorted(self._joints_soft_limits.keys()):
            self._reduced_limits_list.extend(self._joints_soft_limits[i])
        self._api = _with_xarm_error_handling(XArmAPI(config.ip))            # Handle errors
        self._api.clean_error()                                              # Clean all previous errors
        self._api.clean_warn()                                               # Clean all previous warnings
        self._api.clean_gripper_error()                                      # Clean all previous gripper errors
        self._api.set_allow_approx_motion(True)                              # Allow to avoid overspeed near some singularities using approximate solutions
        self._api.set_collision_rebound(False)                               # Disable rebound after collision
        self._api.set_collision_sensitivity(config.collision_sensitivity)    # Set collision sensitivity
        self._api.set_self_collision_detection(True)                         # Enable self collision detection
        self._api.set_collision_tool_model(1)                                # Set XArm Gripper collision tool model
        self._api.set_safe_level(4)                                          # Default safe level
        self._api.set_fence_mode(False)                                      # Disable xarm safety boundaries
        self._api.set_reduced_max_joint_speed(self._config.max_joint_speed)  # Reduce joints angular speed
        self._api.set_reduced_joint_range(self._reduced_limits_list)         # Reduce joints angles
        self._api.set_reduced_mode(True)                                     # Enable joints restrictions
        self._api.set_gripper_speed(config.gripper_max_speed * 10)           # Set current gripper speed
        self._api.set_tcp_maxacc(config.max_acc)
        self._api.set_tcp_jerk(config.max_jerk)
        self._api.ft_sensor_enable(True)
        self._api.set_tcp_offset([0, 0, 52.5, 0, 0, 0])
        self._accept_move_commands = False

        if self._config.report_info:
            self._api.register_report_callback(self.report_info, report_joints=True)

        self._mode_change_event = Event()
        self._api.register_mode_changed_callback(self.report_mode_change)
        self._api.register_state_changed_callback(self.report_state_change)

        self._max_intense_speed = self._config.max_speed if FALLBACK_TO_NORMAL_SPEED_FOR_INTENSE else self._config.max_intense_speed

    # - Robot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def steady(self, control_mode=False):
        self.hold()
        self._api.reset()
        self._accept_move_commands = True

    def hold(self):
        if FALLBACK_TO_SLEEP_IN_API_CALLS:
            self._api.reset()
            time.sleep(1)
        else:
            self._api.reset(wait=True)
            self._accept_move_commands = False

    def relax(self):
        self._api.reset()
        self._api_set_mode(2)
        self._api_set_state(0)
        self._accept_move_commands = False

    def park(self):
        self._api.reset()
        self._api_set_state(4)
        self._api.motion_enable(False)
        self._accept_move_commands = False

    def go_home(self):
        retries = 0
        sleep = 0.1
        self._api.set_reduced_mode(False)  # Disable joint limits
        while True:
            self._api.reset(wait=True)
            self._api.set_collision_sensitivity(0)  # Disable collision detection
            self._api_set_mode(0)
            self._api_set_state(0)
            try:
                x, y, z = self._config.home_pos
                roll, pitch, yaw = self._config.home_angles
                ang = ruka_to_xarm([roll,pitch,yaw])
                roll = ang[0]
                pitch = ang[1]
                yaw = ang[2]
                self._api.set_gripper_position(800, auto_enable=True, wait=False) # open gripper for safety
                self._api.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=self._max_intense_speed, wait=True)
                self._api.set_collision_sensitivity(self._config.collision_sensitivity)
                break
            except XArmControllerRecoverableError:
                if retries >= self._config.max_reset_retries:
                    raise
                if retries % 5 == 0:
                    sleep *= 2
                logging.error("Raised error during reset")
                traceback.print_exc()
                time.sleep(min(random.expovariate(1.0 / sleep), 0.5))
                retries += 1

        self._api.set_reduced_mode(True)  # Enable reduced mode back
        self._api.ft_sensor_set_zero()
        self.hold()

    def check(self):
        if self._api.has_error:
            _raise_for_code(self._api, APIState.HAS_ERROR)

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

    # - ForceInfo  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def external_force(self) -> Vec3:
        tool_force = np.array(self._api.ft_ext_force[:3]) * np.array([1, -1, 1])
        world_force, _ = tool_to_world(tool_force, [0] * 3, self.angles)
        return world_force

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
        # in URDF 1 - closed, 0 - open
        # in gripper_pos - 80 - open, 0 - closed
        joints['drive_joint'] = (80 - self.gripper_pos) / 80
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


    def _api_set_mode(self, new_mode: int, force: bool = False):
        """
        Change XArm mode and wait it to be changed.

        Has "wait=True" semantics: waits until the mode is changed.
        """
        # already in request mode
        if self._api.mode == new_mode and not force:
            #print('MODE switch not needed at ', datetime.datetime.now())
            return True

        # use "old approach"
        # TODO: remove when new tech is working fine
        if FALLBACK_TO_SLEEP_IN_API_CALLS:
            self._api.set_mode(new_mode)
            time.sleep(.5)
            return True

        self._mode_change_event.clear()

        #print('MODE switch requested at ', datetime.datetime.now())

        self._api.set_mode(new_mode)
        is_set = self._mode_change_event.wait(timeout=.5)
        if self._api.mode != new_mode:
            #print('MODE switch failed ', datetime.datetime.now())
            raise XArmControllerRecoverableError(error_code, "Mode switch failed")

        #print('MODE switch done at ', datetime.datetime.now())
        return True

    def _api_set_state(self, new_state: int, force: bool = True):
        """
        Change XArm state no need to wait after
        """
        self._api.set_state(new_state)
        if FALLBACK_TO_SLEEP_IN_API_CALLS:
            time.sleep(.5)

    def report_mode_change(self, data):
        if FALLBACK_TO_SLEEP_IN_API_CALLS:
            return
        self._mode_change_event.set()
        #print('MCHANGED:', data, ' at ', datetime.datetime.now())

    def report_state_change(self, data):
        if FALLBACK_TO_SLEEP_IN_API_CALLS:
            return
        #print('SCHANGED:', data, ' at ', datetime.datetime.now())


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
            raise XArmControllerUnrecoverableError(APIState.HAS_ERROR, "Cannot control non Pos in PosControlled")
        _XArm.steady(self, control_mode)
        self._api_set_mode(0)
        self._api_set_state(0)

    @raise_for_not_finite
    def set_pos(self, pos: Vec3, angles: Vec3):
        self.check()
        if not self._accept_move_commands:
            return

        self._target_pos = pos
        self._target_angles = angles

        angles = ruka_to_xarm(angles)
        self._api.set_position(
            x=pos[0], y=pos[1], z=pos[2],
            roll=angles[0], pitch=angles[1], yaw=angles[2],
            speed=self._config.max_speed, is_radian=False, wait=False
        )

    def is_target_reached(self, pos_tolerance=3, angles_tolerance=3):
        pos_diff = np.array(self._target_pos) - np.array(self.pos)
        angle_diff = (np.array(self._target_angles).squeeze() - np.array(self.angles))

        angle_diff = angle_diff % 360
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)

        return (np.sqrt(np.sum(pos_diff ** 2)) < pos_tolerance) and (np.sqrt(np.sum(angle_diff ** 2)) < angles_tolerance)


class XArmVelOverPosControlled(ArmInfo, GripperInfo, GripperPosControlled, ArmPosVelControlled):
    def __init__(self, config: XArmConfig, dt: float = 0.04):
        self._conn, child_conn = mp.Pipe()
        self._control_loop_process = mp.Process(
            target=XArmVelOverPosControlled._control_loop,
            args=(config, dt, child_conn),
            daemon=True
        )
        self._control_loop_process.start()


    @raise_for_not_finite
    def set_pos(self, pos: Vec3, angles: Vec3):
        self._rpc('set_pos', pos, angles)

    @raise_for_not_finite
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

    def go_home(self):
        self._rpc('go_home')

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
            def steady(self, control_mode=False):
                self._api_set_mode(7)
                self._api_set_state(0)
                self._accept_move_commands = True

            def set_pos(self, pos: Vec3, angles: Vec3, speed: float):
                self.check()
                if not self._accept_move_commands:
                    return
                angles = ruka_to_xarm(angles)
                self._api.set_tcp_jerk(self._config.max_jerk)
                self._api.set_position(
                    x=pos[0], y=pos[1], z=pos[2],
                    roll=angles[0], pitch=angles[1], yaw=angles[2],
                    speed=speed, is_radian=False, wait=False, mvacc=self._config.max_acc
                )

            @property
            def accept_move_commands(self):
                return self._accept_move_commands

        if USE_VEL_WATCHDOG:
            watchhound = WatchHoundOSPiped(WatchdogParams(
                dt=dt,
                grace_period=0.001 if WATCHDOG_AGGRESSIVE_REALTIME else 0.01,
                max_fail_time=0.01 if WATCHDOG_AGGRESSIVE_REALTIME else 0.05,
                max_fail_rate=0.5 if WATCHDOG_AGGRESSIVE_REALTIME else 0,
                window_in_steps=10 if WATCHDOG_AGGRESSIVE_REALTIME else 0
            ))
            timer_conn = watchhound.conn
            #connections = [timer_conn, conn]
            fn = conn.fileno()
            fds = [timer_conn, conn.fileno()]

        pos_control = _XArmPosControlledWithSpeed(config)
        vel = np.zeros((3,))
        angular_vel = np.zeros((3,))
        control_mode = None
        error = None
        try:
            while True:

                remote_event = False
                timed_event = False
                if USE_VEL_WATCHDOG:
                    # Doesn't work as the Pipe.send() sometimes freeze for 3 secs
                    #for r in mp.connection.wait(connections):
                    #    if r is conn:
                    #        remote_event = True
                    #   if r is timer_conn:
                    #        x = timer_conn.recv()
                    #        timed_event = True
                    reads, _, _ = select.select(fds, [], [])
                    for cn in reads:
                        if cn==fn:
                            remote_event = True
                        if cn==timer_conn:
                            x = os.read(timer_conn,1)
                            timed_event = True
                else:
                    remote_event = conn.poll()
                    timed_event = True

                if remote_event:
                    try:
                        action, args = conn.recv()
                        if action == 'park':
                            pos_control.park()
                            error = None
                            control_mode = None
                            conn.send(None)
                        elif action == 'hold':
                            pos_control.hold()
                            error = None
                            control_mode = None
                            conn.send(None)
                        elif action == 'relax':
                            pos_control.relax()
                            error = None
                            control_mode = None
                            conn.send(None)
                        elif action == 'go_home':
                            pos_control.go_home()
                            error = None
                            control_mode = None
                            conn.send(None)
                        elif action == 'steady':
                            control_mode = args[0]
                            if control_mode == ControlMode.POS:
                                vel = np.zeros((3,))
                                angular_vel = np.zeros((3,))
                            pos_control.steady()
                            error = None
                            conn.send(None)
                        elif error:
                            conn.send(error)
                        elif action == 'set_vel':
                            vel = np.array(args[0])
                            angular_vel = np.array(args[1])
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
                        elif action == 'set_pos':
                            pos, angles = args
                            pos_control.set_pos(pos, angles, config.max_speed)
                            conn.send(None)
                        else:
                            conn.send(None)
                    except Exception as e:
                        traceback.print_exc()
                        conn.send(e)

                # this is watch dog event!
                if timed_event:
                    if not error and control_mode == ControlMode.VEL and pos_control.accept_move_commands:
                        speed = np.linalg.norm(vel)
                        angular_speed = np.linalg.norm(angular_vel)
                        speed = max(speed, angular_speed)
                        if speed > config.max_speed:
                            speed = config.max_speed
                        pos = np.array(pos_control.pos)
                        angles = np.array(pos_control.angles)
                        try:
                            pos_control.set_pos(list(pos + vel), list(angles + angular_vel), speed)
                        except RobotError as e:
                            error = e

                if not USE_VEL_WATCHDOG:
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
            raise XArmControllerUnrecoverableError(APIState.HAS_ERROR, "Cannot control non Vel in VelControlled")
        _XArm.steady(self,control_mode)
        self._api_set_mode(5)
        self._api_set_state(0)

    @raise_for_not_finite
    def set_vel(self, vel: Vec3, angular_vel: Vec3):
        self.check()
        if not self._accept_move_commands:
            return

        # Clip velocity.
        vel = np.array(vel, dtype=np.float32)
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


class XArmRecoverableError(RobotRecoverableError):
    pass


class XArmUnrecoverableError(RobotUnrecoverableError):
    pass


class XArmAPIError(XArmUnrecoverableError):
    def __init__(self, code: APIState, message: str = ""):
        super().__init__(f"Unrecoverable API error {code}: {message}")
        self.code = code


class XArmControllerRecoverableError(XArmRecoverableError):
    def __init__(self, code, message: str = ""):
        super().__init__(f"Recoverable controller error {code}: {message}")
        self.code = code


class XArmControllerUnrecoverableError(XArmUnrecoverableError):
    def __init__(self, code, message: str = ""):
        super().__init__(f"Unrecoverable controller error {code}: {message}")
        self.code = code


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
        if error_code == 0:
            return
        if error_code >= 1 and error_code <= 3:
            raise XArmControllerUnrecoverableError(error_code, "Emergency stop occured")
        elif error_code >= 10 and error_code <= 17:
            raise XArmControllerUnrecoverableError(error_code, "Servo motor error")
        elif error_code == 18:
            raise XArmControllerUnrecoverableError(error_code, "Force torque sensor communication error")
        elif error_code == 19 or error_code == 28:
            raise XArmControllerUnrecoverableError(error_code, "End module communication error")
        elif error_code == 21:
            raise XArmControllerUnrecoverableError(error_code, "Kinematic error")
        elif error_code == 22:
            raise XArmControllerRecoverableError(error_code, "Self collision detected")
        elif error_code == 23:
            raise XArmControllerRecoverableError(error_code, "Joint angle exceeds limit")
        elif error_code == 24:
            raise XArmControllerRecoverableError(error_code, "Speed exceeds limit")
        elif error_code == 25:
            raise XArmControllerRecoverableError(error_code, "Planning error")
        elif error_code == 26:
            raise XArmControllerUnrecoverableError(error_code, "Linux RT Error")
        elif error_code == 27:
            raise XArmControllerUnrecoverableError(error_code, "Command Reply Error")
        elif error_code == 31:
            raise XArmControllerRecoverableError(error_code, "Collision caused abnormal current")
        elif error_code == 33:
            raise XArmControllerUnrecoverableError(error_code, "Controller GPIO error")
        elif error_code == 35:
            raise XArmControllerRecoverableError(error_code, "Safety boundary limit")
        elif error_code == 38:
            raise XArmControllerUnrecoverableError(error_code, "Abnormal Joint Angle")
        else:
            raise XArmControllerUnrecoverableError(error_code, "Other error")
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
