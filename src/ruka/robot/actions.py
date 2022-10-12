import copy
import gym
import numpy as np

from typing import Tuple

from ruka.environments.common.controller import Controller
from ruka.robot.robot import ArmInfo, ArmVelControlled, GripperInfo
from ruka.util.x3d import Vec3, compose_matrix


# ---------------------------------------------------------------- Cartesian --


class Cartesian(Controller):
    @property
    def action_space(self) -> gym.spaces.Space:
        pos_low, pos_high = self.pos_limits
        angles_low, angles_high = self.angle_limits
        return gym.spaces.Dict({
            'xyz': gym.spaces.Box(pos_low, pos_high),
            'rpy': gym.spaces.Box(angles_low, angles_high)
        })

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


class VelWorld(Cartesian):
    """
    A controller, whose input is world-space velocity.

    act() has semantics of "wait=False" and returns immediately.
    """


class PosWorld(Cartesian):
    """
    A controller, whose input is world-space position.

    act() has semantics of "wait=False" and returns immediately.
    """


class PosWorldWait(Cartesian):
    """
    A controller, whose input is world-space position.

    act() has semantics of "wait=True" and returns only after the move is
    completed.
    """


# ------------------------------------------------------- Cartesian wrappers --


class VelWorldOverArm(Cartesian):
    def __init__(self, arm: ArmVelControlled):
        self._arm = arm

    def act(self, action):
        self._arm.check()
        self._arm.set_vel(action['xyz'], action['rpy'])

    def


class VelToolOverWorld(Cartesian):
    """
    A contoller, whose input is tool-space velocity, implemented over the
    controller, whose input is world-space velocity.

    act() has semantics of "wait=False" and returns immediately.
    """
    def __init__(self, c: VelWorld, tcp: TCPInfo):
        self._c = c
        self._tcp = tcp

    def act(self, action):
        world_to_tool = compose_matrix(angles=self._tcp.rpy)
        action = {'xyz': world_to_tool @ np.array(action['xyz'])}
        yield from self._c.act(action)


class PosWorldWaitOverPosWorld(XYZ):
    """
    A position controller, which waits until the position is reached,
    implemented over the position controller which doesn't.

    If position is not reached, but velocity and acceleration are below
    respective thresholds, the controller assumes that the robot is stuck and
    acts as if the target position is reached.

    act() has semantics of "wait=True" and returns only after the move is
    completed.
    """
    def __init__(
            self,
            c: PosWorld,
            tcp: TCPInfo,
            pos_thr: float,
            vel_thr: float,
            acc_thr: float
        ):
        self._c = c
        self._tcp = tcp
        self._pos_thr = pos_thr
        self._vel_thr = vel_thr
        self._acc_thr = acc_thr

    def act(self, action):
        target = np.array(action['xyz'])
        yield from self._c.act(action)
        yield lambda: (
            np.linalg.norm(np.array(self._tcp.xyz) - target) < self._pos_thr or
            (self._tcp.vel < self._vel_thr and self._tcp.acc < self._acc_thr)
        )


class PosRelToolOverAbsWorld(XYZ):
    """
    A controller, whose input is delta to position, specified in tool space,
    implemented over the controller, whose input is absolute position,
    specified in world-space.

    act() can have the "wait=False" or "wait=True" semantics, depending on the
    underlying controller
    """

    def __init__(self, c: Union[PosWorld, PosWorldWait], tcp: TCPInfo):
        self._c = c
        self._tcp = tcp

    def act(self, action):
        world_to_tool = compose_matrix(
            angles=self._tcp.rpy, translate=self._tcp.position)
        action = {'xyz': self._tcp.xyz + world_to_tool @ np.array(action['xyz'])}
        yield from self._c.act(action)


class PosRelWorldOverAbsWorld(XYZ):
    """
    A controller, whose input is delta to position, specified in world space,
    implemented over the controller, whose input is absolute position,
    specified in world space.
    """

    def __init__(self, c: Union[PosWorld, PosWorldWaiting], tcp: TCPInfo):
        self._c = c
        self._tcp = tcp

    def act(self, action):
        action = {'xyz': self._tcp.xyz + np.array(action['xyz'])}
        yield from self._c.act(action)


# ------------------------------------------------------------------ Gripper --


class Gripper(Controller):
    @property
    def action_space(self) -> gym.spaces.Space:
        low, high = self.limits
        return gym.spaces.Dict({'gripper': gym.spaces.Box(low, high)})

    @property
    def limits(self) -> Tuple[float, float]:
        """
        Return (low, high).
        """
        raise NotImplementedError()


class GripperPos(Gripper):
    """
    A controller, whose input is gripper position.
    """


class GripperPosWaitOverVerbatim(Gripper):
    """
    A position controller, which waits until the position is reached,
    implemented over the position controller which doesn't.

    If position is not reached, but velocity and acceleration are below
    respective thresholds, the controller assumes that the robot is stuck and
    acts as if the target position is reached.
    """

    def __init__(
            self,
            c: GripperPos,
            gripper: GripperInfo,
            pos_thr: float,
            vel_thr: float,
            acc_thr: float
        ):
        self._c = c
        self._gripper = gripper

    def act(self, action):
        target = action['gripper']
        yield from self._c.act(action)
        yield lambda: (
            np.abs(self._gripper.pos - target) < self._pos_thr or
            (
                self._gripper.vel < self._vel_thr and
                self._gripper.acc < self._acc_thr
            )
        )


# -------------------------------------------------------------- Whole robot --


class GripperAndMove(Controller):
    pass


class GripperOrMove(Controller):
    pass