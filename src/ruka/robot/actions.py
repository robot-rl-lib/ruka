import enum
import gym
import numpy as np

from typing import Tuple, Union

from ruka.environments.common.controller import Controller
from ruka.robot.robot import ArmInfo, ArmPosControlled, ArmVelControlled, \
    GripperPosControlled, Robot, ControlMode, RobotRecoverableError
from ruka.util.x3d import Vec3, chain, compose_matrix_tool, \
    compose_matrix_world, decompose_matrix_world
from ruka.robot.collision import CollisionDetector


# -------------------------------------------------------------------- Robot --


class RobotAction(enum.Enum):
    NOP = 0
    STEADY = 1
    HOLD = 2
    RELAX = 3
    PARK = 4
    GO_HOME = 5


class RobotController(Controller):
    def __init__(self, robot: Robot):
        self._robot = robot

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(6)

    def reset(self):
        #self._robot.go_home()
        self._robot.steady(ControlMode.POS)

    def act(self, action):
        """
        Has wait=True semantics.
        """
        if action == RobotAction.STEADY.value:
            self._robot.steady()
        if action == RobotAction.HOLD.value:
            self._robot.hold()
        if action == RobotAction.RELAX.value:
            self._robot.relax()
        if action == RobotAction.PARK.value:
            self._robot.park()
        if action == RobotAction.GO_HOME.value:
            self._robot.go_home()


# ---------------------------------------------------------------- Cartesian --


class CartesianController(Controller):
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
        raise NotImplementedError()


class VelWorld(CartesianController):
    """
    A controller, whose input is world-space velocity.

    act() has semantics of "wait=False" and returns immediately.
    """


class PosWorld(CartesianController):
    """
    A controller, whose input is world-space position.

    act() has semantics of "wait=False" and returns immediately.
    """


class PosWorldWait(CartesianController):
    """
    A controller, whose input is world-space position.

    act() has semantics of "wait=True" and returns only after the move is
    completed.
    """


# ------------------------------------------------------- Cartesian wrappers --


class VelWorldOverRobot(VelWorld, CartesianController):
    def __init__(self, arm: ArmVelControlled):
        self._arm = arm

    def reset(self):
        # Reset is performed through Robot, not through Arm, because reset is a
        # system-level procedure. You cannot reset arm and gripper separately,
        # they work together when performing the reset.
        pass

    def act(self, action):
        self._arm.set_vel(action['xyz'], action['rpy'])

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._arm.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._arm.angle_limits


class CollisionAwareVelWorldOverRobot(VelWorldOverRobot):
    def __init__(self, arm: ArmVelControlled, detector: CollisionDetector):
        super().__init__(arm)
        self._detector = detector

    def act(self, action):
        action['xyz'] = self._detector.test_vel(action['xyz'])
        super().act(action)


class RestrictedYawVelWorld(VelWorld):
    '''
    Restrict TCP yaw angle
    '''

    def __init__(self, vel_c: VelWorld, robot_c: RobotController, arm_info: ArmInfo, restrictions: Tuple[float, float]):
        '''
        :param vel_c: Any VelWorld controller
        :param robot_c: Any RobotController supporting RobotAction.HOLD
        :param arm_info: ArmInfo object reporting angles property
        :param restrictions: Tuple of min and max yaw angles, 0 matches home yaw angle of TCP (90 degrees by convention)
        '''

        self._vel_c = vel_c
        self._robot_c = robot_c
        self._arm_info = arm_info
        self._restrictions = restrictions

    def reset(self):
        self._vel_c.reset()

    def act(self, action):
        l, h = self._restrictions
        yaw = self._arm_info.angles[2]
        if yaw < -90:
            yaw += 360
        yaw -= 90
        if yaw > h or yaw < l:
            self._robot_c.act(RobotAction.HOLD.value)
            raise RobotRecoverableError('Yaw restriction violated! Limits are (%f, %f) and yaw is %f' % (l, h, yaw))
        self._vel_c.act(action)

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._vel_c.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._vel_c.angle_limits


class VelToolOverWorld(CartesianController):
    """
    A contoller, whose input is tool-space velocity, implemented over the
    controller, whose input is world-space velocity.

    act() has semantics of "wait=False" and returns immediately.
    """
    def __init__(self, c: VelWorld, arm_info: ArmInfo):
        self._c = c
        self._arm_info = arm_info

    def reset(self):
        self._c.reset()

    def act(self, action):
        xyz = action['xyz']
        rpy = action['rpy']

        # # Normalize angles to be < 180 deg.
        # rpy_const = np.max(1, *np.abs(rpy))
        # rpy /= rpy_const

        # Compute world-space velocity.
        world_to_tool = compose_matrix_world(angles=self._arm_info.angles)
        tool_velocity = compose_matrix_tool(pos=xyz, angles=rpy)
        tool_to_world = np.linalg.inv(world_to_tool)
        world_velocity = chain(world_to_tool, tool_velocity, tool_to_world)

        # Act.
        xyz, rpy = decompose_matrix_world(world_velocity)
        self._c.act({'xyz': xyz, 'rpy': rpy })

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.angle_limits


class PosWorldOverArm(PosWorld, CartesianController):
    def __init__(self, arm: ArmPosControlled):
        self._arm = arm

    def reset(self):
        # Reset is performed through Robot, not through Arm, because reset is a
        # system-level procedure. You cannot reset arm and gripper separately,
        # they work together when performing the reset.
        pass

    def act(self, action):
        self._arm.set_pos(action['xyz'], action['rpy'])

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._arm.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._arm.angle_limits


class PosRelToolOverAbsWorld(CartesianController):
    """
    A controller, whose input is delta to position, specified in tool space,
    implemented over the controller, whose input is absolute position,
    specified in world-space.

    act() can have the "wait=False" or "wait=True" semantics, depending on the
    underlying controller
    """

    def __init__(self, c: Union[PosWorld, PosWorldWait], arm_info: ArmInfo):
        self._c = c
        self._arm_info = arm_info

    def reset(self):
        self._c.reset()

    def act(self, action):
        world_to_tool = compose_matrix_world(
            pos=self._arm_info.pos, angles=self._arm_info.angles)
        tool_to_target = compose_matrix_tool(
            pos=action['xyz'], angles=action['rpy'])
        world_to_target = chain(world_to_tool, tool_to_target)

        pos, angles = decompose_matrix_world(world_to_target)
        self._c.act({'xyz': pos, 'rpy': angles})

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.angle_limits


class PosRelWorldOverAbsWorld(CartesianController):
    """
    A controller, whose input is delta to position, specified in world space,
    implemented over the controller, whose input is absolute position,
    specified in world space.
    """

    def __init__(self, c: Union[PosWorld, PosWorldWait], arm_info: ArmInfo):
        self._c = c
        self._arm_info = arm_info

    def reset(self):
        self._c.reset()

    def act(self, action):
        world_to_tool = compose_matrix_world(
            pos=self._arm_info.pos, angles=self._arm_info.angles)
        tool_to_target = compose_matrix_world(
            pos=action['xyz'], angles=action['rpy'])
        world_to_target = chain(world_to_tool, tool_to_target)

        pos, angles = decompose_matrix_world(world_to_target)
        self._c.act({'xyz': pos, 'rpy': angles})

    @property
    def pos_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.pos_limits

    @property
    def angle_limits(self) -> Tuple[Vec3, Vec3]:
        return self._c.angle_limits


# ------------------------------------------------------------------ Gripper --


class GripperController(Controller):
    @property
    def action_space(self) -> gym.spaces.Space:
        low, high = self.limits
        return gym.spaces.Box(low, high)

    @property
    def gripper_limits(self) -> Tuple[float, float]:
        """
        Return (low, high).
        """
        raise NotImplementedError()


class GripperPos(GripperController):
    """
    A controller, whose input is gripper target position.

    act() has semantics of "wait=False" and returns immediately.
    """


class GripperPosWorldWait(GripperController):
    """
    A controller, whose input is gripper target position.

    act() has semantics of "wait=True" and returns only after the move is
    completed.
    """


# --------------------------------------------------------- Gripper wrappers --


class GripperPosOverRobot(GripperController):
    def __init__(self, gripper: GripperPosControlled):
        self._gripper = gripper

    def reset(self):
        # Reset is performed through Robot, not through Gripper, because reset
        # is a system-level procedure. You cannot reset arm and gripper
        # separately, they work together when performing the reset.
        pass

    def act(self, action):
        self._gripper.set_gripper_pos(action)

    @property
    def gripper_limits(self) -> Tuple[float, float]:
        return self._gripper.gripper_limits


# -------------------------------------------------------------- Whole robot --


class GripperAndMove(Controller):
    def __init__(
            self,
            robot_c: RobotController,
            move_c: CartesianController,
            gripper_c: GripperController,
        ):
        self._robot_c = robot_c
        self._move_c = move_c
        self._gripper_c = gripper_c
        self._last_gripper_action = None

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict({
            'robot': self._robot_c.action_space,
            'move': self._move_c.action_space,
            'gripper': self._move_c.action_space
        })

    def reset(self):
        self._robot_c.reset()
        self._move_c.reset()
        self._gripper_c.reset()
        self._last_gripper_action = None

    def act(self, action):
        if action['robot'] != RobotAction.NOP.value:
            self._robot_c.act(action['robot'])

        if action['gripper'] != self._last_gripper_action:
            self._gripper_c.act(action['gripper'])
            self._last_gripper_action = action['gripper']

        self._move_c.act(action['move'])


"""
c = GripperAndMove(robot_c, move_c, gripper_c)
c = ClippedNormOverVerbatim(c, {'move': {'xyz': True, 'rpy': True}}) # <--- limits or default or per-item
c = SplitVec3(c, {'move': {'xyz': ['x', 'y', 'z'], 'rpy': ['roll', 'pitch', 'yaw']}})
c = Defaults(c, {'robot': {'action': RobotAction.NOP.value}, 'move': {'pitch': 0, 'yaw': 0}})
c = Restrict(c, {'move': {'x': (-100, 100), 'y': (-200, 200), 'z': (-300, 300)}})
c = Scale(c, {'move': {'x': True, 'y': True, 'z': (-2, 2)}}, (-1, 1)) # <-- all boxes or selected
c = ToContinuous(c, {'gripper': True}, [-1, 1]) # closest to which key?
"""
