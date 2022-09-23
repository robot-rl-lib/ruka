import pybullet
import torch

from dataclasses import dataclass
from pybullet_utils import bullet_client
from typing import Iterable, Mapping, Optional


from .base import \
    Vec3, Quat, \
    Simulator, Object, ObjectConfig, Link, LinkConfig, LinkProperty, Joint, \
    JointConfig, JointType, JointProperty, ControlMode, Camera, CameraConfig, \
    ImageType


# ---------------------------------------------------------------- Simulator --


class PyBulletSimulator(Simulator):
    def __init__(self, config: 'PyBulletConfig'):
        self._config = config
        self._pybullet = bullet_client.BulletClient(pybullet.DIRECT)
        self._steps_made = 0
        self._objects = set()
        self._cameras = set()

    def reset(self):
        self._check_alive()
        
        # Reset.
        self._clear()
        self._pybullet.resetSimulation()

        # Set parameters.
        self._pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self._config.fixed_time_step,
            numSolverIterations=self._config.num_solver_iterations,
            enableConeFriction=self._config.enable_cone_friction
        )
        self._pybullet.setGravity(*self._config.gravity)

    def step(self):
        self._check_alive()

        # Joint control.
        for o in self._objects:
            for j in o.joints.values():
                j.control()

        # Simulation.
        self._pybullet.stepSimulation()
        self._steps_made += 1

    def run_for(self, num_seconds: int):
        self._check_alive()
        for _ in range(int(num_seconds / self._config.fixed_time_step)):
            self.step()
    
    def add_object(self, 
            config: ObjectConfig,
            position: Vec3, 
            orientation: Quat
        ) -> Object:

        self._check_alive()
        o = PyBulletObject(self, config, position, orientation)
        self._objects.add(o)
        return o

    @property
    def objects(self) -> Iterable[Object]:
        self._check_alive()
        return self._objects

    def add_camera(
            self, 
            config: CameraConfig, 
            position: Vec3, 
            orientation: Quat
        ) -> Camera:
        self._check_alive()
        c = PyBulletCamera(self, config, position, orientation)
        self._cameras.add(c)
        return c

    @property
    def cameras(self) -> Iterable[Camera]:
        self._check_alive()
        return self._cameras

    def close(self):
        self._check_alive()
        self._clear()
        self._pybullet.disconnect()
        self._pybullet = None

    ### Own interface.

    @property
    def p(self) -> bullet_client.BulletClient:
        self._check_alive()
        return self._pybullet

    @property
    def steps_made(self) -> int:
        self._check_alive()
        return self._steps_made

    def remove_object(self, object: 'PyBulletObject'):
        assert object.simulator is self
        self._pybullet.removeBody(object.body_id)
        self._objects.remove(object)

    def remove_camera(self, camera: 'PyBulletCamera'):
        assert camera.simulator is self
        ...
        self._cameras.remove(camera)

    def _clear(self):
        for o in list(self._objects):
            o.remove()
        for c in list(self._cameras):
            c.remove()

    def _check_alive(self):
        if self._pybullet is None:
            raise RuntimeError(f'{self} was destroyed')


@dataclass(frozen=True)
class PyBulletConfig:
    gravity: Vec3
    fixed_time_step: float
    num_solver_iterations: int
    enable_cone_friction: int


# ------------------------------------------------------------------- Object --


class PyBulletObject(Object):
    def __init__(
            self, 
            simulator: PyBulletSimulator, 
            config: ObjectConfig, 
            position: Vec3, 
            orientation: Quat
        ):
        self._simulator = simulator
        self._config = config

        # Load SDF.
        if config.definition_file_path.endswith('.sdf'):
            self._body_id = self._simulator.p.loadSDF(
                config.definition_file_path, globalScaling=config.scale)[0]
            self._simulator.p.resetBasePositionAndOrientation(
                self._body_id, position, orientation)
            if config.static:
                raise NotImplementedError()

        # Load URDF.
        elif config.definition_file_path.endswith('.urdf'):
            self._body_id = self._simulator.p.loadURDF(
                config.definition_file_path, position, orientation, 
                useFixedBase=config.static, globalScaling=config.scale)

        # Unknown format.
        else:
            assert 0

        # Create links and joints.
        self._links = {}
        self._joints = {}

        for i in range(self._simulator.p.getNumJoints(self._body_id)):
            joint_info = self._simulator.p.getJointInfo(self._body_id, i)

            # - Joint type.
            joint_type = joint_info[2]
            if joint_type == pybullet.JOINT_REVOLUTE:
                joint_type = JointType.REVOLUTE
            elif joint_type == pybullet.JOINT_PRISMATIC:
                joint_type = JointType.PRISMATIC
            elif joint_type == pybullet.JOINT_FIXED:
                joint_type = JointType.FIXED
            else:
                raise NotImplementedError(f'Joint type {joint_type} is not implemented')

            # - Joint.
            joint_config = JointConfig(
                name=joint_info[1].decode('utf8'),
                type=joint_type,
                lower_limit=joint_info[8],
                upper_limit=joint_info[9],
                max_effort=joint_info[10]
            )
            self._joints[joint_config.name] = PyBulletJoint(self, i, joint_config)

            # - Link.
            link_config = LinkConfig(name=joint_info[12].decode('utf8'))
            self._links[link_config.name] = PyBulletLink(self, i, link_config)

        # Root link.
        self._root = PyBulletLink(self, -1, LinkConfig(name=NotImplemented))
        self._links[self._root.config.name] = self._root

    @property
    def simulator(self) -> Simulator:
        self._check_alive()
        return self._simulator

    @property
    def config(self) -> ObjectConfig:
        self._check_alive()
        return self._config

    @property
    def root(self) -> Link:
        self._check_alive()
        return self._root

    @property
    def links(self) -> Mapping[str, Link]:
        self._check_alive()
        return self._links

    @property
    def joints(self) -> Mapping[str, Joint]:
        self._check_alive()
        return self._joints

    def remove(self):
        self._check_alive()
        self._simulator.remove_object(self)
        self._simulator = None

    ### Own interface.

    @property
    def body_id(self) -> int:
        return self._body_id

    def _check_alive(self):
        if self._simulator is None:
            raise RuntimeError(f'{self} was destroyed')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  Link -


class PyBulletLink(Link):
    def __init__(self, object: Object, index: int, config: ObjectConfig):
        self._object = object
        self._index = index
        self._config = config

        self._pos_ori_timestamp = -1
        self._position = None
        self._orientation = None

    @property
    def object(self) -> Object:
        return self._object

    @property
    def config(self) -> LinkConfig:
        return self._config
    
    @property
    def position(self) -> Vec3:
        self.poll(LinkProperty.POSITION)
        return self._position

    @property
    def orientation(self) -> Quat:
        self.poll(LinkProperty.ORIENTATION)
        return self._orientation

    def poll(self, *properties: LinkProperty):
        for p in properties:
            # If not old, don't poll.
            if not self._old(p):
                return

            # Poll.
            self._poll(p)
           
    def _old(self, property: LinkProperty):
        if property in [LinkProperty.POSITION, LinkProperty.ORIENTATION]:
            return self.simulator.steps_made > self._pos_ori_timestamp
        else:
            assert 0

    def _poll(self, property: LinkProperty):
         # Pos & ori.
        if property in [LinkProperty.POSITION, LinkProperty.ORIENTATION]:
            # - Update timestamp.
            self._pos_ori_timestamp = self.simulator.steps_made

            # - Root link.
            if self._index == -1:
                self._position, self._orientation = \
                    self.simulator.p.getBasePositionAndOrientation(self._object.body_id)

            # - Normal link.
            else:
                link_state = \
                    self.simulator.p.getLinkState(self._object.body_id, self._index)
                self._position = link_state[0]
                self._orientation = link_state[1]

        # Unknown.
        else:
            assert 0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Joint -


class PyBulletJoint(Joint):
    def __init__(self, object: PyBulletObject, index: int, config: JointConfig):
        self._object = object
        self._index = index
        self._config = config

        self._pos_timestamp = -1
        self._position = None

        self._control_mode = ControlMode.DISABLE
        self._control_target = None
        self._max_effort = config.max_effort

    @property
    def object(self) -> Object:
        return self._object

    @property
    def config(self) -> JointConfig:
        return self._config

    @property
    def position(self) -> Vec3:
        self.poll(JointProperty.POSITION)
        return self._position

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @control_mode.setter
    def control_mode(self, value: ControlMode):
        assert self._config.type != JointType.FIXED
        self._control_mode = value
        self._control_target = None

    @property
    def control_target(self) -> Optional[float]:
        return self._control_target

    @control_target.setter
    def control_target(self, value: Optional[float]):
        assert self._config.type != JointType.FIXED
        assert self._control_mode != ControlMode.DISABLE or value is None
        self._control_target = value

    @property
    def max_effort(self) -> float:
        return self._max_effort

    @max_effort.setter
    def max_effort(self, value: float):
        self._max_effort = value

    def poll(self, *properties: JointProperty):
        for p in properties:
            # If not old, don't poll.
            if not self._old(p):
                return

            # Poll.
            self._poll(p)

    ### Own interface.

    @property
    def index(self) -> int:
        return self._index

    def control(self):
        # Skip fixed joints.
        if self._config.type == JointType.FIXED:
            return

        # Disable if control_target is None.
        control_mode = self._control_mode
        if self._control_target is None:
            control_mode = ControlMode.DISABLE

        # DISABLE.
        if control_mode == ControlMode.DISABLE:
            self.simulator.p.setJointMotorControl2(
                self._object.body_id, 
                self._index, 
                controlMode=pybullet.VELOCITY_CONTROL, 
                force=0.
            )

        # POSITION_PD.
        elif control_mode == ControlMode.POSITION_PD:
            self.simulator.p.setJointMotorControl2(
                self._object.body_id, 
                self._index,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=self._control_target,
                force=self._max_effort
            )

        # Unknown.
        else:
            assert 0

    def _old(self, property: JointProperty):
        if property in [JointProperty.POSITION]:
            return self.simulator.steps_made > self._pos_timestamp
        else:
            assert 0

    def _poll(self, property: JointProperty):
         # Pos.
        if property in [JointProperty.POSITION]:
            # - Update timestamp.
            self._pos_timestamp = self.simulator.steps_made

            # - Poll.
            joint_state = \
                self.simulator.p.getJointState(self._object.body_id, self._index)
            self._position = joint_state[0]

        # Unknown.
        else:
            assert 0


# ------------------------------------------------------------------- Camera --


class PyBulletCamera(Camera):
    def __init__(
            self, 
            simulator: PyBulletSimulator,
            config: CameraConfig,
            position: Vec3,
            orientation: Quat
        ):
        self._simulator = simulator
        self._config = config
        self._position = position
        self._orientation = orientation

        ... # matrix

    @property
    def simulator(self):
        self._check_alive()
        return self._simulator

    @property
    def config(self) -> CameraConfig:
        self._check_alive()
        return self._config

    @property
    def position(self) -> Vec3:
        self._check_alive()
        return self._position

    @position.setter
    def position(self, value: Vec3):
        self._check_alive()
        ...

    @property
    def orientation(self) -> Quat:
        self._check_alive()
        return self._orientation

    @orientation.setter
    def orientation(self, value: Quat):
        self._check_alive()
        ...

    def render(self) -> torch.Tensor:
        self._check_alive()
        ...

    def remove(self):
        self._check_alive()
        self.simulator.remove_camera(self)
        self._simulator = None

    def _check_alive(self):
        if self._simulator is None:
            raise RuntimeError(f'{self} was destroyed')
