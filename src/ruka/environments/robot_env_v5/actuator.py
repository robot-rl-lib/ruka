from manipulation_main.common import transformations
from manipulation_main.common import transform_utils
import numpy as np
import gym
from sklearn.preprocessing import MinMaxScaler
from ruka.data import get_data


class Actuator:
    def __init__(self, robot, config):
        self._robot = robot
        self._config = config

        # Define action and state spaces
        self._max_translation = config["robot"]["max_translation"]
        self._max_yaw_rotation = config["robot"]["max_yaw_rotation"]
        self._max_force = config["robot"]["max_force"]

        # Simulator objects.
        self._model = None
        self._joints = None
        self._left_finger, self._right_finger = None, None

        # Last gripper action
        self._gripper_open = True

        ### Setup action space.

        high = np.r_[[self._max_translation] * 3, self._max_yaw_rotation, 1.]
        self._action_scaler = MinMaxScaler((-1, 1))
        self._action_scaler.fit(np.vstack((-1. * high, high)))
        self.action_space = gym.spaces.Box(-1.,
                                        1., shape=(5,), dtype=np.float32)

        self._action_wait = config['robot']['action_wait']
        self._gripper_wait = config['robot']['gripper_wait']

    def reset(self):
        self.endEffectorAngle = 0.
        start_pos = [0., 0., self._robot._initial_height]
        model_path = get_data(self._config.robot.model_path, True)
        self._model = self._robot.add_model(model_path, start_pos, self._robot._init_ori)
        self._joints = self._model.joints
        self._left_finger = self._model.joints[7]
        self._right_finger = self._model.joints[9]

        self._open_gripper()
        self._gripper_open = True

    def step(self, action):
        # Denormalize action vector
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()

        ### Execute action

        # Parse the action vector
        translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
        # Open/close the gripper
        open_close = action[4]

        if open_close > 0. and not self._gripper_open:
            self._open_gripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self._close_gripper()
            self._gripper_open = False
        # Move the robot
        else:
            return self._relative_pose(translation, yaw_rotation)

    def get_state(self):
        """Return the current opening width scaled to a range of [0, 1]."""
        return (1. / 0.1) * self._get_gripper_width()

    def object_detected(self, tol=0.005):
        """Grasp detection by checking whether the fingers stalled while closing."""
        return self._target_joint_pos == 0.05 and self._get_gripper_width() > tol

    def _relative_pose(self, translation, yaw_rotation):
        pos, orn = self._model.get_pose()
        _, _, yaw = transform_utils.euler_from_quaternion(orn)
        #Calculate transformation matrices
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        self._absolute_pose(target_pos, self.endEffectorAngle)

    def _absolute_pose(self, target_pos, target_orn):
        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._robot._initial_height)

        yaw = target_orn
        comp_pos = np.r_[target_pos, yaw]

        for i, joint in enumerate([0, 1, 2, 3]):
            self._joints[joint].set_position(comp_pos[i])

        self._robot.run(self._action_wait)

    def _get_gripper_width(self):
        """Query the current opening width of the gripper."""
        left_finger_pos = 0.05 - self._left_finger.get_position()
        right_finger_pos = 0.05 - self._right_finger.get_position()

        return left_finger_pos + right_finger_pos

    def _close_gripper(self):
        self._target_joint_pos = 0.05
        self._left_finger.set_position(self._target_joint_pos, max_force=self._max_force)
        self._right_finger.set_position(self._target_joint_pos, max_force=self._max_force)

        self._robot.run(self._gripper_wait)

    def _open_gripper(self):
        self._target_joint_pos = 0.0
        self._left_finger.set_position(self._target_joint_pos, max_force=self._max_force)
        self._right_finger.set_position(self._target_joint_pos, max_force=self._max_force)

        self._robot.run(self._gripper_wait)

    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw
