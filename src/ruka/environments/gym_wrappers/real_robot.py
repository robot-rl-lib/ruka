import gym
import numpy as np
from ruka.environments.common.env_util import get_supported_robot_env
from ruka.robot.robot import RobotError

class SafetyWrapper(gym.Wrapper):

    def __init__(self, env, min_x, max_x, min_y, max_y, min_z, max_z, no_rot = False):
        self.env = env
        self._robot_env = get_supported_robot_env(env,'_robot')._robot

        self.no_rot = no_rot

        self._min_x = min_x
        self._max_x = max_x

        self._min_y = min_y
        self._max_y = max_y

        self._min_z = min_z
        self._max_z = max_z



    def step(self, action):
        expand_dim = False
        if len(action.shape) == 2:
            assert action.shape[0] == 1, action.shape
            action = action[0]
            expand_dim = True

        x, y, z, roll, pitch, yaw = self._robot_env.arm.position

        dx_min = np.abs(self._min_x - x)
        dx_max = np.abs(self._max_x - x)

        dy_min = np.abs(self._min_y - y)
        dy_max = np.abs(self._max_y - y)

        dz_min = np.abs(self._min_z - z)
        dz_max = np.abs(self._max_z - z)


        if dx_min < 20:
            action[1] = action[1] * (1 / (21 - dx_min))
        if x < self._min_x:
            action[1] = -0.02 * dx_min

        if dx_max < 20:
            action[1] = action[1] * (1 / (21 - dx_max))
        if x > self._max_x:
            action[1] = 0.02 * dx_max

        if dy_min < 20:
            action[0] = action[0] * (1 / (21 - dy_min))
        if y < self._min_y:
            action[0] = -0.02 * dy_min

        if dy_max < 20:
            action[0] = action[0] * (1 / (21 - dy_max))
        if y > self._max_y:
            action[0] = 0.02 * dy_max

        #if dz_min < 20:
        #    action[2] = action[2] * (1 / (21 - dz_min))
        #if z < self._min_z:
        #    action[2] = -0.02 * dz_min



        if self.no_rot:
            action[3] = 0

        if expand_dim:
            action = action[None]

        return self.env.step(action)


class RandomReplaceOnSuccess(gym.Wrapper):

    def __init__(self, env, min_x, max_x, min_y, max_y, min_z, max_z, max_yaw=None, min_yaw=None):
        self.env = env
        self._robot_env = get_supported_robot_env(env,'_robot')._robot

        self._min_x = min_x
        self._max_x = max_x

        self._min_y = min_y
        self._max_y = max_y

        self._max_yaw = max_yaw
        self._min_yaw = min_yaw

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if done and info['is_success']:
            try:
                self._robot_env.go_to_random(
                    self._min_x, self._max_x,
                    self._min_y, self._max_y,
                    self._max_yaw, self._min_yaw
                )
            except RobotError as e:
                print('Robot error while random replace. No need to do anything, just reset')
        return obs, r, done, info
