from typing import Tuple, Dict, Any
import numpy as np

from ruka.observation import Observe
from ruka.util.debug import smart_shape

class RobotLiftRewardNoClosedGripper:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, terminal_reward, lift_dist, time_penalty, max_length):
        self._terminal_reward = terminal_reward
        self.lift_dist = lift_dist
        self._time_penalty = time_penalty
        self._max_length = max_length

        # Placeholders
        self._lifting = False
        self._start_height = None
        #self._old_robot_height = None
        self._step = 0

    def __call__(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> Tuple:
        """ Calculate reward and succes based on observation 
            Args:
                obs: dikt like observation
                info: info dict
            Return:
                (reward: float, is_succes: bool)
        """

        assert Observe.GRIPPER_OPEN.value in obs, smart_shape(obs)
        assert Observe.GRIPPER.value in obs, smart_shape(obs)
        assert Observe.ROBOT_POS.value in obs, smart_shape(obs)

        gripper_open = obs[Observe.GRIPPER_OPEN.value][0]
        gripper_pos = obs[Observe.GRIPPER.value][0]
        robot_height = obs[Observe.ROBOT_POS.value][2]

        reward = 0.

        if not gripper_open and gripper_pos > 5:
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True
            if robot_height - self._start_height > self.lift_dist:
                # Object was lifted by the desired amount
                return self._terminal_reward, True
        else:
            self._lifting = False

        reward += self._time_penalty

        if 'broken' in info and info['broken']:
           reward -= self._max_length - self._step

        self._step += 1
        #self._old_robot_height = robot_height
        return reward, False

    def reset(self):
        #_, _, self._old_robot_height = self._env.robot.pos
        self._lifting = False
        self._step = 0
