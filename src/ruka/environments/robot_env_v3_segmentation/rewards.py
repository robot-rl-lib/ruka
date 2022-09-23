from . import robot

class ShapedCustomTargetReward:

    def __init__(self, config, robot):
        self._robot = robot
        self._shaped = config.shaped

        self._max_delta_z = robot._actuator._max_translation
        self._terminal_reward = config.terminal_reward
        self._grasp_reward = config.grasp_reward
        self._delta_z_scale = config.delta_z_scale
        self._lift_success = config.lift_success
        self._time_penalty = config.time_penalty
        self._terminal_reward_wrong = config.terminal_reward_wrong

        self._running_if_picked_wrong = config.running_if_picked_wrong
        self.lift_dist = None

        # Placeholders
        self._lifting = False
        self._start_height = None
        self._old_robot_height = None
        self._target_model = None

    def __call__(self, obs, action, new_obs):
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        current_highest_id = self._robot.find_highest()

        if self._robot._actuator.object_detected():
            current_highest_id = self._robot.find_highest()
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:

                if ((current_highest_id==self._target_model.model_id) or (not self._robot._use_target_object)):
                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                elif self._running_if_picked_wrong:
                    return self._terminal_reward_wrong, robot.RobotEnv.Status.RUNNING
                else:
                    return self._terminal_reward_wrong, robot.RobotEnv.Status.FAIL

            delta_z = robot_height - self._old_robot_height
            reward = self._grasp_reward + self._delta_z_scale * delta_z

        else:
            self._lifting = False

        # Time penalty
        if self._shaped:
            reward -= self._time_penalty
        else:
            reward -= 0.01

        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING

    def reset(self, target_model):
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]
        self._target_model = target_model
