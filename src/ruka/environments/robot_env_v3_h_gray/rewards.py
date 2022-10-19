from . import robot


class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, config, robot):
        self._robot = robot
        self._shaped = config.shaped

        self._max_delta_z = robot._actuator._max_translation
        self._terminal_reward = config.terminal_reward
        self._grasp_reward = config.grasp_reward
        self._delta_z_scale = config.delta_z_scale
        self._lift_success = config.lift_success
        self._time_penalty = config.time_penalty
        self._table_clearing = config.table_clearing
        self.lift_dist = None

        # Placeholders
        self._lifting = False
        self._start_height = None
        self._old_robot_height = None

    def __call__(self, obs, action, new_obs):
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        if self._robot._actuator.object_detected():
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True
            if robot_height - self._start_height > self.lift_dist:
                # Object was lifted by the desired amount
                return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
            if self._shaped:
                # Intermediate rewards for grasping and lifting
                delta_z = robot_height - self._old_robot_height
                reward = self._grasp_reward + self._delta_z_scale * delta_z

        else:
            self._lifting = False

        # Time penalty
        if self._shaped:
            reward -= self._grasp_reward + self._delta_z_scale * self._max_delta_z
        else:
            reward -= 0.01

        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING

    def reset(self):
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]


class ShapedCustomReward(Reward):

    def __call__(self, obs, action, new_obs):
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        if self._robot._actuator.object_detected():
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:
                if self._table_clearing:
                    # Object was lifted by the desired amount
                    grabbed_obj = self._robot.find_highest()
                    if grabbed_obj != -1:
                        self._robot.remove_model(grabbed_obj)
                    
                    # Multiple object grasping
                    # grabbed_objs = self._robot.find_higher(self.lift_dist)
                    # if grabbed_objs:
                    #     self._robot.remove_models(grabbed_objs)

                    self._robot._actuator._open_gripper()
                    if self._robot.get_num_body() == 2: 
                        return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                    return self._lift_success, robot.RobotEnv.Status.RUNNING
                else:
                    if not self._shaped:
                        return 1., robot.RobotEnv.Status.SUCCESS
                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
            if self._shaped:
                # Intermediate rewards for grasping and lifting
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