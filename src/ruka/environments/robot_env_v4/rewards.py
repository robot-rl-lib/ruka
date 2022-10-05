from ast import Delete
from . import robot
from . import configs

class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, config: configs.EnvironmentConfig.RewardConfig, robot):
        self._robot = robot
        self._max_delta_z = robot._actuator._max_translation
        self._terminal_reward = config.terminal_reward
        self._grasp_reward = config.grasp_reward
        self._delta_z_scale = config.delta_z_scale
        self._lift_success = config.lift_success
        self._time_penalty = config.time_penalty
        self._table_clearing = config.table_clearing
        self._terminal_reward_wrong = config.terminal_reward_wrong
        self.lift_dist = None

        self._lifting = False
        self._start_height = None
        self._old_robot_height = None

    def __call__(self):
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        if self._robot._actuator.object_detected():
            current_highest_id = self._robot.find_highest()


            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:
                if current_highest_id==self._robot.target_object.model_id:
                    if self._table_clearing:

                        # self._robot.target_object.delete()
                        self._robot.remove_model(self._robot.target_object.model_id)
                        self._robot.pickable_objects.pop()
                        self._robot.run(.1)

                        if self._robot.pickable_objects:
                            self._robot.target_object = self._robot.pickable_objects[-1]
                            return self._lift_success, robot.RobotEnv.Status.CLEARING_OBJECT_PICKED

                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                else:
                    return self._terminal_reward_wrong, robot.RobotEnv.Status.PICKED_WRONG


            delta_z = robot_height - self._old_robot_height
            reward = self._grasp_reward + self._delta_z_scale * delta_z

        else:
            self._lifting = False
        reward -= self._time_penalty
        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING

    def reset(self):
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]