import gym
from copy import deepcopy
from typing import Dict
from ruka.util.nested_dict import NestedDict


class GoalCondition(gym.ObservationWrapper):
    def __init__(self,  env: gym.Env, goal_spaces: Dict[str, gym.spaces.Space]):
        self.env = env
        self._goal = None

        self.observation_space = deepcopy(env.observation_space)
        for goal_item_key, goal_item_spaces in goal_spaces.items():
            if not isinstance(goal_item_spaces, gym.spaces.Space):
                raise RuntimeError(f'goal space should derive from gym.spaces.Space')
            self.observation_space[goal_item_key] = goal_item_spaces

    def observation(self, observation: Dict):
        assert self._goal is not None, 'Need to set_goal before start'
        observation.update(self._goal)
        return observation

    def get_observation(self):
        return self.observation(self.env.get_observation())

    def set_goal(self, goal: NestedDict):
        # TODO: add check of condition shape == condition_space
        self._goal = goal
