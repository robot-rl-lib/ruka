
import gym
from ruka.bc2.observation import Observation

class WrapObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return Observation(observation)