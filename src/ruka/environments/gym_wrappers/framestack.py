import gym 
import numpy as np
import collections

class AugmentedFramestackWrapper(gym.Wrapper):
    def __init__(self, env, nstack=3) -> None:
        super().__init__(env)
        self.nstack = nstack
        self.framestack = None

        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                old_shape[0], 
                old_shape[1], 
                (old_shape[2]-1) * self.nstack + 1,
                ),
            dtype=np.float64
        )
    def reset(self):
        obs = self.env.reset()
        self.framestack = collections.deque([obs[..., :-1]]*self.nstack, maxlen=self.nstack)
        return np.concatenate([*self.framestack, obs[..., -1:]], axis=-1)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.framestack.append(obs[..., :-1])
        return np.concatenate([*self.framestack, obs[..., -1:]], axis=-1), rew, done, info
