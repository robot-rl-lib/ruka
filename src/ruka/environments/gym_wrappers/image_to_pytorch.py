import gym 
import numpy as np

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float64
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


class ImageToPyTorchDictLike(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorchDictLike, self).__init__(env)

        space_dict = dict()
        for key, space in env.observation_space.items():
            if len(space.shape) == 3:
                space_dict[key] = gym.spaces.Box(low=0, high=1, shape=(space.shape[-1], space.shape[0], space.shape[1],))
            else:
                space_dict[key] = gym.spaces.Box(low=space.low, high=space.high, shape=space.shape)

        self.observation_space = gym.spaces.Dict(space_dict)

    def observation(self, observation):
        result = dict()
        for key, value in observation.items():
            if len(value.shape) == 3:
                result[key] = np.transpose(value, axes=(2, 0, 1))
            else:
                result[key] = value
        return result

