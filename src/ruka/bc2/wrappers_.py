import gym 
from ruka.bc2.observation import Observation
import numpy as np
import collections


class WrapObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return Observation(observation)


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
        result = Observation()
        for key, value in observation.items():
            if len(value.shape) == 3:
                result[key] = np.transpose(value, axes=(2, 0, 1))
            else:
                result[key] = value
        return result


class SequenceWrapper(gym.Wrapper):
    def __init__(self, env, seq_len=None):
        super().__init__(env)


class SimulateCropWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop):
        self._crop = crop
        super().__init__(env)

    def observation(self, observation):
        
        for key, val in observation.items():
            if len(val.shape) != 3:
                continue
            c, w, h = val.shape
            lhw = (w - self._crop)//2
            rhw = self._crop + lhw

            lhh = (h - self._crop)//2
            rhh = self._crop + lhh

            observation[key] = val[:, lhw:rhw, lhh:rhh]


        return observation


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, discrete_step: float = 0.01, yaw_step: float = .1):
        super().__init__(env)

        self._discrete_step = discrete_step
        self._yaw_step = yaw_step

        self._x = [0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0, 0, 0]
        self._y = [0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0]
        self._z = [0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0]
        self._a = [0, 0, 0, 0, 0, 0, 0, self._yaw_step, -self._yaw_step, 0, 0]
        self._open_close = [0, 0, 0, 0, 0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step]
        self.action_space = gym.spaces.Discrete(len(self._x))

    def action(self, action: int):

        return super().action(
            self._x[action], self._y[action], 
            self._z[action], self._a[action], 
            self._open_close[action],
            )

class ContToDiscrete:
    def __init__(self, discrete_step: float = 0.01, yaw_step: float = .1):

        self._discrete_step = discrete_step
        self._yaw_step = yaw_step

        self._x = [0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0, 0, 0]
        self._y = [0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0]
        self._z = [0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0]
        self._a = [0, 0, 0, 0, 0, 0, 0, self._yaw_step, -self._yaw_step, 0, 0]
        self._open_close = [0, 0, 0, 0, 0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step]
        self._n = len(self._x)
        self.closed = None

    def _to_continious(self, action: int):
        return np.array([self._x[action], self._y[action], 
            self._z[action], self._a[action], 
            self._open_close[action],])
    def transform(self, action) -> int:
        if self.closed is None:
            self.closed = False
            return 9
        if (not self.closed) and action[-1] < 0:
            self.closed = True
            return 10
        if (self.closed) and action[-1] > 0:
            self.closed = False 
            return 9

        action[-1] = 0

        prods = []
        for i in range(self._n - 2):
            prods.append(action @ self._to_continious(i))
        return np.argmax(prods)