import gym


class Controller:
    @property
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    def act(self, action):
        raise NotImplementedError()