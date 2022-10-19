import gym


class Controller:
    @property
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def act(self, action):
        """
        Can have wait=True or wait=False semantics (depending on the specific
        action even).
        """
        raise NotImplementedError()