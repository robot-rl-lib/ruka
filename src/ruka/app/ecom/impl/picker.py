from ruka.app.ecom.picker import EcomPicker, Pick, Place
from ruka.environments.common.env import Env, Episode, Policy
from ruka.environments.common.path_iterators import collect_episodes


class EcomPickerImpl(EcomPicker):
    def __init__(
        self,
        env: Env,
        pick_policy: Policy,
        # TODO: implement
        # place_policy: Policy,
    ):
        self._env = env
        self._pick_policy = pick_policy
        # TODO: implement
        # self._place_policy = place_policy

    def handle_pick(self, cmd: Pick) -> Episode:
        ep_iterator = collect_episodes(self._env, self._pick_policy)
        return next(ep_iterator)

    def handle_place(self, cmd: Place) -> Episode:
        # TODO: implement
        print("Implement place")
        return Episode()
