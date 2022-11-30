import time
import numpy as np
from ruka.app.ecom.picker import EcomPicker, HomeRobot, Pick, Place
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
        self._env.set_goal({"ref_img": cmd.item.reference_img})
        ep_iterator = collect_episodes(
            self._env,
            self._pick_policy,
            reset_env=False
        )
        return next(ep_iterator)

    def handle_place(self, cmd: Place) -> Episode:
        # TODO: implement
        print("Implement place")
        return Episode()

    def handle_home(self, cmd: HomeRobot) -> Episode:
        # TODO: env requres goal image to perform reset
        #  it is better to remove such requirement
        #  or remove go home from reset
        self._env.set_goal({"ref_img": np.ones((32, 32, 3))})
        reset_start = time.time()
        self._env.reset()
        reset_end = time.time()

        ep = Episode()
        ep.meta['episode_time'] = reset_end - reset_start
        return ep