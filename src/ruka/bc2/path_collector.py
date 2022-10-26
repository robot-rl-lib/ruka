from collections import deque, OrderedDict
from typing import List
from ruka.models.policy import ZeroPolicy
from .base import Path


class PathCollector:
    def __init__(
            self,
            env,
            policy,
            continue_last_path: bool = True,
            zero_action_steps = 0
    ):
        self._env = env
        self._policy = policy
        self._last_obs = None
        self._continue_last_path = continue_last_path
        self._zero_action_steps = zero_action_steps
        self._num_collected_steps = 0

    def collect_paths(
            self,
            num_paths: int,
            max_path_len: int = 200,
    ) -> List[Path]:

        output_paths = []

        if self._continue_last_path and (self._last_obs is not None):
            obs = self._last_obs
        else:
            obs = self._env.reset()
        for _ in range(num_paths):
            done = False
            observations = [obs]
            actions = []
            dones = []
            rewards = []
            infos = []

            for _ in range(max_path_len):
                action = self._policy.get_action(obs)
                obs, rew, done, info = self._env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(rew)
                dones.append(done)
                infos.append(info)

                if done:
                    break

            path = Path(
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos
            )
            output_paths.append(path)

            obs = self._env.reset()

        return output_paths