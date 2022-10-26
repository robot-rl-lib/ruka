import numpy as np
from email import policy
from .rollouts import rollout, vec_rollout
from collections import deque, OrderedDict
from stable_baselines3.common.vec_env import VecEnv
from ruka.observation import Observation
from ruka.models.policy import ZeroPolicy


class MdpPathCollector:
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            continue_last_path=False,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._continue_last_path = continue_last_path
        self._last_obs = None
        self._num_steps_since_reset = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length - self._num_steps_since_reset,
                num_steps - num_steps_collected,
            )
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                last_obs=self._last_obs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
            if self._continue_last_path:
                self._last_obs = path['full_next_observations'][-1]
                self._num_steps_since_reset += path_len
                if self._num_steps_since_reset == max_path_length or path['terminals'][-1]:
                    self._last_obs = None
                    self._num_steps_since_reset = 0

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict



class VecTransitionCollector:
    def __init__(
            self,
            env: VecEnv,
            policy,
            continue_last_path: bool = True,
            rollout_fn=vec_rollout,
            zero_action_steps = 0,
            return_paths = False,
    ):
        self._env = env
        self._policy = policy
        self._rollout_fn = rollout_fn
        self._last_obs = None
        self._continue_last_path = continue_last_path
        self._zero_action_steps = zero_action_steps
        self._num_collected_steps = 0

        self._return_paths = return_paths
        self._path_rewards = []
        self._path_actions = []
        self._path_info = []    
        self._paths = []    
        self._success = deque(maxlen=10)

    def collect_transitions(
            self,
            num_transitions,
    ):

        policy = self._policy
        if self._num_collected_steps < self._zero_action_steps:
            print('Zero policy step', num_transitions)
            policy = ZeroPolicy(self._env)

        transitions = vec_rollout(
            vec_env=self._env,
            agent=policy,
            num_steps=num_transitions//self._env.num_envs,
            last_obs=self._last_obs
        )
        if self._continue_last_path:
            self._last_obs = transitions['next_observations'].select_by_index(-1) if isinstance(transitions['next_observations'], Observation) else transitions['next_observations'][-1]

        self._num_collected_steps += num_transitions
        self.save_paths_and_calc_sr(transitions)
        return transitions

    def get_epoch_paths(self):
        if self._return_paths:
            return self._paths

    def end_epoch(self, epoch):
        self._paths = []
        return
    
    def get_diagnostics(self):
        return {
            "success_rate": np.mean(self._success),
            "mean_path_len": np.mean([len(x['actions']) for x in self._paths]) if self._paths else -1,
            }

    def get_snapshot(self):
        return {}


    def save_paths_and_calc_sr(self, transitions):
        for i,t in enumerate(transitions['terminals']):
            if self._return_paths:
                self._path_rewards.append(transitions['rewards'][i])
                self._path_actions.append(transitions['actions'][i])
                    
            # used for calc SR
            self._path_info.append({'is_success': transitions['env_infos'][i][0]['is_success']})
            
            if t:
                if self._return_paths:


                    self._paths.append(dict(
                        actions=np.array(self._path_actions),
                        rewards=np.array(self._path_actions).reshape((-1,1)),
                        env_infos=self._path_info,
                    ))     
                self._success.append(np.any([v['is_success'] for v in self._path_info]))
                self._path_rewards = []
                self._path_actions = []
                self._path_info = []