import abc
import cv2
import uuid
import numpy as np
import collections
import queue as q
from copy import deepcopy
import torch.multiprocessing as mp
from threading import Thread
from typing import Optional, Callable, Dict, Union

import ruka.pytorch_util as ptu
import ruka.util.tensorboard as tb
from ruka.models.policy import Policy
from ruka.observation import Observation
from ruka.models.policy import ZeroPolicy                       
from ruka.util import distributed_fs as dfs
from .rollouts import vec_rollout, join_rollout
from stable_baselines3.common.vec_env import VecEnv
from ruka.environments.common.env_util import get_supported_robot_env


def _worker_loop(env: VecEnv, 
                policy: Policy,
                rollout_fn: Callable,
                chunk_size: int, 
                in_policy_queue: Union[q.Queue, mp.Queue],
                out_transition_queue: Union[q.Queue, mp.Queue],
                save_video_every: int = 10,
                video_prefix: Optional[str] = None,
                zero_action_steps: int = 0,
                device: Optional[str] = None,
                name: Optional[str] = None):
    """ Collect transitions by chunk size in parallel worker """
    print(f'Worker start {name}', flush=True)
    chunk_number = 0
    num_collected_steps = 0
    while True:
        last_obs = None
        
        cur_policy = policy
        if num_collected_steps < zero_action_steps:
            cur_policy = ZeroPolicy(env)

        transitions = rollout_fn(
            vec_env=env,
            agent=cur_policy,
            num_steps=chunk_size//env.num_envs,
            last_obs=last_obs,
            save_image=video_prefix and chunk_number % save_video_every == 0,
            device=device,
        )

        last_obs = transitions['next_observations'].select_by_index(-1) if isinstance(transitions['next_observations'], Observation) else transitions['next_observations'][-1]
        
        out_transition_queue.put(transitions)
        new_policy_state = in_policy_queue.get()

        chunk_number += 1
        num_collected_steps += chunk_size

        if new_policy_state is None:
            break
        elif isinstance(new_policy_state, str) and new_policy_state == 'NO_UPDATE':
            continue
        else:
            policy.load_state_dict(new_policy_state)    
    print(f'Worker terminate {name}', flush=True)


class BaseAsyncVecTransitionCollector(object, metaclass=abc.ABCMeta):
    
    def __init__(
            self,
            orig_policy,
            rollout_fn: Callable = vec_rollout,
            chunk_size: int = 100,
            device: str = None,
            save_video_every: int = 10,
            video_prefix: Optional[str] = None,            
            return_paths: bool = False,
            zero_action_steps: int = 0,
            sr_window_size: int = 10,
            name: Optional[str] = None,
    ):
        self.device = device
        self._orig_policy = orig_policy
        self._name = name if name is not None else uuid.uuid4()

        self._chunk_size = chunk_size
        self._rollout_fn = rollout_fn
        self._chunk_number = 0
        self._zero_action_steps = zero_action_steps

        self._save_video_every = save_video_every
        self._video_prefix = video_prefix
    
        self._success = collections.deque([], sr_window_size)

        # if True get_epoch_paths return collected data
        self._return_paths = return_paths
        self._paths = []
        self._epoch = 0
        self._path_rewards = []
        self._path_actions = []
        self._path_info = []
  
    def start(self):
        raise NotImplementedError()
        
    def stop(self):
        raise NotImplementedError()

    def _recv_transitions(self):
        raise NotImplementedError()

    def _send_policy(self, policy_data):
        raise NotImplementedError()

    def collect_transitions(
            self,
            num_transitions,
    ):
        assert num_transitions >= self._chunk_size and num_transitions % self._chunk_size == 0, \
                                                            (num_transitions, self._chunk_size)
        rollouts = []
        for i in range(num_transitions//self._chunk_size):
            transitions = self._recv_transitions()
            
            if i == 0:
                # send model weights only first time
                self._send_policy(self._orig_policy.state_dict())
            else:
                self._send_policy('NO_UPDATE')
            
            
            self.save_paths_and_calc_sr(transitions)
            
            self.maybe_save_video(transitions)
            rollouts.append(transitions)
            self._chunk_number += 1

        if len(rollouts) > 1:
            return join_rollout(rollouts)
        else:
            assert len(rollouts) == 1, len(rollouts)
            return rollouts[0]


    def get_epoch_paths(self):
        if self._return_paths:
            return self._paths
        return None

    def end_epoch(self, epoch):
        self._epoch = epoch
        self._paths = []
        return

    def get_diagnostics(self):
        return {}

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

    def maybe_save_video(self, transitions):
        if self._video_prefix and self._chunk_number % self._save_video_every == 0:
            video_path = f"{self._video_prefix}_{self._chunk_number}.avi"
            print(f"Recording video to {video_path}")
            frames = transitions['images']
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, frames[0].shape[:2][::-1])
            for frame in frames:
                out.write(frame)
            out.release()
            dfs.upload_maybe(video_path)
            tb.add_video('videos', np.array(frames).transpose(0, 3, 1, 2)[None].astype(np.uint8))   
            del transitions['images']        

    def get_diagnostics(self):
        return {
            "success_rate": np.mean(self._success),
            "mean_path_len": np.mean([len(x['actions']) for x in self._paths]) if self._paths else -1,
            }

class AsyncVecTransitionCollector(BaseAsyncVecTransitionCollector):
    def __init__(
            self,
            env: VecEnv,
            orig_policy: Policy,
            *args,
            **kwargs
    ):
        super().__init__(orig_policy, *args, **kwargs)

        self._env = env
        self._orig_policy = orig_policy
        self._worker_policy = deepcopy(orig_policy)
        self._worker_policy.to(self.device or ptu.device)
        self._worker_policy.train(False)

        self._policy_queue = q.Queue(maxsize=1)
        self._transition_queue = q.Queue(maxsize=1)
        self._thread = Thread(target=_worker_loop, 
                              args=(self._env, self._worker_policy, self._rollout_fn, self._chunk_size,
                                    self._policy_queue, self._transition_queue, self._save_video_every, self._video_prefix,
                                    self._zero_action_steps, self.device))

    def start(self):
        print(f'Starting {self._name}...', flush=True)
        self._thread.start()
        
    def stop(self):
        print(f'Terminating {self._name}...', flush=True)
        self._policy_queue.put(None)
        self._thread.join()
        print(f'Stopped {self._name}', flush=True)

    def _recv_transitions(self):
        return self._transition_queue.get()

    def _send_policy(self, policy_data):
        return self._policy_queue.put(policy_data)

 

class ParallelVecTransitionCollector(BaseAsyncVecTransitionCollector):
    def __init__(
            self,
            make_env: Callable,
            orig_policy: Policy,
            make_policy: Callable,
            *args,
            **kwargs
    ):
        super().__init__(orig_policy, *args, **kwargs)            

        self._orig_policy = orig_policy

        self._policy_queue = mp.Queue(maxsize=1)
        self._transition_queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(target=ParallelVecTransitionCollector._process_worker, 
                              args=(make_env, make_policy, orig_policy.state_dict(), self._rollout_fn, self._chunk_size,
                                    self._policy_queue, self._transition_queue, self._save_video_every, self._video_prefix,
                                    self._zero_action_steps, self.device, self._name))

    @staticmethod
    def _process_worker(make_env: Callable, 
                    make_policy: Callable,
                    policy_state_dict: Optional[Dict],
                    rollout_fn: Callable,
                    chunk_size: int, 
                    in_policy_queue: mp.Queue,
                    out_transition_queue: mp.Queue,
                    save_video_every: int = 10,
                    video_prefix: Optional[str] = None,
                    zero_action_steps: int = 0,
                    device: Optional[str] = None,
                    name: Optional[str] = None):

        """ Collect transitions in parallel thread """
        env = make_env()
        policy = make_policy().to(device or ptu.device).eval()
        if policy_state_dict:
            policy.load_state_dict(policy_state_dict)

        _worker_loop(env,
                    policy,
                    rollout_fn,
                    chunk_size,
                    in_policy_queue,
                    out_transition_queue,
                    save_video_every,
                    video_prefix,
                    zero_action_steps,
                    device,
                    name)

    def start(self):
        print(f'Starting {self._name}...', flush=True)
        self._worker.start()
        
    def stop(self):
        print(f'Terminating {self._name}...', flush=True)
        self._worker.terminate()
        self._worker.join()
        print(f'Stopped {self._name}', flush=True)

    def _recv_transitions(self):
        return self._transition_queue.get()

    def _send_policy(self, policy_data):
        if isinstance(policy_data, dict):
            # for prevent CUDA eror while send via Queue
            for k,v in policy_data.items():
                policy_data[k] = v.to('cpu:0')

        return self._policy_queue.put(policy_data)


