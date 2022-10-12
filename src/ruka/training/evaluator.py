from email import policy
from typing import Callable, Optional
import gym
from ruka.models.policy import TorchStochasticPolicy
import cv2
from ruka.util import distributed_fs as dfs
import numpy as np
import ruka.util.tensorboard as tb
from ruka.observation import Observation

class Evaluator:
    def __init__(
            self,
            env: gym.Env,
            policy: TorchStochasticPolicy,
            save_video_every: int = 10,
            video_prefix: Optional[str] = None,
            prepocess_fn: Callable = lambda x: x,
    ):
        self._env = env
        self._policy = policy
        self._video_prefix = video_prefix
        self._paths = []
        self._epoch = 0
        self._save_video_every = save_video_every
        self._preprocess_fn = prepocess_fn
        self._closes = []

    def collect_transitions(
        self,
        num_transitions: int,
        epoch: int = None,
        save_video: Optional[bool] = None):

        if epoch is None:
            epoch = self._epoch

        if save_video is None:
            save_video = ((epoch % self._save_video_every)==0)

        rewards = []
        actions = []
        infos = []

        path_rewards = []
        path_actions = []
        path_infos = []

        self._paths = []

        paths_frames = []
        frames = []

        obs = self._env.reset()
        if isinstance(obs, dict):
            obs = Observation(obs)

        nclose = 0
        prev_gripper_command = -1
        for _ in range(num_transitions):
            action = self._policy.get_action(self._preprocess_fn(obs))
            obs, reward, done, info = self._env.step(action)
            if isinstance(obs, dict):
                obs = Observation(obs)

            if action[4]>0 and prev_gripper_command < 0:
                nclose += (action[4]>0)
            prev_gripper_command = action[4]

            rewards.append(reward)
            actions.append(action)
            infos.append({})

            path_rewards.append(reward)
            path_actions.append(action)
            path_infos.append({})

            if self._video_prefix and save_video:
                frames.append(self._env.get_image())

            if done:
                self._paths.append(dict(
                    actions=np.array(path_rewards),
                    rewards=np.array(path_actions).reshape((-1,1)),
                    env_infos=path_infos,
                ))
                paths_frames.extend(frames)
                obs = self._env.reset()
                if isinstance(obs, dict):
                    obs = Observation(obs)

                self._closes.append(nclose)
                nclose = 0
                path_rewards = []
                path_actions = []
                path_infos = []


        if self._video_prefix and save_video:
            video_path = f"{self._video_prefix}_{epoch}.avi"
            print(f"Recording video to {video_path}")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, frames[0].shape[:2][::-1])
            for frame in frames:
                out.write(frame)
            out.release()
            dfs.upload_maybe(video_path)
            tb.add_video('videos', np.array(frames).transpose(0, 3, 1, 2)[None].astype(np.uint8))
        return self._paths

    def get_epoch_paths(self):
        return self._paths

    def end_epoch(self, epoch):
        self._epoch = epoch
        self._paths = []
        self._closes = []
        return

    def get_diagnostics(self):

        return {
            "success_rate": self._env.sr_mean,
            "mean_path_len": np.mean([len(x['actions']) for x in self._paths]) if self._paths else -1,
            "mean_gripper_closes": np.mean(self._closes) if self._closes else -1
            }

    def get_snapshot(self):
        return {}



