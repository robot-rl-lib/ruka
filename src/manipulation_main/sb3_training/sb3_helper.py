import logging
import os

import cv2
import ruka.util.tensorboard as tb
import stable_baselines3 as sb
import tensorflow as tf
from manipulation_main.sb3_training.wrapper import ImageToPyTorch
from ruka.util import distributed_fs as dfs
from ruka_os import in_cloud
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization)
from stable_baselines3.sac.policies import CnnPolicy as sacCnn
from stable_baselines3.sac.policies import MlpPolicy as sacMlp

from . import custom_obs_policy
from .base_callbacks import (CheckpointCallback, EvalCallback,
                             SaveVecNormalizeCallback, TensorboardCallback,
                             VideoRecorderCallback)


class SBPolicy:
    def __init__(self, env, test_env, config, model_dir, 
                load_dir=None, algo='SAC', log_freq=1000):
        self.env = env
        self.test_env = test_env
        self.algo = algo
        self.config = config
        self.load_dir = load_dir
        self.model_dir = model_dir
        self.log_freq = log_freq
        self.norm = config['normalize']
        self.is_oracle = len(self.env.observation_space.shape)==1

    def learn(self):
        # Use deterministic actions for evaluation
        eval_path = self.model_dir + "/best_model"
        # TODO save checkpoints with vecnormalize callback pkl file
        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=eval_path)
        if self.norm:
            # Don't normalize the reward for test env
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False,
                                        clip_obs=10.)
        eval_callback = EvalCallback(self.test_env, best_model_save_path=eval_path,
                                    log_path=eval_path+'/logs', eval_freq=10000,
                                    n_eval_episodes=10, callback_on_new_best=save_vec_normalize,
                                    deterministic=True, render=False)
        tensorboard_file = None if self.config[self.algo]['tensorboard_logs'] is None else "tensorboard/"
        if self.algo == 'SAC':
            if self.is_oracle:
                policy_kwargs = {"net_arch": dict(qf=self.config[self.algo]['layers'], pi=self.config[self.algo]['layers']),}
                policy = sacMlp
            elif not self.env.envs[0].is_simplified() and (self.env.envs[0].depth_obs or self.env.envs[0].full_obs):
                policy_kwargs = {
                    "net_arch": dict(qf=self.config[self.algo]['layers'], pi=self.config[self.algo]['layers']),
                    "features_extractor_class": custom_obs_policy.AugmentedNatureCNN,
                    "features_extractor_kwargs": dict(num_direct_features=1),
                    # "normalize_images": False,
                    }
                policy = sacCnn
            elif self.env.envs[0].depth_obs or self.env.envs[0].full_obs:
                policy_kwargs = {}
                policy = sacCnn
            else:
                policy_kwargs = {"net_arch": self.config[self.algo]['layers']}# , "layer_norm": False}
                policy = sacMlp
            if self.load_dir:
                top_folder_idx = self.load_dir.rfind('/')
                top_folder_str = self.load_dir[0:top_folder_idx]
                if self.norm:
                    self.env = VecNormalize(self.env, training=True, norm_obs=False, norm_reward=False,
                                            clip_obs=10.)
                    self.env = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), self.env)
                model = sb.SAC(policy,
                            self.env,
                            policy_kwargs=policy_kwargs,
                            verbose=1,
                            gamma=self.config['discount_factor'],
                            buffer_size=self.config[self.algo]['buffer_size'],
                            batch_size=self.config[self.algo]['batch_size'],
                            learning_rate=self.config[self.algo]['step_size'],
                            tensorboard_log=tensorboard_file)
                model_load = sb.SAC.load(self.load_dir, self.env)
                params = model_load.get_parameters()
                model.load_parameters(params, exact_match=False)
            else:
                if self.norm:
                    self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True,
                                            clip_obs=10.)
                model = sb.SAC(policy,
                            self.env,
                            policy_kwargs=policy_kwargs,
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            buffer_size=self.config[self.algo]['buffer_size'],
                            batch_size=self.config[self.algo]['batch_size'],
                            learning_rate=self.config[self.algo]['step_size'],
                            tensorboard_log=tensorboard_file)
        elif self.algo == 'TRPO':
            raise NotImplementedError

        elif self.algo == 'PPO':
            raise NotImplementedError

        elif self.algo == 'DQN':
            raise NotImplementedError

        elif self.algo == "DDPG":
            raise NotImplementedError

        try:
            model.learn(total_timesteps=int(self.config[self.algo]['total_timesteps']), 
                        callback=[TensorboardCallback(self.env, tensorboard_file, self.algo, self.log_freq, self.model_dir), 
                                   eval_callback, CheckpointCallback(save_freq=10_000, save_path=self.model_dir),
                                   VideoRecorderCallback(test_env=self.test_env, save_freq=20_000, save_path=self.model_dir)])
        except KeyboardInterrupt:
            pass

        self.save(model, self.model_dir, "end_model")

    def load_params(self):
        raise NotImplementedError
    
    def save(self, model, model_dir, model_name):

        folder_path = model_dir + '/' + model_name

        if os.path.isfile(folder_path):
            print('File already exists \n')
            i = 1
            while os.path.isfile(folder_path + '.zip'):
                folder_path = '{}_{}'.format(folder_path, i)
                i += 1
            model.save(folder_path)
        else:
            print('Saving model to {}'.format(folder_path))
            model.save(folder_path)

        if self.norm:
            model.get_vec_normalize_env().save(os.path.join(model_dir, 'vecnormalize.pkl'))
