import os
import time
import warnings
import typing
import logging
from typing import Union, List, Dict, Any, Optional

import gym
import numpy as np
import cv2
from ruka_os import in_cloud
from ruka.util import distributed_fs as dfs
import ruka.util.tensorboard as tb

from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward >= self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, '{}_{}_steps.pkl'.format(self.name_prefix, self.num_timesteps))
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True



class TrainingTimeCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(TrainingTimeCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.start_time = None
        self.start_tot= None
        self.start_simulator_time = None
        self.time_diffs = np.array([])
        self.sim_time_diffs = np.array([])
        self.tot_time = np.array([])

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_time = time.process_time()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.start_tot is None:
            self.start_tot = time.process_time()
        else:
            self.tot_time = np.append(time.process_time()-self.start_tot, self.tot_time)
            self.start_tot = None
        if self.start_simulator_time:
            time_diff = time.process_time() - self.start_simulator_time
            self.sim_time_diffs = np.append(self.sim_time_diffs, time_diff)

        if len(self.sim_time_diffs) >= 1000:
            logging.info("time takes for env step {}".format(np.mean(self.sim_time_diffs)))
            self.sim_time_diffs = np.array([])
        if len(self.tot_time) >= 1000:
            logging.info("time takes for one step in total {}".format(np.mean(self.tot_time)))
            self.tot_time = np.array([])

        return True

    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.num_timesteps == 0:
            return True
        else:
            end_time = time.process_time()
            time_diff = end_time - self.start_time 
            self.time_diffs = np.append(self.time_diffs, time_diff)
            if self.num_timesteps % 1000 == 0:
                logging.info("time takes for one training step {}".format(np.mean(self.time_diffs)))
                self.time_diffs = np.array([])

            self.start_time = end_time

        self.start_simulator_time = time.process_time()

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.start_time = time.process_time()


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            dfs.upload_maybe(path + '.zip')
        return True


class VideoRecorderCallback(BaseCallback):
    
    def __init__(
        self, test_env, save_freq: int, 
        save_path: str, 
        name_prefix: str = "rollout", 
        num_episodes: int = 1,
        max_steps_per_episode: int = 200,
        verbose: int = 0,
        determenistic: bool = True
        ):

        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.test_env = test_env
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes
        self.determenistic = determenistic

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            sync_envs_normalization(self.training_env, self.test_env)
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.avi")
            self.record_video(path)
            if self.verbose > 1:
                print(f"Recording video into checkpoint to {path}")
            dfs.upload_maybe(path)
        return True

    def record_video(self, path):
        images = []
        for _ in range(self.num_episodes):
            obs = (self.test_env.reset())
            images.append(self.test_env.env_method("get_image")[0])
            done = False
            for _ in range(self.max_steps_per_episode):
                action, _ = self.model.predict(obs, deterministic=self.determenistic)
                obs, reward, done, info = self.test_env.step(action)
                images.append(self.test_env.env_method("get_image")[0])
                if done:
                    break
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, images[0].shape[:2])
        for frame in images:
            out.write(frame)
        out.release()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Success rate is integrated to Tensorboard
    """
    def __init__(self, task, tf, algo, log_freq, model_name, verbose=0):
        self.is_tb_set = False
        self.task = task
        self.algo = algo
        self.log_freq = log_freq
        self.old_timestep = -1
        self.model_name = model_name
        self.tf = tf != None
        self.in_cloud = in_cloud()
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        history = self.task.get_attr("history")[0]
        rew = self.task.get_attr("episode_rewards")[0]
        sr = self.task.get_attr("sr_mean")[0]
        curr = self.task.get_attr("curriculum")[0]

        if len(history) != 0 and self.num_timesteps is not self.old_timestep:            
            if self.num_timesteps % self.log_freq == 0:
                logging.info("model: {} Success Rate: {} Timestep Num: {} lambda: {}".format(self.model_name, sr, self.num_timesteps, curr._lambda))
            if self.tf:
                self.logger.record("success_rate", sr)
            self.old_timestep = self.num_timesteps

        if self.in_cloud:
            tb.step(self.num_timesteps)
            tb.scalar("success_rate", sr)
        return True
        
    def on_training_end(self) -> None:
        tb.flush(wait=True)
        return super().on_training_end()
