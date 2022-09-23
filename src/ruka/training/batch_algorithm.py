import abc

from .buffer import ReplayBuffer, VecEnvReplayBuffer
from .path_collector import MdpPathCollector, VecTransitionCollector

import abc
from collections import OrderedDict

from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
import ruka.util.tensorboard as tb
from ruka.training.evaluator import Evaluator
import pickle
from ruka.util import distributed_fs as dfs

from ruka.sac.sac import TorchTrainer

try:
    import wandb
    HAS_WANDB = True
except:
    HAS_WANDB = False
    

from . import eval_util
from .logger import logger
from ruka_os import in_cloud

import time

SHOULD_PRINT = not in_cloud()
class PrintTime:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, type, value, traceback):
        spent = time.time() - self.start
        if SHOULD_PRINT:
            print(f"{self.name}: {spent:.2f}")


def _get_epoch_timings():
    return {}

class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer: TorchTrainer,
            exploration_env: VecNormalize,
            evaluation_env: VecNormalize,
            exploration_data_collector: VecTransitionCollector,
            # evaluation_data_collector: VecTransitionCollector,
            evaluation_data_collector: Evaluator,
            replay_buffer: VecEnvReplayBuffer,
            snapshot_every: int = 10,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []
        self._snapshot_every = snapshot_every

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        
        if (epoch % self._snapshot_every) == 0:
            print(f"saving {epoch}.snapshot")
            snapshot = self._get_snapshot()
            with open(f"{epoch}.snapshot", "wb") as f:
                pickle.dump(snapshot, f)
            print(f"uploading {epoch}.snapshot")
            dfs.upload_maybe(f"{epoch}.snapshot")

        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        use_wandb = HAS_WANDB and (wandb.run is not None)
        tb.step(self.trainer._num_train_steps)
        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )
        if use_wandb:
            wandb.log({'replay_buffer/' + k: v for k, v in self.replay_buffer.get_diagnostics().items()}, step=epoch, commit=False)

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        if use_wandb:
            wandb.log({'trainer/' + k: v for k, v in self.trainer.get_diagnostics().items()}, step=epoch, commit=False)
        
        for k, v in self.trainer.get_diagnostics().items():
            tb.scalar('trainer/' + k, v)
        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        if use_wandb:
            wandb.log({'exploration/' + k: v for k, v in self.expl_data_collector.get_diagnostics().items()}, step=epoch, commit=False)
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if expl_paths:
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )
                if use_wandb:
                    wandb.log({'exploration/' + k: v for k, v in self.expl_env.get_diagnostics(expl_paths).items()}, step=epoch, commit=False)
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )
            if use_wandb:
                wandb.log({'exploration/' + k: v for k, v in eval_util.get_generic_path_information(expl_paths).items()}, step=epoch, commit=False)
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        if use_wandb:
            wandb.log({'evaluation/' + k: v for k, v in self.eval_data_collector.get_diagnostics().items()}, step=epoch, commit=False)
        
        for k, v in self.eval_data_collector.get_diagnostics().items():
            tb.scalar('evaluation/' + k, v)
            
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if eval_paths:
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )
                if use_wandb:
                    wandb.log({'evaluation/' + k: v for k, v in self.eval_env.get_diagnostics(eval_paths).items()}, step=epoch, commit=False)
            
            eval_paths_info = eval_util.get_generic_path_information(eval_paths)

            logger.record_dict(
                eval_paths_info,
                prefix="evaluation/",
            )
            if use_wandb:
                wandb.log({'evaluation/' + k: v for k, v in eval_paths_info.items()}, step=epoch, commit=False)

            for k, v in eval_paths_info.items():
                tb.scalar('evaluation/' + k, v)
                
        """
        Misc
        """
        # logger.record_dict(_get_epoch_timings())
        logger.record_dict(self.timings)
        for k, v in self.timings.items():
            tb.scalar('timings/' + k, v)

        if use_wandb:
            wandb.log(_get_epoch_timings(), step=epoch, commit=False)
        logger.record_tabular('Epoch', epoch)
        if use_wandb:
            wandb.log(dict(epoch=epoch), step=epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass



class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: VecTransitionCollector,
            # evaluation_data_collector: VecTransitionCollector,
            evaluation_data_collector: Evaluator,
            replay_buffer: VecEnvReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            snapshot_every: int = 10,
            warmup_steps: int = 0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            snapshot_every,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.warmup_steps = warmup_steps

    def _train(self):
        print("Filling up replay buffer...")
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_transitions(self.min_num_steps_before_training)
            self.replay_buffer.add_vec_transitions(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        print("The buffer is filled!")
        start_train_timestamp = time.time()

        for _ in range(self.warmup_steps):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            self.trainer.train(train_data)
            
        for epoch in range(self._start_epoch, self.num_epochs):
            start_epoch_timestamp = time.time()
            self.eval_data_collector.collect_transitions(self.num_eval_steps_per_epoch)
            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_transitions(self.num_expl_steps_per_train_loop)
                self.replay_buffer.add_vec_transitions(new_expl_paths)
                
                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)

                    tb.step(self.trainer._num_train_steps)
                    tb.scalar("success_rate", self.expl_env.get_attr("sr_mean")[0])

                self.training_mode(False)

            self.timings = {
                'EPOCH_TIME': time.time() - start_epoch_timestamp,
                "TOTAL_TIME": time.time() - start_train_timestamp,
                "SUCCES RATE": self.expl_env.get_attr("sr_mean")[0],
                "SUCCES RATE EVAL": self.eval_env.get_attr("sr_mean")[0],
                "CURRICULUM LAMBDA": self.expl_env.get_attr("curriculum")[0]._lambda,
                "FPS": self.num_expl_steps_per_train_loop*self.num_train_loops_per_epoch/\
                    (time.time() - start_epoch_timestamp)
            }
            
            sync_envs_normalization(self.expl_env, self.eval_env)
            self._end_epoch(epoch)
            
        tb.flush(wait=True)
            


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)