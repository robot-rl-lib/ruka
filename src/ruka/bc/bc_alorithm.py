import abc
import time

import pickle
from ruka.util import distributed_fs as dfs

from ruka.training.evaluator import Evaluator
from stable_baselines3.common.vec_env import VecNormalize
from ..training.buffer import VecEnvReplayBuffer

import ruka.util.tensorboard as tb

from ..training.logger import logger
from ..training import eval_util

class BaseBCAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            evaluation_env: VecNormalize,
            evaluation_data_collector: Evaluator,
            replay_buffer: VecEnvReplayBuffer,
            snapshot_every: int = 10,
    ):
        self.trainer = trainer
        self.eval_env = evaluation_env
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

        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        tb.step(self.trainer._num_train_steps)
        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        for k, v in self.trainer.get_diagnostics().items():
            tb.scalar('trainer/' + k, v)
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )

        for k, v in self.eval_data_collector.get_diagnostics().items():
            tb.scalar('evaluation/' + k, v)

        eval_paths = self.eval_data_collector.get_epoch_paths()
        if eval_paths:
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )

            eval_paths_info = eval_util.get_generic_path_information(eval_paths)

            logger.record_dict(
                eval_paths_info,
                prefix="evaluation/",
            )

            for k, v in eval_paths_info.items():
                tb.scalar('evaluation/' + k, v)

        """
        Misc
        """
        logger.record_tabular('Epoch', epoch)

        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass


class BCAlgorithm(BaseBCAlgorithm):
    def __init__(
            self,
            trainer,
            evaluation_env,
            evaluation_data_collector: Evaluator,
            replay_buffer: VecEnvReplayBuffer,
            batch_size,
            num_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_epoch,
            snapshot_every: int = 10,
    ):
        super().__init__(
            trainer,
            evaluation_env,
            evaluation_data_collector,
            replay_buffer,
            snapshot_every,
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_epoch = num_trains_per_epoch

    def _train(self):

        for epoch in range(self._start_epoch, self.num_epochs):
            if self.num_eval_steps_per_epoch is not None:
                self.eval_data_collector.collect_transitions(self.num_eval_steps_per_epoch)

            self.training_mode(True)

            for _ in range(self.num_trains_per_epoch):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)

                tb.step(self.trainer._num_train_steps)

            self.training_mode(False)
            self._end_epoch(epoch)

        tb.flush(wait=True)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
