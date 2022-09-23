from ruka.models.policy import TorchStochasticPolicy
from ruka.models.qf import QFunctionPair
from ruka.training.buffer import VecEnvReplayBuffer
from ruka.training.path_collector import VecTransitionCollector
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from .sac import Trainer
import torch.nn as nn

# TODO: Move all Trainer insides to the algorithm?
# TODO: Policy

class SACAlgorithm:
    def __init__(
        self,
        env: VecEnv,
        buffer: VecEnvReplayBuffer, 
        collector: VecTransitionCollector,
        trainer: Trainer,
        num_train_loops_per_epoch: int,
        batch_size: int,
        num_expl_steps_per_train_loop: int,
        num_trains_per_train_loop: int,
        min_num_steps_before_training: int,
        ):
        self.env = env
        self.buffer = buffer
        self.collector = collector
        self.trainer = trainer

        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.batch_size = batch_size
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def train_one_epoch(self):
        for _ in range(self.num_train_loops_per_epoch):
            collected_data = self.collector.collect_transitions(self.num_expl_steps_per_train_loop)
            self.buffer.add_vec_transitions(collected_data)
            self._train_loop()


    def _train_loop(self):
        for _ in range(self.num_trains_per_train_loop):
            batch = self.buffer.random_batch(self.batch_size)
            self.trainer.train(batch)


    def training_mode(self, flag: bool):
        for net in self.trainer.networks:
            net.train(flag)

    def before_training(self):
        collected_data = self.collector.collect_transitions(self.min_num_steps_before_training)
        self.buffer.add_vec_transitions(collected_data)

    def get_eval_policy(self) -> nn.Module:
        pass