from re import L
from typing import Tuple
import torch.nn as nn
import torch
from .cnn_encoders import BaseFeaturesExtractor
from .mlp import ConcatMlp, Mlp
import abc

class QFunctionPair(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        qf1: ConcatMlp,
        qf2: ConcatMlp,
        ):
        super().__init__()
        self.qf1 = qf1
        self.qf2 = qf2
    
    @abc.abstractmethod
    def forward(
        self, observations: torch.Tensor, 
        *inputs, **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        return NotImplementedError

class SharedEncCritics(QFunctionPair):
    def __init__(
        self,
        encoder: BaseFeaturesExtractor,
        qf1: ConcatMlp,
        qf2: ConcatMlp,
        ):

        super().__init__(qf1=qf1, qf2=qf2)
        self._encoder = encoder
    def forward(self, observations, *inputs, **kwargs):
        encoded_observations = self._encoder(observations)
        return (
            self.qf1(encoded_observations, *inputs, **kwargs),
            self.qf2(encoded_observations, *inputs, **kwargs)
            )

    