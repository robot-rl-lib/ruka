from typing import Optional

import torch
import torch.nn as nn
from ruka.types import Dictator

class Loss(nn.Module):
    def forward(batch: Dictator) -> torch.Tensor:
        raise NotImplementedError()

    def log_stats(self, step: int, prefix: Optional[str] = None):
        pass
