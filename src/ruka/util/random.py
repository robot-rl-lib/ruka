import numpy as np
import random
import torch


def determinism_on():
    torch.backends.cudnn.deterministic = True


def seed_everything(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)