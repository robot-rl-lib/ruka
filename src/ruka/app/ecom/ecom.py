import numpy as np

from typing import List
from dataclasses import dataclass


@dataclass
class SKU:
    name: str
    reference_img: np.ndarray


@dataclass
class Basket:
    items: List[SKU]


class Command:
    pass
