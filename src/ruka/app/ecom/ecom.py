import numpy as np

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SKU:
    name: str
    reference_img: np.ndarray
    uri: Optional[str] = None


@dataclass
class Basket:
    items: List[SKU]


class Command:
    pass
