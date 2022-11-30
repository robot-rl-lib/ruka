import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from ruka.util.migrating import Migrating


@dataclass
class TargetObject(Migrating):
    reference_image: NDArray[np.uint8]
