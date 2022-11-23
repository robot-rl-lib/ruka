from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class TargetObject:
    reference_image: NDArray[np.uint8]
