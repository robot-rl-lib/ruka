from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


class Object: # TODO: replace with common ruka.app ... Object
    reference_image: NDArray[np.uint8]
