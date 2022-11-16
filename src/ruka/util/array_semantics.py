import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any


@dataclass
class RGB:
    """
    Allowed shapes:
        [H, W, 3]

    Allowed dtype:
        np.uint8
    """

    buf: NDArray[np.uint8]

    def __getstate__(self) -> Any:
        return {'buf': self.buf}

    def __setstate__(self, state: Any):
        self.buf = state['buf']


@dataclass
class Grayscale:
    """
    Allowed shapes:
        [H, W, 1]

    Allowed dtype:
        np.uint8
    """

    buf: NDArray[np.uint8]

    def __getstate__(self) -> Any:
        return {'buf': self.buf}

    def __setstate__(self, state: Any):
        self.buf = state['buf']


@dataclass
class Depth:
    """
    Allowed shapes:
        [H, W, 1]

    Allowed dtype:
        np.uint16
    """

    buf: NDArray[np.uint16]

    def __getstate__(self) -> Any:
        return {'buf': self.buf}

    def __setstate__(self, state: Any):
        self.buf = state['buf']