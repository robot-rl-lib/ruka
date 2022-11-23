import numpy as np
import ruka.util.nested_dict as nd

from numpy.typing import NDArray
from typing import Any

from .logger import Logger


def visualize_nested_dict(
        logger: Logger,
        data: nd.NestedDict,
        prefix: str,
        sep: str = '/',
        auto: bool = True,
        has_batch_dim: bool = False,
    ):
    """
    Args:
        logger: where to write visualization to
        data: what to visualize
        auto: visualize more stuff using heuristics (see below)
        has_batch_dim: whether to expect an extra leading dimension, a shape of
            which must be equal to 1.

    Supported types:
        float, int, bool  - logs as add_scalar
        str               - logs as add_text
        RGB               - logs as add_video_frame
        Grayscale         - logs as add_video_frame
        Depth             - logs as Grayscale

    Extra suppported types (auto == True):
        1d ndarray        - logs as scalar x N (if N <= 10)
        2d ndarray        - logs as Grayscale, assume [H, W]
        3d ndarray
            1ch           - logs as Grayscale, assume [H, W]
            2ch           - logs as Grayscale x 2, assume [H, W, 2]
            3ch           - logs as RGB, assume [H, W, 3]
            Nch           - logs as Grayscale x N, assume [H, W, N]

        DType treatment when converting to Grayscale/RGB:
            - If dtype == uint8, assume values are in range [0, 255];
            - If dtype == (any float), assume values are in range [0, 1].
              Will be converted to [0, 255] uint8.

    """
    for k, v in nd.items(data, prefix, sep):
        # Remove batch dim.
        if has_batch_dim and isinstance(v, np.ndarray):
            assert v.shape[0] == 1, v.shape
            v = v[0]

        # Scalars.
        if isinstance(v, (float, int, bool)):
            logger.add_scalar(k, v)
        if isinstance(v, str):
            logger.add_text(k, v)

        # Auto.
        if auto and isinstance(v, np.ndarray):
            # - 1D.
            if len(v.shape) == 1 and v.shape[0] <= 10:
                for i, item in enumerate(v.tolist()):
                    logger.add_scalar(f'{k}[{i}]', item)

            # - 2D.
            if len(v.shape) == 2:
                logger.add_video_frame(k, _convert_img(v[:, :, None]))

            # - 3D.
            if len(v.shape) == 3:
                if v.shape[2] in [1, 3]:
                    logger.add_video_frame(k, _convert_img(v))
                else:
                    for i in range(v.shape[2]):
                        logger.add_video_frame(f'{k}[{i}]', _convert_img(v))


def _convert_img(img: NDArray) -> NDArray:
    """
    Args:
        img: [H, W, C]
            If dtype == uint8     assume value in range [0, 255]
            if dtype == (any float) assume value in range [0, 1] -> will be converted to [0, 255] uint8
    """
    assert len(img.shape) == 3

    if img.dtype == np.float32 or img.dtype == np.float16 or img.dtype == np.float64:
        # 0..1 range to 0..255
        img = np.clip(img * 255, 0, 255)

    img = img.astype(np.uint8)
    return img