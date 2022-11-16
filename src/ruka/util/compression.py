import cv2
import numpy as np

from dataclasses import dataclass
from typing import Any, Union
from numpy.typing import NDArray

from .array_semantics import RGB, Grayscale, Depth


# --------------------------------------------------------------------- JPEG --


def img2jpg(img_hwc: NDArray[np.uint8], quality: int = 95) -> NDArray[np.uint8]:
    """
    Encode input image RGB or GRAY with jpeg codec.
    """
    assert 0 < quality <= 100, quality
    assert len(img_hwc.shape) == 3 and img_hwc.shape[2] in [1,3], img_hwc.shape

    if img_hwc.shape[2] == 3:
        # RGB -> BGR
        img_hwc = img_hwc[:, :, [2, 1, 0]]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    buf = cv2.imencode('.jpg', img_hwc, encode_param)[1]
    return buf


def jpg2img(buf: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Decode img RGB or GRAY from jpg buffer.
    Return HWC image.
    """
    img = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
    if len(img.shape) == 2:
        # hw -> hwc for grayscale
        img = img[:,:, None]
    if img.shape[2] == 3:
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]
    return img


# ---------------------------------------------------------------------- PNG --


def depth2png(
        depth_hwc: NDArray[Union[np.uint16, np.float32, np.float16]],
        quality: int = 1
    ) -> NDArray[np.uint8]:
    """
    Encode 16bit depth image with PNG codec.

    quality: it can be the compression level from 0 to 9.
        A higher value means a smaller size and longer compression time
    """
    MAX_DEPTH = 65535
    assert 0 < quality <= 9, quality
    assert len(depth_hwc.shape) == 3 and depth_hwc.shape[2] == 1, depth_hwc.shape
    assert depth_hwc.dtype in [np.uint16, np.float32, np.float16], depth_hwc.dtype

    depth_clip = np.clip(depth_hwc, 0, MAX_DEPTH).astype(np.uint16)
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), quality]
    buf = cv2.imencode('.png', depth_clip, encode_param)[1]
    return buf


def png2depth(buf: NDArray[np.uint8]) -> NDArray[np.uint16]:
    """
    Decode 16bit depth image from PNG.
    """
    depth = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH)
    depth = depth[:,:, None]
    return depth


# ---------------------------------------------------------------------- RGB --


@dataclass
class CompressedRGB:
    buf_jpg: NDArray[np.uint8]

    def __getstate__(self) -> Any:
        return {'buf_jpg': self.buf_jpg}

    def __setstate__(self, state: Any):
        self.buf_jpg = state['buf_jpg']


def compress_rgb_maybe(x: Any) -> Any:
    if isinstance(x, RGB):
        return CompressedRGB(img2jpg(x.buf))
    return x


def decompress_rgb_maybe(x: Any) -> Any:
    if isinstance(x, CompressedRGB):
        return RGB(jpg2img(x.buf_jpg))
    return x


# ---------------------------------------------------------------- Grayscale --


@dataclass
class CompressedGrayscale:
    buf_jpg: NDArray[np.uint8]

    def __getstate__(self) -> Any:
        return {'buf_jpg': self.buf_jpg}

    def __setstate__(self, state: Any):
        self.buf_jpg = state['buf_jpg']


def compress_grayscale_maybe(x: Any) -> Any:
    if isinstance(x, Grayscale):
        return CompressedGrayscale(img2jpg(x.buf))
    return x


def decompress_grayscale_maybe(x: Any) -> Any:
    if isinstance(x, CompressedGrayscale):
        return Grayscale(jpg2img(x.buf_jpg))
    return x


# -------------------------------------------------------------------- Depth --


@dataclass
class CompressedDepth:
    buf_png: NDArray[np.uint8]

    def __getstate__(self) -> Any:
        return {'buf_png': self.buf_png}

    def __setstate__(self, state: Any):
        self.buf_png = state['buf_png']


def compress_depth_maybe(x: Any) -> Any:
    if isinstance(x, Depth):
        return CompressedDepth(depth2png(x.buf))
    return x


def decompress_depth_maybe(x: Any) -> Any:
    if isinstance(x, CompressedDepth):
        return Depth(png2depth(x.buf_png))
    return x


# --------------------------------------------------------------- Everything --


def compress_everything_maybe(x: Any) -> Any:
    x = compress_rgb_maybe(x)
    x = compress_grayscale_maybe(x)
    x = compress_depth_maybe(x)
    return x


def decompress_everything_maybe(x: Any) -> Any:
    x = decompress_rgb_maybe(x)
    x = decompress_grayscale_maybe(x)
    x = decompress_depth_maybe(x)
    return x