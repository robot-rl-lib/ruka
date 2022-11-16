import copy
import datetime
import json
import numpy as np
import os
import pickle
import re
import subprocess
import tempfile

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from ruka.util.compression import img2jpg
from ruka.util.json import JSONSerializable
from typing import Any, Collection, Dict, Iterator, Optional, Tuple

from ..logger import Logger, LogReader, DataType, FPSParams, VideoInfo


# ------------------------------------------------------------------- Logger --


class RukaLogger(Logger):
    def __init__(self, local_path: str) -> None:
        local_path = os.path.abspath(os.path.normpath(local_path))
        if os.path.exists(local_path):
            raise ValueError(f'Local path {local_path} already exists')
        os.makedirs(local_path, exist_ok=True)

        self._local_path = local_path
        self._step_no = 0
        self._datatype: Dict[str, DataType] = {}
        self._data: Dict[str, Dict[int, Any]] = {}  # key => step => data
        self._video_fps: Optional[FPSParams] = None

    def add_scalar(self, key: str, value: float, step_no: int = -1):
        step_no = self._get_step_no(step_no)
        self._set_datatype(key, DataType.SCALAR)
        self._data[key][step_no] = value

    def add_text(self, key: str, value: str, step_no: int = -1):
        step_no = self._get_step_no(step_no)
        self._set_datatype(key, DataType.TEXT)
        self._data[key][step_no] = value

    def add_video_frame(self, key: str, value: ArrayLike, step_no: int = -1):
        step_no = self._get_step_no(step_no)
        ref_step_no = self._set_datatype(key, DataType.VIDEO_FRAME)

        # Check dtype.
        if isinstance(value, np.ndarray) and value.dtype != np.uint8:
            raise ValueError(f'value has dtype {value.dtype}, expected uint8')
        value = np.array(value, dtype=np.uint8)

        # Check dimensions.
        if len(value.shape) != 3 or value.shape[2] not in [1, 3]:
            raise ValueError(
                f'value has shape {value.shape}, expected [H, W, 1] or [H, W, 3]')
        if ref_step_no != -1:
            ref_shape = self._data[key][ref_step_no].shape
            if value.shape != ref_shape:
                raise ValueError(
                    f'value has shape {value.shape}, expected {ref_shape}')

        # Save as JPG.
        self._data[key][step_no] = _VideoFrameInfo(img2jpg(value), value.shape)

    def add_data(self, key: str, value: Any, step_no: int = -1):
        """
        Add data that is to be serialized (pickled) without any changes. Later
        it could be restored exactly the same way from the archive.
        """
        step_no = self._get_step_no(step_no)
        self._set_datatype(key, DataType.DATA)
        self._data[key][step_no] = copy.deepcopy(value)

    def add_metadata(self, key: str, value: JSONSerializable):
        """
        Add some metadata.
        """
        self._set_datatype(key, DataType.METADATA)
        self._data[key][0] = copy.deepcopy(value)

    def set_video_fps(self, video_fps: 'FPSParams'):
        self._video_fps = video_fps

    def step(self, step_no: int = -1):
        if step_no == -1:
            self._step_no += 1
        else:
            if step_no < 0:
                raise ValueError(f'invalid step_no: {step_no}')
            self._step_no = step_no

    def flush(self):
        pass

    def close(self):
        self._close()

    def __enter__(self) -> 'Logger':
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def _get_step_no(self, step_no: int):
        if step_no < 0:
            return self._step_no
        return step_no

    def _set_datatype(self, key: str, datatype: DataType) -> int:
        """
        Doing this the first time for a key?
            Yes => return -1
            No =>
                Datatype matches?
                    Yes => return some step_no to compare against
                    No => raise exception

        Also, validate key.
        """
        parts = key.split('/')
        if '' in parts or '.' in parts or '..' in parts:
            raise KeyError(f'invalid key: {key}')

        if key not in self._datatype:
            self._datatype[key] = datatype
            self._data[key] = {}
            return -1

        if self._datatype[key] != datatype:
            raise ValueError(
                f'key {key} already exists with datatype {self._datatype[key]}')

        some_step_no = next(iter(self._data[key].keys()))
        return some_step_no

    def _close(self):
        """
        system.json
        scalar/{key}.json
        text/{key}.json
        video/{key}.mp4
              {key}.json
        data/{key}.pickle
        metadata.json
        """
        min_step_no = -1
        max_step_no = -1
        num_steps = 0
        if self._data:
            min_step_no = min(min(d.keys()) for d in self._data.values())
            max_step_no = max(max(d.keys()) for d in self._data.values())
            num_steps = max_step_no - min_step_no

        # system.json
        with open(f'{self._local_path}/system.json', 'wt') as f:
            f.write(json.dumps({
                'date': datetime.datetime.now().strftime("%B %d, %Y %I:%M%p"),
                'min_step_no': min_step_no,
                'max_step_no': max_step_no,
                'format': 'ruka_logger',
                'version': 1,
            }))

        # scalar/{key}.json
        for key, value in self._data.items():
            if self._datatype[key] != DataType.SCALAR:
                continue
            path = f'{self._local_path}/scalar/{key}.json'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wt') as f:
                f.write(json.dumps(value))

        # text/{key}.json
        for key, value in self._data.items():
            if self._datatype[key] != DataType.TEXT:
                continue
            path = f'{self._local_path}/text/{key}.json'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wt') as f:
                f.write(json.dumps(value))

        # video/{key}.mp4
        # video/{key}.json
        for key, value in self._data.items():
            # - Skip non-video.
            if self._datatype[key] != DataType.VIDEO_FRAME:
                continue

            # - Paths.
            path_video = f'{self._local_path}/video/{key}.mp4'
            path_meta = f'{self._local_path}/video/{key}.json'
            os.makedirs(os.path.dirname(path_video), exist_ok=True)

            # - Meta.
            shape = next(iter(value.values())).shape
            if self._video_fps is None:
                raise RuntimeError('set_video_fps() was not called')
            fps = self._video_fps.get_fps(num_steps)
            with open(path_meta, 'wt') as f:
                f.write(json.dumps({
                    'height': shape[0],
                    'width': shape[1],
                    'channels': shape[2],
                    'fps': fps
                }))

            # - Video.
            with tempfile.TemporaryDirectory() as tmpdir:
                ndigits = len(str(max_step_no))
                frame = _VideoFrameInfo(img2jpg(np.zeros(shape)), shape)
                for step_no in range(min_step_no, max_step_no + 1):
                    frame = value.get(step_no, frame)
                    with open(f'{tmpdir}/{step_no:0{ndigits}}.jpg', 'wb') as f:
                        f.write(frame.buf_jpg.tobytes())
                subprocess.check_call(
                    [
                        'ffmpeg',
                        '-framerate', str(fps),
                        '-pattern_type', 'glob',
                        '-i', '*.jpg',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        path_video
                    ],
                    cwd=tmpdir
                )

        # data/{key}.pickle
        for key, value in self._data.items():
            if self._datatype[key] != DataType.DATA:
                continue
            path = f'{self._local_path}/data/{key}.pickle'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(pickle.dumps(value))

        # metadata.json
        metadata = {}
        for key, value in self._data.items():
            if self._datatype[key] != DataType.METADATA:
                continue
            metadata[key] = value[0]
        path = f'{self._local_path}/metadata.json'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wt') as f:
            f.write(json.dumps(metadata))


@dataclass
class _VideoFrameInfo:
    buf_jpg: NDArray[np.uint8]
    shape: Tuple[int, ...]


# ---------------------------------------------------------------- LogReader --


class RukaLogReader(LogReader):
    def __init__(self, local_path: str):
        # Path.
        local_path = os.path.normpath(os.path.abspath(local_path))
        if not os.path.isdir(local_path):
            raise ValueError(f'{local_path} does not exist or is not a dir')
        self._local_path = local_path

        # System.
        with open(f'{local_path}/system.json', 'rt') as f:
            self._system = json.load(f)

        # Datatypes.
        self._datatype: Dict[str, DataType] = {}
        for key in _ls_recursive(f'{local_path}/scalar'):
            key = _strip_suffix(key, '.json')
            self._datatype[key] = DataType.SCALAR
        for key in _ls_recursive(f'{local_path}/text'):
            key = _strip_suffix(key, '.json')
            self._datatype[key] = DataType.TEXT
        for key in _ls_recursive(f'{local_path}/video'):
            if not key.endswith('.json'):
                continue
            key = _strip_suffix(key, '.json')
            self._datatype[key] = DataType.VIDEO_FRAME
        for key in _ls_recursive(f'{local_path}/data'):
            key = _strip_suffix(key, '.pickle')
            self._datatype[key] = DataType.DATA

        # Metadata.
        with open(f'{local_path}/metadata.json', 'rt') as f:
            self._metadata = json.load(f)
        for key in self._metadata:
            self._datatype[key] = DataType.METADATA

    def get_keys(self) -> Collection[str]:
        return self._datatype.keys()

    def get_datatype(self, key: str) -> DataType:
        return self._datatype[key]

    def get_value(self, key: str) -> Any:
        """
        For 'scalar', return Dict[int, float]
        For 'text', return Dict[int, str]
        For 'video_frame', return ...
        For 'data', return Dict[int, Any]
        For 'metadata', return JSONSerializable

        Raise KeyError if key doesn't exist.
        """
        datatype = self._datatype[key]
        if datatype == DataType.SCALAR:
            with open(f'{self._local_path}/scalar/{key}.json', 'rt') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        elif datatype == DataType.TEXT:
            with open(f'{self._local_path}/text/{key}.json', 'rt') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        elif datatype == DataType.VIDEO_FRAME:
            return f'{self._local_path}/video/{key}.mp4'
        elif datatype == DataType.DATA:
            with open(f'{self._local_path}/data/{key}.pickle', 'rb') as f:
                return pickle.load(f)
        elif datatype == DataType.METADATA:
            return self._metadata[key]
        else:
            assert 0

    def get_max_step_no(self) -> int:
        return self._system['max_step_no']

    def get_video_info(self, key: str) -> VideoInfo:
        if self._datatype[key] != DataType.VIDEO_FRAME:
            raise ValueError(f'key {key} is not a video')
        with open(f'{self._local_path}/video/{key}.json', 'rt') as f:
            info = json.load(f)
        return VideoInfo(
            height=info['height'],
            width=info['width'],
            channels=info['channels'],
            fps=info['fps']
        )


def _ls_recursive(top: str) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(top):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            yield path[len(top) + 1:]


def _strip_suffix(s: str, suffix: str) -> str:
    assert s.endswith(suffix)
    return s[:-len(suffix)]



