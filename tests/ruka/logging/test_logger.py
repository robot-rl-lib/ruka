import numpy as np
import os
import pytest
import tempfile

from ruka.logging.logger import \
    create_ruka_logger, create_ruka_log_reader, FPSParams, VideoInfo, DataType


class CustomClass:
    def __init__(self, i: int):
        self.i = i

    def __eq__(self, other):
        return isinstance(other, CustomClass) and self.i == other.i


def test_ruka_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write.
        with create_ruka_logger(f'{tmpdir}/logdir') as l:
            l.step(1000)
            for i in range(200):
                l.add_scalar('s', np.sin(i))
                l.add_text('t', f'cos={np.cos(i)}')
                frame = np.ones((64, 128, 1), dtype=np.uint8) * i
                l.add_video_frame('v/1', frame)
                l.add_video_frame('v/3', np.dstack([frame, 255-frame, frame]))
                l.add_data('d', ('this', 'is', 'a', {'tuple'}, CustomClass(i)))
                l.step()
            l.add_metadata('m', {'1': 2, '3': 4})
            l.set_video_fps(FPSParams(dt=1/20))

        # Keys.
        r = create_ruka_log_reader(f'{tmpdir}/logdir')
        assert sorted(r.get_keys()) == ['d', 'm', 's', 't', 'v/1', 'v/3']

        # Unknown key.
        with pytest.raises(KeyError):
            r.get_datatype('foo/bar')

        # Scalar.
        assert r.get_datatype('s') == DataType.SCALAR
        assert r.get_value('s') == {1000+i: np.sin(i) for i in range(200)}

        # Text.
        assert r.get_datatype('t') == DataType.TEXT
        assert r.get_value('t') == {1000+i: f'cos={np.cos(i)}' for i in range(200)}

        # Video (monochrome).
        assert r.get_datatype('v/1') == DataType.VIDEO
        assert os.path.isfile(r.get_value('v/1'))
        assert r.get_video_info('v/1') == VideoInfo(
            height=64, width=128, channels=1, fps=20)

        # Video (RGB).
        assert r.get_datatype('v/3') == DataType.VIDEO
        assert os.path.isfile(r.get_value('v/3'))
        assert r.get_video_info('v/1') == VideoInfo(
            height=64, width=128, channels=1, fps=20)

        # Data.
        assert r.get_datatype('d') == DataType.DATA
        assert r.get_value('d') == {
            1000+i: ('this', 'is', 'a', {'tuple'}, CustomClass(i))
            for i in range(200)
        }

        # Metadata.
        assert r.get_datatype('m') == DataType.METADATA
        assert r.get_value('m') == {'1': 2, '3': 4}

        # Other data.
        assert r.get_max_step_no() == 1199