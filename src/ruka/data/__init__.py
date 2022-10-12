import os
import pybullet_data


HERE = os.path.dirname(os.path.abspath(__file__))


def get_data(path: str) -> str:
    path = os.path.normpath(path)
    assert not os.path.isabs(path)
    assert not path.startswith('..')
    return os.path.join(HERE, path)


def get_pybullet_data(path: str) -> str:
    path = os.path.normpath(path)
    assert not os.path.isabs(path)
    assert not path.startswith('..')
    return os.path.join(pybullet_data.getDataPath(), path)