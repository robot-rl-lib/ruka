import numpy as np

from ruka.util.x3d import compose_matrix_world


def test_compose_matrix_world():
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    def mul(mat, vec):
        vec = np.array(list(vec) + [1])
        vec = mat @ vec
        return list(vec)[:3]

    # Roll.
    M = compose_matrix_world(angles=[90, 0, 0])
    assert np.allclose(mul(M, x), x)
    assert np.allclose(mul(M, y), z)
    assert np.allclose(mul(M, z), -y)

    # Pitch.
    M = compose_matrix_world(angles=[0, 90, 0])
    assert np.allclose(mul(M, x), z)
    assert np.allclose(mul(M, y), y)
    assert np.allclose(mul(M, z), -x)

    # Yaw.
    M = compose_matrix_world(angles=[0, 0, 90])
    assert np.allclose(mul(M, x), -y)
    assert np.allclose(mul(M, y), x)
    assert np.allclose(mul(M, z), z)
