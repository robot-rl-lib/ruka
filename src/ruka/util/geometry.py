import numpy as np

from numpy.typing import NDArray


def camera_project_points(
    intrinsics: NDArray[np.float],
    points: NDArray[np.float]):
    '''
    Project 3D points to camera image

    Args:
        intrinsics (np.array): (3, 3) intrinsics matrix
        points (np.array): (n_points, 3) or (3) array with XYZ points to project

    Returns:
        points (np.array): (n_points, 2) or (2) XY array with projected points
    '''

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    ppx, ppy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = points[..., 0].copy(), points[..., 1].copy(), points[..., 2]

    x = ((x / (z + 1e-8)) * fx) + ppx
    y = ((y / (z + 1e-8)) * fy) + ppy

    return np.ascontiguousarray(np.array([x, y]).T)


def camera_deproject_points(
    intrinsics: NDArray[np.float],
    points: NDArray[np.float],
    depths: NDArray[np.float]):
    '''
    Get 3D points from 2D points and depths

    Args:
        intrinsics (np.array): (3, 3) intrinsics matrix
        points (np.array): (n_points, 2) or (2) array with XY points on image
        depths (np.array): (n_points) array with depths corresponding to points

    Returns:
        points (np.array): (n_points, 3) or (3) XYZ array with deprojected points
    '''

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    ppx, ppy = intrinsics[0, 2], intrinsics[1, 2]

    x, y = points[..., 0].copy(), points[..., 1].copy()

    x = ((x - ppx) / fx) * depths
    y = ((y - ppy) / fy) * depths

    return np.ascontiguousarray(np.array([x, y, depths]).T)
