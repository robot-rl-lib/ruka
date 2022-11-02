import numpy as np
import transformations

from typing import List, Tuple, Optional


Vec3 = List[float]
Quat = List[float]


def compose_matrix_world(pos: Optional[Vec3] = None, angles: Optional[Vec3] = None):
    m = np.identity(4)

    if angles is not None:
        r, p, y = np.deg2rad(angles)
        m = transformations.euler_matrix(r, -p, -y, 'sxyz')

    if pos is not None:
        m[:3, 3] = np.array(pos)

    return m


def compose_matrix_tool(pos: Optional[Vec3] = None, angles: Optional[Vec3] = None):
    m = np.identity(4)

    if angles is not None:
        r, p, y = np.deg2rad(angles)
        m = transformations.euler_matrix(r, p, y, 'szxy')

    if pos is not None:
        m[:3, 3] = np.array(pos)

    return m


def decompose_matrix_world(m) -> Tuple[Vec3, Vec3]:
    m = np.array(m)
    pos = M[:3, 3]
    r, p, y = np.rad2deg(transformations.euler_from_matrix(M, 'sxyz'))
    return pos, np.array([r, -p, -y])


def decompose_matrix_tool(m) -> Tuple[Vec3, Vec3]:
    M = np.array(m)
    pos = M[:3, 3]
    angles = np.rad2deg(transformations.euler_from_matrix(M, 'szxy'))
    return pos, angles


def chain(*transformations):
    m = np.identity(4)
    for t in transformations:
        m = m @ t
    return m

