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
    pos = m[:3, 3]
    r, p, y = np.rad2deg(transformations.euler_from_matrix(m, 'sxyz'))
    return pos, np.array([r, -p, -y])

def decompose_matrix_tool(m) -> Tuple[Vec3, Vec3]:
    m = np.array(m)
    pos = m[:3, 3]
    angles = np.rad2deg(transformations.euler_from_matrix(m, 'szxy'))
    return pos, angles


def chain(*transformations):
    m = np.identity(4)
    for t in transformations:
        m = m @ t
    return m

# Do a rotation of angles in Conventional World
def conventional_rotation(angles: Vec3, axis: int, angle: float) -> Vec3:
    tool_matrix = compose_matrix_world(angles=angles)
    rotate_matrix = transformations.euler_matrix(
        np.deg2rad(angle) if axis == 0 else 0,
        np.deg2rad(angle) if axis == 1 else 0,
        np.deg2rad(angle) if axis == 2 else 0, 'sxyz')
    conventional_tool_matrix = chain(tool_matrix, rotate_matrix)

    _, angles = decompose_matrix_world(conventional_tool_matrix)
    return angles


def tool_to_world(tool_xyz, tool_rpy, robot_rpy) -> Tuple[Vec3, Vec3]:
    world_to_tool = compose_matrix_world(angles=robot_rpy)
    tool = compose_matrix_tool(pos=tool_xyz, angles=tool_rpy)
    tool_to_world = np.linalg.inv(world_to_tool)
    world = chain(world_to_tool, tool, tool_to_world)

    return decompose_matrix_world(world)
