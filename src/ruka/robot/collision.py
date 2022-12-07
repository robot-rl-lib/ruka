import abc
import copy

import open3d as o3d
import numpy as np

from ruka.util.x3d import Vec3, compose_matrix_world
from .robot import ArmInfo


class CollisionDetector(abc.ABC):
    @abc.abstractmethod
    def test_vel(self, vel: Vec3) -> Vec3:
        """
        Tests input velocity for collisions, corrects the velocity if any.
        """
        pass


class SimpleCollisionDetector(CollisionDetector):
    """
    Simple collision detection algorithm

    1. Check that speed is nonzero
    2. Apply forward kinematics to robot mesh
    3. Assuming linear movement min collision distance in velocity direction
    4. Decrease the speed to avoid collision after dt seconds
    """

    def __init__(self, arm_info: ArmInfo, mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, dt: float):
        self._arm_info = arm_info
        self._mesh = mesh
        self._pcd = pcd
        self._dt = dt

    def test_vel(self, vel: Vec3) -> Vec3:
        target_vel = np.array(vel, dtype=np.float32)
        target_speed = np.linalg.norm(target_vel)
        if target_speed < 1:  # Check if speed is nonzero
            return [0] * 3
        mesh = copy.deepcopy(self._mesh).transform(compose_matrix_world(pos=self._arm_info.pos, angles=self._arm_info.angles))
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        tpcd = o3d.t.geometry.PointCloud.from_legacy(self._pcd)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        positions = tpcd.point.positions
        velocities = o3d.core.Tensor.from_numpy(np.tile(-target_vel, (positions.shape[0], 1))).to(o3d.core.Dtype.Float32)
        rays = o3d.core.concatenate((positions, velocities), 1)
        raycast = scene.cast_rays(rays)
        collision_distance = raycast['t_hit'].numpy().min()
        if not np.isfinite(collision_distance):
            return vel
        return list(target_vel / target_speed * collision_distance / self._dt)
