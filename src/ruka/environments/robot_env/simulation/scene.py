import os
import pybullet as p
import numpy as np
import sys
import pybullet_data
from .base_scene import BaseScene
from manipulation_main.common import transform_utils
from ruka.data import get_data

class OnTable(BaseScene):
    """Tabletop settings with geometrically different objects."""
    def __init__(self, world, config, rng, test=False, validate=False, only_ball=False):
        super().__init__(world, config, rng, test, validate,)
        self.block_models = []
        self.only_ball=only_ball

    def reset(self):
        self.block_models = []
        self.table_path = 'table/table.urdf'
        self.plane_path = 'plane.urdf'
        self._model_path = pybullet_data.getDataPath()
        tray_path = os.path.join(self._model_path, 'tray/tray.urdf')
        # plane_urdf = os.path.join("models", self.plane_path)
        plane_urdf = get_data(os.path.join("models", self.plane_path))
        # table_urdf = os.path.join("models", self.table_path)
        table_urdf = get_data(os.path.join("models", self.table_path))
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        self._world.add_model(tray_path, [0, 0.075, -0.19],
                              [0.0, 0.0, 1.0, 0.0], scaling=1.2)

        if self.only_ball:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            block_model = self._world.add_model(os.path.join(self._model_path, 'sphere2red.urdf'), position, orientation, scaling=.07)
            self.block_models.append(block_model)
            return
            
        # Sample random objects
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)

        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            block_model = self._world.add_model(path, position, orientation, scaling=scale)
            # self.block_ids.append(block_model.model_id)
            self.block_models.append(block_model)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

class OnFloor(BaseScene):
    """Curriculum paper setup."""
    def reset(self):
        self.plane_path = 'plane.urdf'
        plane_urdf = os.path.join("models", self.plane_path)
        self._world.add_model(plane_urdf, [0., 0., -0.196], [0., 0., 0., 1.])
        # Sample random objects
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)
        

        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)


class OnTableWithBox(BaseScene):
    """Google Q-opt setup."""
    pass

