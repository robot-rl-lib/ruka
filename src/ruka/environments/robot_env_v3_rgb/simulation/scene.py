import os
import pybullet as p
import numpy as np
import pybullet_data
import numpy.random as npr
import sys
from manipulation_main.common import transform_utils
from ruka.data import get_data


class OnTable:
    """Tabletop settings with geometrically different objects."""
    def __init__(self, world, config, validate=False):
        self._world = world
        self._validate = validate
        self.extent = config.extent
        self.max_objects = config.max_objects
        self.min_objects = config.min_objects

    def reset(self):
        tray_path = os.path.join(pybullet_data.getDataPath(), 'tray/tray.urdf')
        plane_urdf = get_data('models/plane.urdf')
        table_urdf = get_data('models/table/table.urdf')
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        self._world.add_model(tray_path, [0, 0.075, -0.19],
                              [0.0, 0.0, 1.0, 0.0], scaling=1.2)
            
        # Sample random objects
        n_objects = npr.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._sample_random_objects(n_objects)

        # Spawn objects
        for path in urdf_paths:
            position = np.r_[npr.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(npr.rand(3))
            block_model = self._world.add_model(path, position, orientation, scaling=scale)
            # self.block_ids.append(block_model.model_id)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

    def _sample_random_objects(self, n_objects):
        if self._validate:
            self.object_range = np.arange(700, 850)
        else: 
            self.object_range = 700
        selection = npr.choice(self.object_range, size=n_objects)
        paths = [os.path.join(pybullet_data.getDataPath(), 'random_urdfs',
                            '{0:03d}/{0:03d}.urdf'.format(i)) for i in selection]
        return paths, 1.