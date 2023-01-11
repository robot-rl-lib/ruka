import os
import pybullet as p
import numpy as np
import pybullet_data
import numpy.random as npr
import glob
from manipulation_main.common import transform_utils
from ruka.data import get_data, get_pybullet_data
from ruka.environments.robot_env_v4.configs import ObjectDataset


class OnTable:
    """Tabletop settings with geometrically different objects."""
    def __init__(self, world, config, validate=False):
        self._world = world
        self._validate = validate
        self.extent = config.extent
        self.max_objects = config.max_objects
        self.min_objects = config.min_objects
        self.pickable_objects = []

        self.dataset = config.object_dataset

        self._sampling_functions = {
            ObjectDataset.RANDOM_URDFS: self._sample_random_urdfs,
            ObjectDataset.CUSTOM_GAZEBO: self._sample_custom_gazebo,
        }
        self._sample_objects_fn = self._sampling_functions[self.dataset]


    def reset(self):
        tray_path = get_pybullet_data('tray/tray.urdf')
        plane_urdf = get_data('models/plane.urdf', True)
        table_urdf = get_data('models/table/table.urdf', True)
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        self._world.add_model(tray_path, [0, 0.075, -0.19],
                              [0.0, 0.0, 1.0, 0.0], scaling=1.2)

        # Sample random objects
        n_objects = npr.randint(self.min_objects, self.max_objects + 1)
        urdf_paths_and_scales = self._sample_objects_fn(n_objects)

        # Spawn objects
        self.pickable_objects = []
        for path, scale in urdf_paths_and_scales:
            position = np.r_[npr.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(npr.rand(3))
            self.pickable_objects.append(self._world.add_model(path, position, orientation, scaling=scale))
            # self.block_ids.append(block_model.model_id)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

    def _sample_random_urdfs(self, n_objects):
        if self._validate:
            self.object_range = np.arange(700, 850)
        else:
            self.object_range = 700
        selection = npr.choice(self.object_range, size=n_objects)
        paths_and_scales = [(os.path.join(pybullet_data.getDataPath(), 'random_urdfs',
                            '{0:03d}/{0:03d}.urdf'.format(i)), 1)for i in selection]
        return paths_and_scales

    def _sample_custom_gazebo(self, n_objects):
        path = get_data('custom_dataset', True)
        # objects = glob.glob(os.path.join(path, "*", ))
        object_folders = glob.glob(os.path.join(path, "*"))

        objects_and_scales = [
            (os.path.join(folder, "model.sdf"), self.read_scale_from_folder(folder)) \
                for folder in object_folders]
        selection = npr.choice(len(objects_and_scales), size=n_objects)
        paths_and_scales = [objects_and_scales[i] for i in selection]
        return paths_and_scales

    @staticmethod
    def read_scale_from_folder(folder):
        path = os.path.join(folder, "scale.txt")
        if os.path.exists(path):
            with open(path) as f:
                scale = float(f.readline().strip())
        else:
            scale = 1.
        return scale