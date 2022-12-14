import collections
import functools
import os

import numpy as np
import pybullet

from . import robot


class WorkspaceCurriculum:
    """Adaptively grow the workspace to increase the diversity of traing samples."""

    def __init__(self, config, robot, evaluate):
        self._robot = robot
        self._scene = robot._scene
        self._reward_fn = robot._reward_fn

        self._n_steps = config['n_steps']
        self._success_threshold = config['success_threshold']
        self._window_size = 100

        self._extent_range = config['extent']
        self._robot_height_range = config['robot_height']
        self._max_objects_range = config['max_objects']
        self._min_objects_range = config.min_objects
        self._lift_dist_range = config.lift_dist

        self._history = collections.deque(maxlen=self._window_size)
        self._lambda = config.init_lambda if not evaluate else 1.
        self._update_parameters()

        self._policy_iteration = 1

    def update(self, task):
        """Update history and update parameters if appropriate."""
        self._history.append(task.status == robot.RobotEnv.Status.SUCCESS)
        
        if len(self._history) < self._history.maxlen:
            return
        if np.mean(self._history) > self._success_threshold and self._lambda != 1.:
            # Increase workspace
            self._lambda = min(1., self._lambda + 1. / self._n_steps)
            self._update_parameters()
            self._history.clear()
            print('Increased the step of the curriculum sequence to', self._lambda)

    def _update_parameters(self):
        extent = _convert(self._lambda, self._extent_range)
        height = _convert(self._lambda, self._robot_height_range)
        max_objects = int(
            round(_convert(self._lambda, self._max_objects_range)))
        min_objects = int(
            round(_convert(self._lambda, self._max_objects_range)))

        self._scene.extent = extent
        self._scene.max_objects = max_objects
        self._scene.min_objects = min_objects
        self._robot._initial_height = height

        if self._lift_dist_range is not None:
            lift_dist = _convert(self._lambda, self._lift_dist_range)
            self._reward_fn.lift_dist = lift_dist


def _convert(val, new_range):
    """Convert val from range [0., 1.] to new_range."""
    new_min, new_max = new_range[0], new_range[1]
    return new_min + (new_max - new_min) * val