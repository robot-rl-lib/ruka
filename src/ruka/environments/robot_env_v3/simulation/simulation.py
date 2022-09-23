from enum import Enum
import pybullet as p
import time
import gym

from gym.utils import seeding
from .model import Model
from . import scene
from pybullet_utils import bullet_client

class World(gym.Env):
    def __init__(self, config, validate):
        """Initialize a new simulated world."""

        self._scene = scene.OnTable(self, config.scene, validate)

        self.sim_time = 0.
        self._time_step = 1. / 240.

        if config.sensor.visualize:
            self.physics_client = bullet_client.BulletClient(p.GUI)
        else:
            self.physics_client = bullet_client.BulletClient(p.DIRECT)

        self.models = []

    def run(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.step_sim()

    def add_model(self, path, start_pos, start_orn, scaling=1.):
        model = Model(self.physics_client)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model

    def step_sim(self):
        """Advance the simulation by one step."""
        self.physics_client.stepSimulation()
        self.sim_time += self._time_step

    def reset_sim(self):
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=150,
            enableConeFriction=1)
        self.physics_client.setGravity(0., 0., -9.81)    
        self.models = []
        self.sim_time = 0.

        self._scene.reset()

    def close(self):
        self.physics_client.disconnect()
    
    def find_highest(self):
        highest = -float('inf')
        model_id = -1
        for obj in self.models[1:len(self.models)-1]:
            if obj:
                pos, _ = obj.getBase()
                if pos[2] > highest: 
                    highest = pos[2]
                    model_id = obj.model_id
        return model_id

    def find_higher(self, lift_dist):
        #TODO make robust
        #FIXME not working with small lift distance

        # For OnTable scene.
        thres_height = self.models[2].getBase()[0][2]

        grabbed_objs = []
        for obj in self.models[1:len(self.models)-1]:
            if obj:
                pos, _ = obj.getBase()
                # print("height", pos[2])
                # print("threshold", thres_height + lift_dist)
                if pos[2] > (thres_height + lift_dist):
                    grabbed_objs.append(obj.model_id)
        return grabbed_objs

    def remove_model(self, model_id):
        self.physics_client.removeBody(model_id)
        self.models[model_id] = False

    def remove_models(self, model_ids):
        for model_id in model_ids:
            self.physics_client.removeBody(model_id)
            self.models[model_id] = False

    def get_num_body(self):
        self.physics_client.syncBodyInfo()
        # For OnTable scene.
        return self.physics_client.getNumBodies() - 2