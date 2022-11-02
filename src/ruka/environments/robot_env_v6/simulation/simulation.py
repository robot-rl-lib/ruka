from enum import Enum
import pybullet as p
import time
import gym
import os

from gym.utils import seeding
from .model import Model
from . import scene
from pybullet_utils import bullet_client

class World(gym.Env):
    def __init__(self, config, validate):
        """Initialize a new simulated world."""
        self._config = config
        self._scene = scene.OnTable(self, config.scene, validate)
        self.real_time = config.real_time

        self.sim_time = 0.
        self._time_step = 1. / 240.

        self.physics_client = bullet_client.BulletClient(p.GUI if os.environ.get("RUKA_GUI", False) else p.DIRECT)

        self.models = []
        
    def run(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.step_sim()

    def add_model(self, path, start_pos, start_orn, scaling=1.):
        model = Model(self.physics_client, max_speed=self._config.robot.max_speed)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model

    def step_sim(self):
        """Advance the simulation by one step."""
        self.physics_client.stepSimulation()
        self.sim_time += self._time_step
        if self.real_time: time.sleep(self._time_step)

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

    def remove_model(self, model_id):
        self.physics_client.removeBody(model_id)
        self.models[model_id] = False