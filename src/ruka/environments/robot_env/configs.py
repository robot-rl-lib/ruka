from enum import Enum
import dataclasses
from typing import Tuple, List, Optional

class SceneType(Enum):
    ON_TABLE = 1
    ON_FLOOR = 2

class DataSet(Enum):
    WOODEN_BLOCKS = 1
    RANDOM_URDFS= 2

 
@dataclasses.dataclass
class BaseDataclass:
    def __getitem__(self, item):
        return getattr(self, item)

@dataclasses.dataclass
class EnvironmentConfig(BaseDataclass):
    @dataclasses.dataclass
    class RobotConfig(BaseDataclass):
        model_path: str
        max_translation: float
        max_yaw_rotation: float
        max_force: float 
        discrete: bool
        step_size: float
        yaw_step: float
        num_actions_pad: int
        include_robot_height: bool
    
    @dataclasses.dataclass
    class SceneConfig(BaseDataclass):
        scene_type: SceneType
        data_set: DataSet
        extent: float
        max_objects: int
        min_objects: int

    @dataclasses.dataclass
    class SimulationConfig(BaseDataclass):
        real_time: bool
        visualize: bool

    @dataclasses.dataclass
    class SensorConfig(BaseDataclass):
        @dataclasses.dataclass
        class CameraInfoConfig(BaseDataclass):
            height: int
            width: int
            K: Tuple[float, float, float, float, float, float, float, float, float]
            near: float
            far: float

        @dataclasses.dataclass
        class CameraTransformConfig(BaseDataclass):
            translation: Tuple[float, float, float]
            rotation: Tuple[float, float, float, float]

        @dataclasses.dataclass
        class CameraRandomizationConfig(BaseDataclass):
            focal_length: float
            optical_center: float
            translation: float
            rotation: float

        camera_info: CameraInfoConfig
        transform: CameraTransformConfig
        visualize: bool
        randomize: CameraRandomizationConfig
            
    @dataclasses.dataclass
    class RewardConfig(BaseDataclass):
        custom: bool
        shaped: bool
        terminal_reward: float
        lift_success: float
        grasp_reward: float
        delta_z_scale: float
        time_penalty: float
        table_clearing: bool
            
    @dataclasses.dataclass
    class CurriculumConfig(BaseDataclass):
        init_lambda: float
        n_steps: int
        success_threshold: float
        window_size: int
        extent: Tuple[float, float]
        robot_height: Tuple[float, float]
        lift_dist: Tuple[float, float]
        max_objects: Tuple[int, int]
        min_objects: Tuple[int, int]
        workspace: Optional[Tuple[float, float]]
        work_height: Optional[Tuple[float, float]]
            
    skip_empty_initial_state: bool
    simplified: bool
    depth_observation: bool
    full_observation: bool
    timefeature: bool
    time_horizon: int
    only_ball: bool
    robot: RobotConfig
    scene: SceneConfig
    simulation: SimulationConfig
    sensor: SensorConfig
    reward: RewardConfig
    curriculum: CurriculumConfig
        
