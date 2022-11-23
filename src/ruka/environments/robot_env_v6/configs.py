import enum
import dataclasses
import enum
from typing import Tuple, List


class Observe(enum.Enum):
    DEPTH = 'depth'
    RGB = 'rgb'
    GRAY = 'gray'
    TARGET_SEGMENTATION = 'target_segmentation'
    ROBOT_POS = 'robot_pos'
    GRIPPER = 'gripper'
    SENSOR_PAD = 'sensor_pad'
    HEIGHT = 'height'
    TIMESTEP = 'timestep'
    TRANSITION_TIME = 'transition_time'
    GOAL = 'goal'

@dataclasses.dataclass
class BaseDataclass:
    def __getitem__(self, item):
        return getattr(self, item)


class ObjectDataset(enum.Enum):
    RANDOM_URDFS = 'random_urdfs'
    CUSTOM_GAZEBO = 'custom_gazebo'

@dataclasses.dataclass
class EnvironmentConfig(BaseDataclass):

    @dataclasses.dataclass
    class OnGripperCameraConfig(BaseDataclass):
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

        @dataclasses.dataclass
        class CameraRandomizationConfig(BaseDataclass):
            focal_length: float
            optical_center: float
            translation: float
            rotation: float

        camera_info: CameraInfoConfig
        transform: CameraTransformConfig
        randomize: CameraRandomizationConfig

    @dataclasses.dataclass
    class RobotConfig(BaseDataclass):
        model_path: str
        action_wait: float
        gripper_wait: float
        max_speed: float
        max_translation: float
        max_yaw_rotation: float
        max_force: float

    @dataclasses.dataclass
    class SceneConfig(BaseDataclass):
        extent: float
        max_objects: int
        min_objects: int
        object_dataset: ObjectDataset

    @dataclasses.dataclass
    class RewardConfig(BaseDataclass):
        terminal_reward: float
        lift_success: float
        grasp_reward: float
        delta_z_scale: float
        time_penalty: float
        terminal_reward_wrong: float
        table_clearing: bool

    @dataclasses.dataclass
    class CurriculumConfig(BaseDataclass):
        init_lambda: float
        n_steps: int
        success_threshold: float
        extent: Tuple[float, float]
        robot_height: Tuple[float, float]
        lift_dist: Tuple[float, float]
        max_objects: Tuple[int, int]
        min_objects: Tuple[int, int]

    robot: RobotConfig
    scene: SceneConfig
    reward: RewardConfig
    curriculum: CurriculumConfig
    on_gripper_camera_config: OnGripperCameraConfig
    observation_types: List[Observe]
    time_horizon: int
    real_time: bool = False

