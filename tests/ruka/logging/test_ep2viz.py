import pytest
import time
import numpy as np
from ruka.util.debug import smart_shape
from ruka.logging.episode import create_episode_logger
from ruka.environments.common.path_iterators import collect_episodes
from ruka.environments.common.env import Policy
from ruka.observation import Observe
from ruka.environments.robot_env_v5 import RobotEnv
from ruka.environments.robot_env_v5.configs import EnvironmentConfig, ObjectDataset
from manipulation_main.sb3_training.wrapper import ImageToPyTorchDictLike
from ruka.logging.ep2viz import log_visualize_episode

OBSERVATION_TYPES = [Observe.RGB, Observe.DEPTH, Observe.GRAY, Observe.TARGET_SEGMENTATION, Observe.ROBOT_POS, Observe.GRIPPER]

def env_fn(img2pytorch: bool, add_batch_dim=False):
    env = RobotEnv(
        config=EnvironmentConfig(
            robot=EnvironmentConfig.RobotConfig(
                model_path='models/gripper/wsg50_one_motor_gripper_new.sdf',
                action_wait=.05,
                gripper_wait=.2,
                max_translation=.03,
                max_yaw_rotation=.015,
                max_force=100,
                max_speed=1.
            ),
            scene=EnvironmentConfig.SceneConfig(
                extent=.1,
                max_objects=1,
                min_objects=1,
                object_dataset=ObjectDataset.RANDOM_URDFS,

            ),
            reward=EnvironmentConfig.RewardConfig(
                terminal_reward=0.0,
                lift_success=0.0,
                grasp_reward=0.0,
                delta_z_scale=0.0,
                time_penalty=1.0,
                table_clearing=False,
                terminal_reward_wrong=-1,
            ),
            curriculum=EnvironmentConfig.CurriculumConfig(
                init_lambda=0.0,
                n_steps=4,
                success_threshold=0.7,
                extent=[0.1, 0.1],
                robot_height=[0.25, 0.25],
                lift_dist=[0.1, 0.1],
                max_objects=[1, 1],
                min_objects=[1, 1],
            ),
            on_gripper_camera_config=EnvironmentConfig.OnGripperCameraConfig(
                camera_info=EnvironmentConfig.OnGripperCameraConfig.CameraInfoConfig(
                    height=64,
                    width=64,
                    K=[69.76, 0.0, 32.19, 0.0, 77.25, 32.0, 0.0, 0.0, 1.0],
                    near=0.02,
                    far=2.0,
                ),
                transform=EnvironmentConfig.OnGripperCameraConfig.CameraTransformConfig(
                    translation=[0.0, 0.0573, 0.0451],
                    rotation=[0.0, -0.1305, 0.9914, 0.0],
                ),
                randomize=EnvironmentConfig.OnGripperCameraConfig.CameraRandomizationConfig(
                    focal_length=4,
                    optical_center=2,
                    translation=0.002,
                    rotation=0.0349
                )
        ),

            time_horizon=10,
            observation_types=OBSERVATION_TYPES,
            real_time=False,
        ),
        validate=True
    )
    if img2pytorch:
        env = ImageToPyTorchDictLike(env, add_batch_dim=add_batch_dim)

    return env

class RandomPolicy(Policy):
    def get_action(self, _):
        return np.random.uniform(size=(5,))    

class TestEnvViz:
    def test_hwc(self):
        env = env_fn(img2pytorch = False)
        ep = next(collect_episodes(env, RandomPolicy()))

        ep_logger = create_episode_logger()
        log_visualize_episode(ep_logger, ep, img_format='HWC', has_batch_dim=False)
        ep_logger.assign_tag('test4')
        ep_logger.assign_tag('test5')
        ep_logger.close(episode_time=len(ep.observations)/10)
        env.close()

    def test_chw(self):
        env = env_fn(img2pytorch = True)
        ep = next(collect_episodes(env, RandomPolicy()))

        print(smart_shape(ep.observations[0]))
        ep_logger = create_episode_logger()
        ep_logger.assign_tag('test5')
        ep_logger.assign_tag('test6')
        log_visualize_episode(ep_logger, ep, img_format='CHW', has_batch_dim=False)
        ep_logger.close(episode_time=len(ep.observations)/10)
        env.close()