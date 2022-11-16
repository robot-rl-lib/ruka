import gym
import numpy as np
import pickle
import pytest
import sys
import tempfile

from collections.abc import Set, Mapping
from collections import deque
from numbers import Number
from ruka.logging.logger import create_ruka_logger, create_ruka_log_reader, FPSParams
from ruka.environments.common.path_iterators import collect_episodes
from ruka.environments.common.env import Policy
from ruka.observation import Observe
from ruka.environments.robot_env_v5 import RobotEnv
from ruka.environments.robot_env_v5.configs import EnvironmentConfig, ObjectDataset
from manipulation_main.sb3_training.wrapper import ImageToPyTorchDictLike
from ruka.logging.episode import EpisodeLogParams, log_episode, get_episode, \
    save_episode, load_episode
from ruka.util.array_semantics import RGB, Grayscale, Depth


# -------------------------------------------------------------------- Setup --


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
            observation_types=[
                Observe.RGB,
                Observe.DEPTH,
                Observe.GRAY,
                Observe.TARGET_SEGMENTATION,
                Observe.ROBOT_POS,
                Observe.GRIPPER
            ],
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


def PSNR(original, compressed, max_pixel=255.0):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


class ConvImgs(gym.ObservationWrapper):
    def observation(self, obs):
        gray = obs['gray']
        obs['rgb'] = RGB((obs['rgb'] * 255).astype(np.uint8))
        obs['gray'] = Grayscale((gray * 255).astype(np.uint8))
        obs['mask'] = Depth((gray * 255).astype(np.uint16))
        obs['depth'] = Depth((obs['depth'] * 1000).astype(np.uint16))
        return obs


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)

    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = 0
        if not isinstance(obj, np.ndarray):
            size = sys.getsizeof(obj)
        else:
            size = obj.nbytes
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))

        return size
    return inner(obj_0)


def compare_items(v1, v2, name):
    assert type(v1) == type(v2), (type(v1), type(v2), name)
    if isinstance(v1, dict):
        for k,vv1 in v1.items():
            assert k in v2.keys(), (k, list(v2.keys()), list(v1.keys()), name)
            vv2 = v2[k]
            compare_items(vv1, vv2, f"{name}/{k}")

    if isinstance(v1, (float, int, str, bool)):
        assert v1 == v2, (v1, v2, name)

    if isinstance(v1, (tuple, list)):
        assert len(v1) == len(v2), (len(v1), len(v2), name)
        for i in range(len(v1)):
            compare_items(v1[i], v2[i], f"{name}/{i}")

    if isinstance(v1, np.ndarray):
        assert v1.shape == v2.shape, (v1.shape, v2.shape, name)
        v_max = 255
        if 'depth' in name:
            v_max = 1000
        assert PSNR(v1, v2, v_max) > 35, (PSNR(v1, v2, v_max), v_max, name)

    if v1 is None:
        assert v2 is None, (v2, name)

    if not isinstance(v1, (bool, type(None), float, int, str)):
        assert id(v1) != id(v2), (v1, v2, name)


def compare_episodes(ep1, ep2):
    assert len(ep1) == len(ep2), (len(ep1),len(ep2))

    for i, (v1, v2) in enumerate(zip(ep1.observations, ep2.observations)):
        compare_items(v1, v2, f"observations/{i}")

    for i, (v1, v2) in enumerate(zip(ep1.infos, ep2.infos)):
        compare_items(v1, v2, f"infos/{i}")

    for i, (v1, v2) in enumerate(zip(ep1.rewards, ep2.rewards)):
        compare_items(v1, v2, f"rewards/{i}")

    for i, (v1, v2) in enumerate(zip(ep1.actions, ep2.actions)):
        compare_items(v1, v2, f"actions/{i}")

    for i, (v1, v2) in enumerate(zip(ep1.dones, ep2.dones)):
        compare_items(v1, v2, f"actions/{i}")



# -------------------------------------------------------------------- Tests --


def test_uncompressed():
    env = env_fn(img2pytorch = False)
    episode = next(collect_episodes(env, RandomPolicy()))
    env.close()
    video_fps = FPSParams(total_time=(len(episode) / 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Store.
        path = f'{tmpdir}/logdir'
        with create_ruka_logger(path) as logger:
            log_episode(
                logger,
                episode,
                EpisodeLogParams(video_fps=video_fps, compress=False)
            )

        # Read back.
        reader = create_ruka_log_reader(path)
        for key in reader.get_keys():
            assert key.startswith('episode/')
        assert pickle.dumps(episode) == pickle.dumps(get_episode(reader))


@pytest.mark.external
def test_remote():
    env = env_fn(img2pytorch = False)
    episode = next(collect_episodes(env, RandomPolicy()))
    env.close()
    video_fps = FPSParams(total_time=(len(episode) / 10))

    # Store.
    remote_path = f'test/logging/episode'
    save_episode(
        remote_path,
        episode,
        EpisodeLogParams(video_fps=video_fps, compress=False),
        wait=True
    )

    # Read back.
    assert pickle.dumps(episode) == pickle.dumps(load_episode(remote_path))


def test_compressed():
    env = ConvImgs(env_fn(img2pytorch=False))
    episode = next(collect_episodes(env, RandomPolicy()))
    env.close()
    video_fps = FPSParams(total_time=(len(episode) / 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Store uncompressed.
        path_u = f'{tmpdir}/logdir_u'
        with create_ruka_logger(path_u) as logger:
            log_episode(
                logger,
                episode,
                EpisodeLogParams(video_fps=video_fps, compress=False)
            )
            size_u = getsize(logger)

        # Store compressed.
        path_c = f'{tmpdir}/logdir_c'
        with create_ruka_logger(path_c) as logger:
            log_episode(
                logger,
                episode,
                EpisodeLogParams(video_fps=video_fps, compress=True)
            )
            size_c = getsize(logger)

        # Read back.
        assert size_c < int(size_u * 0.6), (size_c, size_u)
        reader = create_ruka_log_reader(path_c)
        compare_episodes(episode, get_episode(reader))