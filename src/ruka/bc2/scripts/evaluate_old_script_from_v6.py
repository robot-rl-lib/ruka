import os
import sys
import functools
import pickle

# For debug
sys.path.extend(['/home/amanoshin/arcadia/ytech/cobots/lib/ruka/src','/home/amanoshin/arcadia/ytech/cobots/os'])

import ruka.pytorch_util as ptu
# from manipulation_main.sb3_training.wrapper import ImageToPyTorchDictLike
from ruka.environments.robot_env_v6 import RobotEnv
from ruka.environments.robot_env_v6.configs import EnvironmentConfig, Observe, ObjectDataset
from ruka.models.cnn_encoders import AugmentedNatureCNNDictLike, AugmentedNatureCNN
from ruka.models.mlp import ConcatEncoderMlp, ConcatMlp
from ruka.models.policy import EncoderTanhGaussianPolicy, MakeDeterministic

from ruka.models.qf import SharedEncCritics
from ruka.training.evaluator import Evaluator
import ruka.util.distributed_fs as dfs
from ruka.environments.gym_wrappers import LatencyWrapper


import ruka.util.distributed_fs as dfs
import argparse
import numpy as np
from ruka.bc2.path_collector import PathCollector
from ruka.bc2.base import Path
from ruka.bc2.wrappers_ import WrapObs, ImageToPyTorchDictLike

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default="exp/37_rad_segmgoal_sparse_lat2__tl200-1/190.snapshot")
    parser.add_argument('--n-rollouts', type=int, default=50)
    args = parser.parse_args()
    return args

def execute_policy(args):
    ptu.set_gpu_mode(True)
    print(f"using {ptu.device} device!")

    def env_fn(eval: bool):
        env = RobotEnv(
            config=EnvironmentConfig(
                robot=EnvironmentConfig.RobotConfig(
                    model_path='models/gripper/wsg50_one_motor_gripper_new.sdf',
                    action_wait=.1,
                    gripper_wait=.1,
                    max_translation=.03,
                    max_yaw_rotation=.015,
                    max_force=100,
                    max_speed=1.
                ),
                scene=EnvironmentConfig.SceneConfig(
                    extent=.1,
                    max_objects=6,
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
                    max_objects=[3, 3],
                    min_objects=[3, 3],
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

                time_horizon=200,
                observation_types=[Observe.DEPTH, 
                                    Observe.TARGET_SEGMENTATION, Observe.SENSOR_PAD, 
                                    Observe.GRAY, Observe.HEIGHT, Observe.GRIPPER],
                real_time=False,
            ),
            validate=eval
        )
        env = ImageToPyTorchDictLike(env)
        env = LatencyWrapper(env, latency_steps=2)
        env = WrapObs(env)
        return env
        
    eval_env = env_fn(True)


    enc_p = AugmentedNatureCNN(
        input_dims=(3, 64, 64),
        num_direct_features=1,
    )    
    remote_path = args.snapshot# "baseline.snapshot" # sys.argv[1]
    local_path = os.path.join("aux_data", os.path.basename(remote_path.replace("/", "-")))

    if os.path.isfile(local_path):
        print("baseline policy found")
    else:
        print("downloading baseline policy")
        dfs.download(
            remote_path,
            local_path
            )
    
    print(f"loading policy from {local_path}")
    with open(local_path, 'rb') as f:
        snapshot = pickle.load(f)
    
    policy = EncoderTanhGaussianPolicy(
        encoder=enc_p,
        hidden_init=None,
        action_dim=eval_env.action_space.shape[0],
        hidden_sizes=[64, 64],
    )

    policy.load_state_dict(snapshot['trainer/policy'])
    policy.to('cuda')

    class AddReset(EncoderTanhGaussianPolicy):
        def reset(self):
            pass

    stateful_policy = AddReset

    from ruka.bc2.evaluator import Evaluator
    eval_path_collector = Evaluator(
        env=eval_env,
        policy=stateful_policy,
        video_prefix="vid",
        save_video_every=10
    )
    print(eval_path_collector.calculate_metrics())


if __name__=="__main__":
    args = parse_args()
    execute_policy(args)
      

