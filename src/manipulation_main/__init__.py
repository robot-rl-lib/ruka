from gym.envs.registration import register

register(
    id='gripper-env-v0',
    entry_point='manipulation_main.gripperEnv.robot:RobotEnv',
)

register(
    id='gripper-env-oracle-v1',
    entry_point='manipulation_main.gripperEnv.state_robot:RobotEnv',
    kwargs=dict(
        obs_version=1,
    )
)

register(
    id='gripper-env-oracle-v2',
    entry_point='manipulation_main.gripperEnv.state_robot:RobotEnv',
    kwargs=dict(
        obs_version=2,
    )
)

register(
    id='gripper-env-oracle-v3',
    entry_point='manipulation_main.gripperEnv.state_robot:RobotEnv',
    kwargs=dict(
        obs_version=3,
    )
)

register(
    id='gripper-env-oracle-v4',
    entry_point='manipulation_main.gripperEnv.state_robot:RobotEnv',
    kwargs=dict(
        obs_version=3,
    )
)
