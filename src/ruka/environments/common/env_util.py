import gym
from ruka.util.nested_dict import NestedDict

def get_supported_robot_env(env, attr_name):
    if hasattr(env, attr_name):
        return env
    if hasattr(env, 'envs'):
        return get_supported_robot_env(env.envs[0], attr_name)
    if hasattr(env, 'env'):
        return get_supported_robot_env(env.env, attr_name)
    return None

def get_space_from_obs(obs: NestedDict):
    space = dict()
    for key, value in obs.items():
        if isinstance(value, dict):
            space[key] = get_space_from_obs(value)
        else:
            space[key] = gym.spaces.Box(low=0, high=1, shape=value.shape)
    return gym.spaces.Dict(space)
