
def get_supported_robot_env(env, attr_name):
    if hasattr(env, attr_name):
        return env
    if hasattr(env, 'envs'):
        return get_supported_robot_env(env.envs[0], attr_name)
    if hasattr(env, 'env'):
        return get_supported_robot_env(env.env, attr_name)
    return None
