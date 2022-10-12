import numpy as np
import copy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from ruka.observation import Observation

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
        last_obs=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    next_observations = []
    path_length = 0
    if last_obs is None:
        agent.reset()
        o = env.reset()
        if reset_callback:
            reset_callback(env, agent, o)
    else:
        o = last_obs
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)

        a = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_next_obs,
    )


def vec_rollout(
        vec_env,
        agent,
        num_steps,
        get_action_kwargs=None,
        last_obs=None,
    ):

    if get_action_kwargs is None:
        get_action_kwargs = {}

    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    next_observations = []

    if last_obs is None:
        agent.reset()
        o = vec_env.reset()
        if isinstance(o, dict):
            o = Observation(o)
    else:
        o = last_obs

    for _ in range(num_steps):
        a = agent.get_actions(o.to_pytorch() if isinstance(o, Observation) else o, **get_action_kwargs)
        next_o, r, d, env_info = vec_env.step(copy.deepcopy(a))
        if isinstance(next_o, dict):
            next_o = Observation(next_o)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        env_infos.append(env_info)
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = Observation.stack(observations) if isinstance(observations[0], Observation) else np.array(observations)
    next_observations = Observation.stack(next_observations) if isinstance(next_observations[0], Observation) else np.array(next_observations)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,

    )
