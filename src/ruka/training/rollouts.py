import copy
import numpy as np
import torch as th
from typing import List, Dict

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import ruka.pytorch_util as ptu
from ruka.observation import Observation
from ruka.environments.common.env_util import get_supported_robot_env

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
        save_image=False,
        device=None,
    ):

    if get_action_kwargs is None:
        get_action_kwargs = {}

    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    next_observations = []
    images = []

    if last_obs is None:
        agent.reset()
        o = vec_env.reset()
        if isinstance(o, dict):
            o = Observation(o)
        if save_image:
            images.append(get_supported_robot_env(vec_env, 'get_image').get_image())         
    else:
        o = last_obs

    for _ in range(num_steps):
        o_on_device = o
        if isinstance(o, Observation):
            o_on_device = o.to_pytorch(other_device=device)
        elif isinstance(o, np.ndarray):
            o_on_device = ptu.from_numpy(o, other_device=device)
        elif isinstance(o, th.Tensor):
            o_on_device = o.to(device=device or ptu.device)

        a = agent.get_actions(o_on_device, **get_action_kwargs)
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
        if save_image:
            images.append(get_supported_robot_env(vec_env, 'get_image').get_image())        

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
        images=np.array(images) if images else None,
    )


def join_rollout(rollouts: List[Dict]):
    """ Concatinate all rollouts in one """
    res = {}
    for k in rollouts[0].keys():
        first_item = rollouts[0][k]
        if isinstance(first_item, np.ndarray):
            res[k] = np.concatenate([r[k] for r in rollouts], axis=0)
        elif isinstance(first_item, Observation):
            res[k] = Observation.stack([r[k] for r in rollouts])
        elif isinstance(first_item, list):
            # join lists
            res[k] = [j for i in [r[k] for r in rollouts] for j in i]
        elif first_item is None:
            res[k] = None
        else:
            er_message = f'Unsupported rollout item {k} type {type(res[k])} for join'
            raise ValueError(er_message)
    return res