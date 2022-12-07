from typing import Iterator, List, Dict, Union
from ruka.types import Dictator, Array
from ruka.environments.common.env import Observation, Episode
import numpy as np
import copy

def zeros_like(elem: Dictator):
    if isinstance(elem, dict):
        out = dict()
        for k, v in elem.items():
            out[k] = zeros_like(v)
        return out
    else:
        return np.zeros_like(elem)

def oa_sequence_iterator(
    it: Iterator[Episode],
    n_collect: int,
    random: bool,
    block_size: int,
    infinity=True,
    ) -> Iterator[Dict[str, List[Union[Dictator, Array]]]]:

    done_idxs = []
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []

    for i, path in enumerate(it):
        if i >= n_collect:
            break
        observations.extend(path.observations[:-1])
        actions.extend(path.actions)
        rewards.extend(path.rewards)
        dones.extend(path.dones)
        infos.extend(path.infos)
        done_idxs.append(len(observations))
        print(f"Collected {i} path")
    
    while True:
        indexes = list(range(len(observations)))
        if random:
            np.random.shuffle(indexes)
        for idx in indexes:
            # selecting left end of sequence
            left_idx = max(idx - block_size + 1, 0)
            for done_idx in reversed(done_idxs):
                if done_idx < idx: # first done_idx less than idx
                    left_idx = max(done_idx + 1, left_idx)
                    break
            observations_block = observations[left_idx : idx + 1]
            actions_block = actions[left_idx : idx + 1]

            # padding to length. TODO: here is a leakage
            if len(observations_block) < block_size:
                number_to_pad = block_size - len(observations_block)
                observations_block = \
                    [copy.deepcopy(observations_block[0]) for _ in range(number_to_pad)] + observations_block
                actions_block = \
                    [zeros_like(actions_block[0]) for _ in range(number_to_pad)] + actions_block

            yield dict(
                observation_sequence=observations_block,
                action_sequence=actions_block
            )
    
        if not infinity:
            break
