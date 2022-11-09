from typing import Iterator, List, Dict, Union
from ruka.types import Dictator, Array
from ruka.environments.common.env import Observation, Episode
import numpy as np
import copy 

def oa_sequence_iterator(
    it: Iterator[Episode],
    n_collect: int,
    random: bool,
    block_size: int,
    ) -> Iterator[Dict[str, List[Union[Dictator, Array]]]]:
    
    done_idxs = []
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []

    for i in range(n_collect):
        path = next(it)
        observations.extend(path.observations[:-1])
        actions.extend(path.actions)
        rewards.extend(path.rewards)
        dones.extend(path.dones)
        infos.extend(path.infos)
        done_idxs.append(len(observations))
        print(f"Collected {i} path")
    idx = -1
    while True:
        # choosing next index
        if random:
            idx = np.random.randint(len(observations))
        else:
            idx += 1
            idx = idx % len(observations)
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
                [copy.deepcopy(actions_block[0]) for _ in range(number_to_pad)] + actions_block
                        
        yield dict(
            observation_sequence=observations_block,
            action_sequence=actions_block
        )