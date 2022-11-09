# from gym.spaces import Dict
import numpy as np
from .base import Path
import copy


class OAPrefixDataset:
    def __init__(self, block_size, collate_fn=None):
        self._block_size = block_size
        self._done_idxs = []
        self._observations = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._infos = []
        self.loader = None
        self._collate_fn = collate_fn if collate_fn is not None else lambda x: x
        

    def add_path(self, path: Path):
        assert path.is_finished()
        self._observations.extend(path.observations[:-1])
        self._actions.extend(path.actions)
        self._rewards.extend(path.rewards)
        self._dones.extend(path.dones)
        self._infos.extend(path.infos)
        self._done_idxs.append(len(self))

    def __getitem__(self, idx):
        left_idx = max(idx - self._block_size + 1, 0)
        for done_idx in reversed(self._done_idxs):
            if done_idx < idx: # first done_idx less than idx
                left_idx = max(done_idx + 1, left_idx)
                break
        # idx = left_idx + block_size
        observations_block = self._observations[left_idx : idx + 1]
        actions_block = self._actions[left_idx : idx + 1]
        if len(observations_block) < self._block_size:
            number_to_pad = self._block_size - len(observations_block)
            observations_block = \
                [copy.deepcopy(observations_block[0]) for _ in range(number_to_pad)] + observations_block
            actions_block = \
                [copy.deepcopy(actions_block[0]) for _ in range(number_to_pad)] + actions_block
        return observations_block, actions_block

    def __len__(self):
        return len(self._observations) - self._block_size

    def random_batch(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=len(self) < batch_size)
        out_obss = []
        out_acts = []
        targets = []
        for idx in indices:
            obb, acb = self[idx]
            target_action = acb[-1]
            acb = acb[:-1]
            out_obss.append(obb)
            out_acts.append(acb)
            targets.append(target_action)
        return dict(
            observation_sequences=self._collate_fn(out_obss), 
            action_sequences=self._collate_fn(out_acts),
            targets=self._collate_fn(targets),
            )
            
    def get_diagnostics(self):
        return dict(
            buffer_size=len(self),
        )
    def end_epoch(self, epoch):
        pass
