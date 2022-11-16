import os
import pickle
import json
import warnings
from typing import List, Dict

def label_local_from_infos(paths: List[str], save_to: str) -> Dict[str, Dict[str, float]]:
    """ Temporary function that sets general interface for labeling: 
        accepts local episode paths (maybe episode DFS IDs later in other variations), 
        and handles saving: this label function saves json locally, some other might push
        labels from toloka or whatever to dfs directly

        Labeling functions work with fields that cannot be labeled automaticaly in general

        paths - local path to files
        save_to - where to locally save json

        dict:
            episode_path:
                is_success: bool
                episide_len: int
    """
    for path in paths:
        with open(path, 'rb') as f:
            episode = pickle.load(f)

        name = os.path.basename(path) + '.json'
        save_path = os.path.join(save_to, name)
        if os.path.exists(save_path):
            warnings.warn(
                f"skipping episode '{os.path.basename(path)}' since it is already labeled"
                )
            continue

        episode_labels = {}
        episode_labels['is_success'] = episode.infos[-1]['is_success']
        with open(save_path, 'w') as f:
            json.dump(episode_labels, f)