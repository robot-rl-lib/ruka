import os
import tempfile
import json
import warnings
from typing import List, Dict
from ruka.logging.episode import load_episode
import ruka_os.distributed_fs_v2 as dfs

def label_episode_from_infos(path: str):
    """ Temporary function that sets general interface for labeling: 
        accepts local episode path on DFS, and handles saving: 
        this label function saves json locally and then pushes it, 
        some other might push paths to toloka queue or whatever. 
        So it is this function's responsibility to publish labels to DFS.

        saves labels in json format as 'episode_path' + '.json' in the same dir as path.

        Labeling functions work with fields that cannot be labeled automaticaly in general
        
        args:
            - path - path to a episode on DFS

        label fields:
            - success rate : bool

    """
    save_path = path + '.json'
    
    if dfs.exists(save_path):
        warnings.warn(
            f"skipping episode '{save_path}' since it is already labeled"
            )
        return
    episode = load_episode(path)
    episode_labels = {}
    episode_labels['is_success'] = episode.infos[-1]['is_success']
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = os.path.join(tmpdir, os.path.basename(save_path))
        with open(tmppath, 'w') as f:
            json.dump(episode_labels, f)
        dfs.upload(tmppath, save_path)
