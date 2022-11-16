import json
import os

from ruka_os.globals import ROBOT_REPORT_FILE_TPL

# get file to report robot params
def get_live_robot_params_file(name):    
    return ROBOT_REPORT_FILE_TPL.replace('%name%', name)

# get live robot params
def get_live_robot_params(name):
    """
    Load robot live parameters

    name - robot identifier
    """
    fn = get_live_robot_params_file(name)
    if not os.path.isfile(fn):
        return None
    with open(fn, 'r') as f:
        try:
            params = json.load(f)
        except:
            return None
    return params

# store live robot params
def store_live_robot_params(name, data):
    """
    Store robot live parameters
    
    name - robot identifier
    """
    fn = get_live_robot_params_file(name)
    with open(fn, 'w') as f:
        json.dump(data, f)
