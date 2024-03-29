import atexit
import json
import glob
import os
import pwd
import grp
import shutil
import signal
import time
import threading

import multiprocessing as mp

from pathlib import Path
from ruka_os.globals import ROBOT_REPORT_FILE_TPL, ROBOT_HISTORY_LOG_TPL, \
                            ROBOT_HISTORY_REPORT_ON, ROBOT_HISTORY_REPORT_MAXSIZE_BYTES
from .portal import open_portal
from subprocess import run
from typing import Dict

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

    if ROBOT_HISTORY_REPORT_ON:
        global HistoryReport
        if not HistoryReport:
            HistoryReport = RobotHistoryReporter(name)
        HistoryReport.log(data)


HistoryReport = False
PORTAL_NAME = 'robot_history'

# get directory of robot historical reports
def get_robot_history_file(name, num = None):
    if not num:
        return ROBOT_HISTORY_LOG_TPL.replace('%name%', name)
    s = ROBOT_HISTORY_LOG_TPL.replace('%name%', name)
    return f'{s}.{num:04d}'


class RobotHistoryReporter:
    def __init__(self, name):        
        self._name = name
        self._portal = None
        self._last_try = 0

    def _activate_portal(self):
        if not self._portal and (time.time()-self._last_try > 20):
            self._portal = open_portal(PORTAL_NAME)
            if not self._portal:
                self._last_try = time.time()
            else:
                self._last_try = 0
        
    def log(self, data: Dict):
        try:
            self._activate_portal()
            if self._portal:
                self._portal.send({'name': self._name, 'data': data})
        except:
            del self._portal
            self._portal = None


class HistoryLogger:
    """
    Waits command from pipe and:
    1. Exists if requested
    2. Sends frame if requested
    """
    def __init__(self, name):
        # from main thread
        self._name = name
        
        self._curfile = get_robot_history_file(self._name)
        self._rotation_check = 0

        self.fp = None
        self.open_log_file()

        self.prev_data = False

    def make_log_file(self):
        if not os.path.isfile(self._curfile):
            Path(self._curfile).touch()
            os.chmod(self._curfile, 0o666)
            uid = pwd.getpwnam("root").pw_uid
            gid = grp.getgrnam("video").gr_gid
            os.chown(self._curfile, uid, gid)
        
    def open_log_file(self):
        self.make_log_file()
        try:
            self.fp = open(self._curfile, "a")
        except:
            self.fp = None

    def close_log_file(self):
        if self.fp:
            self.fp.close()

    def rotate(self):
        self._rotation_check += 1
        
        # not too often check file size
        if self._rotation_check < 100:
            return

        self._rotation_check = 0

        size = os.path.getsize(self._curfile)
        if size < ROBOT_HISTORY_REPORT_MAXSIZE_BYTES:
            return 

        self.close_log_file()

        # do rotation
        files = glob.glob(self._curfile + ".*")
        files.sort(reverse=True)
        if len(files)>0:
            name = files[0]
            d = name.split(self._curfile+'.')
            if len(d)<2:
                raise RuntimeError('Wrong file name for reporting')
            lastnum = int(d[1])
            lastnum += 1
            prev = get_robot_history_file(self._name, lastnum)
            for f in files:
                os.rename(f,prev)
                prev = f

        lastnum = 1
        prev = get_robot_history_file(self._name, lastnum)
        os.rename(self._curfile, prev)


        self.open_log_file()

    def log(self, data: Dict):
        if not self.fp:
            return

        self.rotate()

        # if position is the same as previous, dont log it
        # this will keep log clean, and, as we put timestamp in data,
        # we may still track that robot was standing same position for
        # some time
        sd = json.dumps(data)
        if sd == self.prev_data:
            return
        self.prev_data = sd

        data['ts'] = time.time()
        json.dump(data, self.fp)
        self.fp.write('\n')
        self.fp.flush()


# get live robot params
def get_history_robot_data(name, num = None):
    """
    Load robot historical reports

    name - robot identifier
    """
    files = []
    if num:
        files.append(get_robot_history_file(name, num))
    else:
        files.append(get_robot_history_file(name,1))
        files.append(get_robot_history_file(name))
        
    logs = []
    for f in files:
        if os.path.isfile(f):
            with open(f) as file:
                for line in file:
                    l = line.rstrip()
                    try:
                        d = json.loads(l)
                    except:
                        continue
                    logs.append(d)
    return logs

def list_history_available_file(name):
    tpl = get_robot_history_file(name)
    files = glob.glob(tpl + ".*")
    res = []
    for f in sorted(files):
        d = f.split(tpl+'.')
        if len(d)<2:
            continue
        lastnum = int(d[1])
        res.append(lastnum)
    return res
    
