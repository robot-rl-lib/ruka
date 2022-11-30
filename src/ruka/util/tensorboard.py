import os.path
import threading
import time

from torch.utils.tensorboard import SummaryWriter
from ruka_os import distributed_fs as dfs, DFS_CWD, register_tensorboard

from .distributed_fs import should_sync


# ------------------------------------------------------------------- Public --


DOWNLOAD = False


def set_upload_interval(interval_sec):
    """
    Does not cause init().
    """
    _check_is_in_main_thread()

    # If called before init().
    global _DEFAULT_UPLOAD_INTERVAL
    _DEFAULT_UPLOAD_INTERVAL = interval_sec
    
    # Re-create summary writer.
    global _SUMMARY_WRITER
    if _SUMMARY_WRITER is not None:
        _SUMMARY_WRITER.close()
        _SUMMARY_WRITER = SummaryWriter(_LOCAL_PATH, flush_secs=_DEFAULT_UPLOAD_INTERVAL)

    # Notify thread.
    if _THREAD is not None:
        _THREAD.set_upload_interval(interval_sec)


def init():
    """
    Activate tensorboard SummaryWriter, use 'tensorboard' folder;

    If dfs.should_sync(), additionally do this:

    - Download and unpack tensorboard.tar.gz from DFS_CWD if it exists;
    - Launch a thread that periodically packs tensorboard logs into 
      tensorboard.tar.gz and uploads the archive to DFS_CWD.

    Second init() call with any arguments makes no effect.

    In most cases, there is no need to call this function explicitly: all other
    functions call init() themselves.
    """
    _check_is_in_main_thread()

    # Don't run twice.
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    if should_sync():
        # Download.
        do_initial_upload = True
        if dfs.exists(_REMOTE_PATH) and DOWNLOAD:
            print(f'Tensorboard sync: downloading {dfs.get_url(_REMOTE_PATH)}\n')
            dfs.download_and_unpack(_REMOTE_PATH, _LOCAL_PATH, wait=True)
            do_initial_upload = False

        # Start packer thread.
        global _THREAD
        _THREAD = _TensorboardPackerThread(do_initial_upload)
        _THREAD.start()

        # Create folder.
        dfs.mkdir(os.path.dirname(_REMOTE_PATH))

        # Register tensorboard.
        register_tensorboard(_REMOTE_PATH)

    # Create SummaryWriter.
    global _SUMMARY_WRITER
    _SUMMARY_WRITER = SummaryWriter(_LOCAL_PATH, flush_secs=_DEFAULT_UPLOAD_INTERVAL)


def step(step_no: int):
    init()
    global _STEP
    _STEP = step_no


def flush(wait=True):
    init()
    _SUMMARY_WRITER.flush()
    if _THREAD is not None:
        _THREAD.flush(wait)


def scalar(tag: str, scalar: float, step: int = None):
    init()

    if step is None:
        step = _STEP

    _SUMMARY_WRITER.add_scalar(tag, scalar, step)


def add_video(tag: str, vid_tensor: float, step: int = None, fps: int = 10):
    init()

    if step is None:
        step = _STEP

    _SUMMARY_WRITER.add_video(tag, vid_tensor, global_step=step, fps=fps)


def add_image(tag: str, vid_tensor: float, step: int = None, dataformats='CHW'):
    init()

    if step is None:
        step = _STEP

    _SUMMARY_WRITER.add_image(tag, vid_tensor, global_step=step, dataformats=dataformats)

def add_images(tag: str, vid_tensor: float, step: int = None):
    init()

    if step is None:
        step = _STEP

    _SUMMARY_WRITER.add_images(tag, vid_tensor, global_step=step)
    

def add_histogram(tag: str, tensor: float, step: int = None):
    init()

    if step is None:
        step = _STEP

    _SUMMARY_WRITER.add_histogram(tag, tensor, global_step=step)    
    
# ----------------------------------------------------------- Implementation --


_LOCAL_PATH = 'tensorboard'
_REMOTE_PATH = f'{DFS_CWD}/tensorboard.tar.gz'


_INITIALIZED = False
_THREAD = None
_SUMMARY_WRITER = None
_STEP = None
_DEFAULT_UPLOAD_INTERVAL = 300


def _check_is_in_main_thread():
    if not(threading.current_thread() is threading.main_thread()):
        raise RuntimeError('can be called only from the main thread')


class _TensorboardPackerThread(threading.Thread):
    def __init__(self, do_initial_upload):
        super().__init__(name='ruka-tensorboard-packer', daemon=True)

        self._upload_interval_sec = _DEFAULT_UPLOAD_INTERVAL
        self._last_upload_time = 0 if do_initial_upload else time.time()
        self._wake_up = threading.Event()
        self._flush_complete = threading.Event()

    def set_upload_interval(self, interval_sec: int):
        assert interval_sec > 0
        self._upload_interval_sec = interval_sec
        self._wake_up.set()

    def run(self):
        while True:
            timeout = self._get_timeout()
            self._wake_up.wait(timeout)
            self._wake_up.clear()

            timeout = self._get_timeout()
            if timeout == 0:
                self._upload()
                self._flush_complete.set()
                self._last_upload_time = time.time()

    def flush(self, wait):
        self._last_upload_time = 0
        self._flush_complete.clear()
        self._wake_up.set()
        if wait:
            self._flush_complete.wait()

    def _get_timeout(self):
        next_upload = self._last_upload_time + self._upload_interval_sec
        return max(next_upload - time.time(), 0)

    def _upload(self):
        if not os.path.exists(_LOCAL_PATH):
            return

        if not os.path.isdir(_LOCAL_PATH):
            print(f'WARNING: {_LOCAL_PATH} is not a directiory. '
                   'Tensorboard packer will not pack or upload it.')
            return
        
        print(f'Tensorboard sync: uploading ./{_LOCAL_PATH}\n')
        dfs.pack_and_upload(_LOCAL_PATH, _REMOTE_PATH, wait=True)