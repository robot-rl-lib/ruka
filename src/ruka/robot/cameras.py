import atexit
import copy
import cv2
import fcntl, sys, os
import numpy as np
import signal
import socket
import struct
import threading
import time

import multiprocessing as mp

from .robot import Camera
from .realtime import Watchdog, WatchdogParams
from enum import Enum
from functools import wraps
from pathlib import Path
from ruka.util.compression import img2jpg
from ruka_os.globals import streaming_file_from_id, get_infra_setup, \
                            streaming_sock_from_id, streaming_info_from_file, \
                            LIVE_STREAMING_TIME_DELTA, DEFAULT_STREAM_JPG_QUALITY, \
                            DEBUG_CAMERA_FPS
from typing import Dict, Tuple, Any, Optional, Union, Callable
from v4l2 import *

class FakeRGBDCamera(Camera):
    """
    Setup an empty camera, so the engine is happy.
    But we may attach the camera to exclusive usage to watch.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        
    def start(self):
        pass        

    def stop(self):
        pass
        
    def capture(self) -> np.ndarray:
        color = np.zeros((self._height, self._width, 3))
        depth = np.zeros((self._height, self._width ))
        return np.dstack([color, depth])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


def ConvertToYUYV(sizeimage, bytesperline, im):
    buff = np.zeros((sizeimage, ), dtype=np.uint8)
    imgrey = im[:,:,0] * 0.299 + im[:,:,1] * 0.587 + im[:,:,2] * 0.114
    Pb = im[:,:,0] * -0.168736 + im[:,:,1] * -0.331264 + im[:,:,2] * 0.5
    Pr = im[:,:,0] * 0.5 + im[:,:,1] * -0.418688 + im[:,:,2] * -0.081312
    for y in range(imgrey.shape[0]):
        #Set lumenance
        cursor = y * bytesperline
        for x in range(imgrey.shape[1]):
            try:
                buff[cursor] = imgrey[y, x]
            except IndexError:
                pass
            cursor += 2

        #Set color information for Cb
        cursor = y * bytesperline
        for x in range(0, imgrey.shape[1], 2):
            try:
                buff[cursor+1] = 0.5 * (Pb[y, x] + Pb[y, x+1]) + 128
            except IndexError:
                pass
            cursor += 4

        #Set color information for Cr
        cursor = y * bytesperline
        for x in range(0, imgrey.shape[1], 2):
            try:
                buff[cursor+3] = 0.5 * (Pr[y, x] + Pr[y, x+1]) + 128
            except IndexError:
                pass
            cursor += 4

    return buff.tostring()


class CloneRGBDCamera(Camera):
    """
    Camera which clones the base case picture to a loopback video device
    Usage - wrap around the base came into this one:
        CloneRGBDCamera(RealsenseCamera(RealsenseConfig(width=640, height=480, fps=30)), 21),    

    Streaming from the device have up-to 1 sec latency, but may allows to work
    both the camera in engine and to view the results
    """

    def __init__(self, orig_cam, devid = False, fileid = False):
        self._orig_cam = orig_cam
        width = self._orig_cam.width
        height = self._orig_cam.height
        self.format = False
        self.file = False

        if not devid is False:
            devName = f'/dev/video{devid}'
            if not os.path.exists(devName):
                raise NameError(f'Bad /dev/video{devid} device')
            self.device = open(devName, 'wb')
            format                      = v4l2_format()
            format.type                 = V4L2_BUF_TYPE_VIDEO_OUTPUT
            format.fmt.pix.pixelformat  = V4L2_PIX_FMT_YUYV
            format.fmt.pix.width        = width
            format.fmt.pix.height       = height
            format.fmt.pix.field        = V4L2_FIELD_NONE
            format.fmt.pix.bytesperline = width * 2
            format.fmt.pix.sizeimage    = width * height * 2
            format.fmt.pix.colorspace   = V4L2_COLORSPACE_JPEG
            fcntl.ioctl(self.device, VIDIOC_S_FMT, format)
            self.format = format
        if not fileid is False:
            self.file = streaming_file_from_id(fileid)
 
    def start(self):
        self._orig_cam.start()

    def stop(self):
        self._orig_cam.stop()

    def capture(self) -> np.ndarray:
        d = self._orig_cam.capture()
        self.send_frame(d[:,:,:3])
        return d

    @property
    def width(self):
        return self._orig_cam.width

    @property
    def height(self):
        return self._orig_cam.height

    def send_frame(self, img):
        if self.format:
            buff = ConvertToYUYV(self.format.fmt.pix.sizeimage, self.format.fmt.pix.bytesperline, img)
            self.device.write(buff)
        if self.file:
            with open(self.file, 'wb') as f:
                f.write(img2jpg(img))


def is_streaming_live(fn: str):
    info_file = streaming_info_from_file(fn)
    if not sys.path.isfile(f):
        return False
    mtime = os.path.getmtime(info_file)
    tnow = time.time()
    return tnow - mtime < LIVE_STREAMING_TIME_DELTA


def update_streaming_live(fn: str, prev_update_ts: float) -> float:
    tnow = time.time()
    if tnow - prev_update_ts < LIVE_STREAMING_TIME_DELTA / 2:
        return prev_update_ts
    info_file = streaming_info_from_file(fn)
    Path(info_file).touch()
    return tnow


def visualize_camera(sn_field: str, extract = lambda frame,obj: frame, capture_method: str = 'capture'):
    """
    Decorator for the camera to extract image and send it to cloning file:

    sn_field - field in the Camera Object where the Camera SN is stored
    extract - how to extract the RGB frame from the results of capture().
              called as: extract(result, object)    
              so based on your config of object you may return frame
              if extract returns None - skip the frame
    capture_method - name of the main capture method (default: capture)
    """
    def decorator(cls):
        def method_decorator(func):
            @wraps(func)
            def inner(*args, **kwargs):

                obj = args[0]

                result = func(*args, **kwargs)

                # check we are good
                if hasattr(obj, 'debug_visualizer'):
                    if obj.debug_visualizer is False:
                        return result
                else:
                    # initialize
                    obj.debug_visualizer = False
                    conf = get_infra_setup()
                    if not isinstance(conf.get('cams'), list):
                        return result
                    clonecam = False
                    for cam in conf.get('cams'):
                        if not cam.get('clone_from'):
                            continue
                        if not cam.get('clone_from').get('class') or not cam.get('clone_from').get('sn'):
                            raise ValueError('clone_from setup is wrong need two fields: class and sn')
                        if type(obj).__name__ == cam.get('clone_from').get('class') and cam.get('clone_from').get('sn') == getattr(obj, sn_field):
                            clonecam = cam
                            break
                    if not clonecam:
                        return result

                    obj.debug_visualizer = get_image_streamer(clonecam, cls)

                if obj.debug_visualizer:
                    t = extract(result, obj)
                    if t is not None:
                        obj.debug_visualizer.write(t)
                return result
            return inner

        setattr(cls, capture_method, method_decorator(getattr(cls, capture_method)))

        return cls
    return decorator


class ThreadProxyCamera(Camera):
    """
    Creates a separate thread and inside captures data from the slave camera
    In main thread the capture() returns last available frame.
    Thus, there is almost no delay in calling capture(), which is good.

    slave argument to constructor is the original camera (e.g. Realsense)
    ThreadProxyCamera is the interface to the threaded slave()

    User code queries `capture()` from ThreadProxyCamera, while slave is working in worker() loop.
    """
    # keep slave as a class variable
    def __init__(self, slave, fps: Optional[float] = None):
        self.slave = slave
        self.begin(fps)

    def begin(self, fps: Optional[float] = None):
        # MT thread
        self.started = False
        self.closing = False

        # if we want proxy camera to simulate the "freeze" behaviour of slave
        # then we should delay in capture method()
        self.delay = None if fps is None else 1/fps
        self.wdog = None
        if self.delay:
            self.wdog = Watchdog(WatchdogParams(
                dt=self.delay,
                grace_period=self.delay*10,
                max_fail_time=0,
                max_fail_rate=0,
                window_in_steps=0
            ))

        # lock
        self.pic_lock = threading.Lock()

        # shared data object so we should lock when reading
        self.last_pic = [[],[]]
        self.cur_pic_index = 0 # index of the actual pic is located
        
        # create and launch thread
        self.t1 = threading.Thread(target=self.worker)
        self.t1.daemon = True
        

    def worker(self):
        """
        Waiting for the event from a main thread,
        then copy passed data to local var, which later encoded to JPEG
        and written to our streaming file
        """

        nn = 0
        start = time.time()
        camname = type(self.slave).__name__

        while not self.closing:
            # camera is freezing in this function
            # to get another frame according to its FPS
            pic = self.slave.capture()

            # after we should update the pic storage with changes from step to
            # step from zero to 1, thus we always have the fresh frame we we 
            # need it without waiting
            # this lock potentially may freeze this worker, but only on the time
            # of deepcopy(), which expected to be faster than the capture() call
            # above - so this is OKAY!
            self.pic_lock.acquire()
            self.cur_pic_index = 1 - self.cur_pic_index
            self.last_pic[self.cur_pic_index] = pic
            self.pic_lock.release()

            if DEBUG_CAMERA_FPS:
                nn += 1
                if nn % 30 == 0:
                    b = time.time()

                    print(f'=== FPS of {camname}:', nn/(b-start))
                    start = b
                    nn = 1

    # stop on emregncy when deleting ThreadProxyCamera - ouch - never!
    def __del__(self):
        #Carefully stops thread
        self.force_stop()

    def force_stop(self):
        if self.started:
            self.closing = True
            self.t1.join()
            if self.slave:
                self.slave.stop()

    # something changed in config - reload the slave
    def reload(self, slave, fps: Optional[float]):
        self.force_stop()
        self.slave = slave
        self.begin(fps)

    # start only once and keep the slave
    def start(self):
        if not self.started:
            self.slave.start()
            self.started = True
            self.t1.start()

    # we do not stop the slave - never
    def stop(self):
        pass

    def capture(self):
        if not (self.wdog is None):
            wait_time = self.wdog.get_time_to_sleep()
            time.sleep(wait_time)
            self.wdog.step()
        
        while True:
            # get last good frame without waiting capture() to be finished
            self.pic_lock.acquire()
            picdata = copy.deepcopy(self.last_pic[self.cur_pic_index])
            self.pic_lock.release()
            if len(picdata)>0:
                break
            # for the first frame we may got an empty array []
            # so lets sleep a bit and whait while worker fills the data
            # this shoould be a sub fps wait:
            time.sleep(0.001)
        
        return picdata

    @property
    def width(self):
        return self.slave.width

    @property
    def height(self):
        return self.slave.height

# This var is set to True in child proccess, so we donot generate the
# ProcessProxyCamera decorator again
IN_SLAVE_SUBPROCESS = False

class ProcessProxyCamera(Camera):
    """
    Creates a separate process which captures data from the real camera, then
    sends frames back to the caller.

    This class creates a child process, which inside loops forever the original
    camera (we also call it slave here) for frames.

    The ProcessProxyCamera class is used by the caller to capture() frames.
    So the flow is the following:
    1. User calls: ProcessProxyCamera.capture() 
    2. ProcessProxyCamera sends request for a frame through pipe to child process
    3. Child process deblocks on arrived request for a frame
    4. Child process gets last_frame from camera loop
    5. Child process sends the frame back to ProcessProxyCamera through pipe

    Thus, the ProcessProxyCamera class in the MAIN process doesn't create the
    original camera (Realsense) object at all! It only proxies queries to child
    process. The child process creates the Realsense camera in a thread and queries it.

    Here we are using SPAWN instead of FORK for generating new process.
    This is required because the FORKed process could have issues with opened
    resources, which causes camera not to start working.

    Here:
    https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    it is said that fork is unsafe, which we see here with the camera.

    So let's go with the spawn.
    """
    # keep slave as a class variable
    def __init__(self,
                 cam_setup: Tuple[str,Any,Any],
                 fps: Optional[float] = None,
                 width: Optional[int] = None,
                 height: Optional[int]=None):
        self.cam_setup = cam_setup
        self._width = width
        self._height = height
        self.begin(fps)

    def begin(self, fps: Optional[float] = None):

        self.started = False

        self.delay = None if fps is None else 1/fps
        self.wdog = None
        if self.delay:
            self.wdog = Watchdog(WatchdogParams(
                dt=self.delay,
                grace_period=self.delay*10,
                max_fail_time=0,
                max_fail_rate=0,
                window_in_steps=0
            ))

        self._conn, child_conn = mp.Pipe()
        self._camera_loop_process = mp.get_context('spawn').Process(
            target=ProcessProxyCamera._forked_camera_loop,
            args=(child_conn, self.cam_setup)
            #, daemon=True    cannot do for a spawned process
        )
        signal.signal(signal.SIGCHLD, self.child_exited)

        self._camera_loop_process.start()
        if not self._conn.poll(10):
            raise RuntimeError('Slave camera in child process not started for 10 secs. Something is dead.')
        d = self._conn.recv()
        
    @staticmethod
    def child_exited(sig, frame):
        pass

    @staticmethod
    def _forked_camera_loop(conn, cam_setup):
        """
        We should:
        1. Create capture loop same was as in ThreadProxyCamera
        2. Create thread which sends data to caller() using pipe
        """

        class FrameSender:
            """
            Waits command from pipe and:
            1. Exists if requested
            2. Sends frame if requested
            """
            def __init__(self, pipe, slave):
                # from main thread
                self.pipe = pipe
                self.pic_lock = threading.Lock()
                self.closing = False
                self._slave = slave

                # create and launch thread
                self.t1 = threading.Thread(target=self.worker)
                self.t1.daemon = True
               
                # shared data object so we should lock when reading
                self.last_pic = [[],[]]
                self.cur_pic_index = 0 # index of the actual pic is located

            def worker(self):
                """
                Waiting for the request from pipe and send back the last captured frame
                """
                while self.pipe.poll(None):
                    d = self.pipe.recv()
                    if d == 'F':
                        while True:                    
                            # get last good frame without waiting capture() to be finished
                            self.pic_lock.acquire()
                            picdata = copy.deepcopy(self.last_pic[self.cur_pic_index])
                            self.pic_lock.release()
                            if len(picdata)>0:
                                break
                            # for the first frame we may got an empty array []
                            # so lets sleep a bit and whait while worker fills the data
                            # this shoould be a sub fps wait:
                            time.sleep(0.001)

                        self.pipe.send(picdata)
                    if d == 'W':
                        self.pipe.send(self._slave.width)
                    if d == 'H':
                        self.pipe.send(self._slave.height)
                    if d == 'E':
                        self.closing = True
                        break

            def start(self):
                self.t1.start()
        
        # mark that we are in the subprocess
        global IN_SLAVE_SUBPROCESS
        IN_SLAVE_SUBPROCESS = True
        
        # starting slave camera
        cname = cam_setup[0]
        mname = cam_setup[1]
        cammod = __import__(mname, globals(), locals(), [cname], 0)
        cls = getattr(cammod, cname)
        args = cam_setup[2]
        kwargs = cam_setup[3]
        slave = cls(*args, **kwargs)
        slave.start()
        
        # starting the thread
        framer = FrameSender(conn, slave)
        framer.start()

        nn = 0
        start = time.time()

        # notify that we have started
        conn.send('S')

        while not framer.closing:
            # camera is freezing in this function
            # to get another frame according to its FPS
            pic = slave.capture()

            # after we should update the pic storage with changes from step to
            # step from zero to 1, thus we always have the fresh frame we we 
            # need it without waiting
            # this lock potentially may freeze this worker, but only on the time
            # of deepcopy(), which expected to be faster than the capture() call
            # above - so this is OKAY!
            framer.pic_lock.acquire()
            framer.cur_pic_index = 1 - framer.cur_pic_index
            framer.last_pic[framer.cur_pic_index] = pic
            framer.pic_lock.release()

            if DEBUG_CAMERA_FPS:
                nn += 1
                if nn % 30 == 0:
                    b = time.time()
                    print(f'=== FPS of new process:', nn/(b-start))
                    start = b
                    nn = 1
        conn.send('OK')

    def send_cmd(self, cmd):
        self._conn.send(cmd)

    # stop on emregncy when deleting ThreadProxyCamera - ouch - never!
    def __del__(self):
        self.force_stop()

    def force_stop(self):
        if self.started:
            try:
                self.send_cmd('E')
                msg = self._conn.recv()
            except:
                pass
            self.started = False

    def reload(self,
               cam_setup: Tuple[str,Any,Any],
               fps: Optional[float],
               width: Optional[int] = None,
               height: Optional[int] = None):
        self.force_stop()
        self.cam_setup = cam_setup
        self._width = width
        self._height = height
        self.begin(fps)

    # start only once and keep the slave
    def  start(self):
       if not self.started:
            self.started = True

    # we do not stop the slave - never
    def stop(self):
        pass

    def capture(self):
        if not self.started:
            return None

        if not (self.wdog is None):
            wait_time = self.wdog.get_time_to_sleep()
            time.sleep(wait_time)
            self.wdog.step()

        self.send_cmd('F')
        return self._conn.recv()
        
    @property
    def width(self):
        if not self._width:
            self._conn.send('W')
            self._width = self._conn.recv()
        return self._width

    @property
    def height(self):
        if not self._height:
            self._conn.send('H')
            self._height = self._conn.recv()
        return self._height


# ProcessProxyCamera handles the camera in a separate process
# while the Proxy uses thread
USE_PROCESS_PROXY_CAMERA = False


# --------------------------
instances = {}
def slaveize_camera(sn_field: str, config_field: str, fps_field: Optional[Union[str,float]]=None):
    """
    Decorator for the camera to slaveize it under proxy.

    Users wants to use the camera as:

    cam = RealsenseCamera(...)
    cam.start()
    while(True):
        frame = cam.capture()

    Our goal to keep this code the same for user, but try to detach camera in a 
    separate job:
    1. thread
    2. process

    So for User, the fact that camera started in a separate thread or process
    is transparent. That is why we use a DECORATOR.

    The decorator replaces the real camera with its own object.

    WITHOUT DECORATOR:

    cam = RealsenseCamera(...)
    cam.start()
    while(True):
        frame = cam.capture()


    WITH DECORATOR:

    slave = RealsenseCamera(...)
    cam = ThreadProxyCamera(slave)
    cam.start()
    while(True):
        frame = cam.capture()


    So with decorator, the user is working with a "virtual camera", which we call
    ThreadProxyCamera. Inside the Proxy, we create a thread, where the ORIGINAL camera
    object is working. We call it a slave.

    The slave camera (which is e.g. Realsense) is working on 30fps inside a thread
    or process. Slave just obtains new frame and store it in a local var (last_frame).

    ThreadProxyCamera objects reads the last stored frame from a loval variable 
    (last_frame). Thus the ThreadProxyCamera may work on ANY FPS without any 
    interruption to the Realsense camera flow.

    In this case we may:
     - easily visualize the Realsense camera in high fps, as it is not linked 
       to and env()/step() processes.
     - request frames from ThreadProxyCamera at any fps (slower or higher)

    Glossary:
      ThreadProxyCamera      - class that is using the THREAD
      ProcessProxyCamera - class that is using the PROCESS

    The decorator also checks whether the RealsenseCamera was already slavized
    and do not re-create a new instance. So once we started the slave camera 
    it works till the end of the python script.

    Arguments:
    sn_field - field of the slave object where the camera serial number is stored
    config_field - field name containing the camera config, needed to detect
                   when we requested new config to reload the detached proxy
    fps_field - if not set, then the proxy may feed the frames as fast as you request
                if set as str, this is the field of the slave camera object to read the fps value
                if set as float, this is the fps directly for the proxy camerafor reading
                last 2 options mostly used for debug, when we need a loop like this:
                while True:
                    cam.capture() <--- expecting here to block
    """
    def decorator(cls):        
        def getinstance(*args, **kwargs):
            global instances

            if not USE_PROCESS_PROXY_CAMERA:
                cam = cls(*args, **kwargs)
                sn = getattr(cam, sn_field)
                config = getattr(cam, config_field)
                fps = None
                if fps_field:
                    if isinstance(fps_field, str):
                        fps = getattr(cam, fps_field)
                    else:
                        fps = fps_field
                uid = (cls, sn)
                if uid in instances:
                    if instances[uid]['config'] != config:
                        instances[uid]['camera'].reload(cam, fps)
                        instances[uid]['config'] = config
                if uid not in instances:
                    instances[uid] = {
                        'config': config,
                        'camera': ThreadProxyCamera(cam, fps)
                    }
                return instances[uid]['camera']

            else:
                # if we are in slave - just generate the camera object as requested
                # as this decorator is needed in proxy process only
                global IN_SLAVE_SUBPROCESS
                if IN_SLAVE_SUBPROCESS:
                    return cls(*args, **kwargs)

                if True:
                    # we create cam object just to get the sn and other config
                    cam = cls(*args, **kwargs)
                    sn = getattr(cam, sn_field)
                    config = getattr(cam, config_field)
                    fps = None
                    if fps_field:
                        if isinstance(fps_field, str):
                            fps = getattr(cam, fps_field)
                        else:
                            fps = fps_field
                    width = cam.width
                    height = cam.height

                # this is needed to build camera, we cannot use the local function
                # or camera object it self as they could not be pickled
                # pickling is required as we use "SPAWN" instead of "FORK"
                cname = cls.__name__
                mname = cls.__module__
                cam_setup = cname, mname, args, kwargs
                
                uid = (cls, sn)
                if uid in instances:
                    if instances[uid]['config'] != config:
                        instances[uid]['camera'].reload(cam_setup, fps, width, height)
                        instances[uid]['config'] = config
                if uid not in instances:
                    instances[uid] = {
                        'config': config,
                        'camera': ProcessProxyCamera(cam_setup, fps, width, height)
                    }
                return instances[uid]['camera']

        return getinstance
    return decorator


def exit_handler():
    global instances
    for ins in instances:
        instances[ins]['camera'].force_stop()

atexit.register(exit_handler)

@visualize_camera('sn')
class TestCamera(Camera):
    def __init__(self, num = 20):
        self.cap = cv2.VideoCapture(f'/dev/video{num}')
        self.sn = '222'

    def start(self):
        if not (self.cap.isOpened()):
            raise NotImplementedError()

    def stop(self):
        self.cap.release()

    def capture(self) -> np.ndarray:
        ret, frame = self.cap.read()
        return frame

    @property
    def width(self):
        return 1080

    @property
    def height(self):
        return 720


@slaveize_camera('sn', 'config', 30)
@visualize_camera('sn')
class MockCamera(Camera):
    def __init__(self, config):
        self.sn = '111'
        self.config = config

    def start(self):
        self.fnum = 0
        self.start_t = time.time()

    def stop(self):
        pass

    def capture(self) -> np.ndarray:
        from PIL import Image, ImageDraw
        time.sleep(1/self.config['fps'])
        
        img = Image.new('RGB', (self.width, self.height), color = 'black')
        draw = ImageDraw.Draw(img)

        self.fnum += 1
        b = time.time()
        timed = b-self.start_t
        fps = self.fnum/timed
        
        a = "{:10.4f}".format(timed)
        b = "{:10.4f}".format(fps)
        
        draw.text((0, 0),f"Frame {self.fnum}, elasped {a}, fps {b}",(255,255,255))
                
        return np.array(img)

    @property
    def width(self):
        return self.config['width']

    @property
    def height(self):
        return self.config['height']

    @property
    def fps(self):
        return self.config['fps']

class StreamTo(Enum):
    FILE = 'file'
    SOCK = 'sock'


class StreamDataTransfer:
    def __init__(self, obj):
        self.obj = obj

    def generate_data():
        raise NotImplementedError()


class StreamDataLatency(StreamDataTransfer):
    def generate_data(self):
        data = time.time()
        #print(data, end='\r')
        return data


def stream_key_from_config(stream: Dict) -> Tuple:
    id = False
    if stream.get('fileid'):
        return StreamTo.FILE, stream.get('fileid')
    if stream.get('sockid'):
        return StreamTo.SOCK, stream.get('sockid')
    raise NotImplementedError()


streamers = {}
def get_image_streamer(config: Dict, cls: Any = False):
    """
    Resturns an object of StreamImagesToFile
    if the object was already created returns it

    So we do not re-run any thread for each launch of the camera streaming
    """
    global streamers

    stype, id = stream_key_from_config(config)
    data_transfer = False
    resize = False
    quality = DEFAULT_STREAM_JPG_QUALITY
    if config.get('clone_from'):
        if config.get('clone_from').get('data_transfer'):
            data_transfer = config.get('clone_from').get('data_transfer')
            if not callable(data_transfer.get('data_generator')):
                raise ValueError('data_generator field is not callable')
            data_transfer['get_data'] = data_transfer.get('data_generator')(cls)
        if config.get('clone_from').get('resize'):
            resize = config.get('clone_from').get('resize')
        if config.get('clone_from').get('quality'):
            quality = config.get('clone_from').get('quality')
    
    sid = f'{stype}-{id}'
    if streamers.get(sid):
        return streamers.get(sid)

    if stype == StreamTo.FILE:
        streamers[sid] = StreamImagesToFile(id, data_transfer=data_transfer, resize=resize, quality=quality)
    elif stype == StreamTo.SOCK:
        streamers[sid] = StreamImagesToSocket(id, data_transfer=data_transfer, resize=resize, quality=quality)
    else:
        raise NotImplementedError()

    return streamers[sid]


class StreamImagesTo():
    """
    Send image to file and not wasting too much time from main thread
    when encoding and writing JPG
    """
    def __init__(self, data_transfer = False, resize = False, quality = 95):
        """
        Initilize our thread for writing frames to a streaming engine
        """
        self.closing = False

        # event + lock
        self.pic_updated_event = threading.Event()
        self.pic_lock = threading.Lock()
 
        # create and launch thread
        self.t1 = threading.Thread(target=self.worker)
        self.t1.daemon = True 
        self.t1.start()

        # shared data object so we should lock when reading
        self.picdata = []

        self.prepare_data_transfer(data_transfer)

        self.debug = True if os.environ.get('RUKA_DEBUG_STREAM') else False
        self.debug_n = 0
        self.debug_ts = -1

        self.live_udpate_ts = 0

        self.resize = False
        if resize:
            if not resize.get('width'):
                raise ValueError('Resize for stream is wrong: width not set')
            if not resize.get('height'):
                raise ValueError('Resize for stream is wrong: height not set')
            self.resize = resize
        self.quality = quality
    
    def __del__(self):
        """
        Carefully stops thread
        """
        self.closing = True
        self.pic_updated_event.set()
        self.t1.join()

    def stream_image(self, pic_data):
        raise NotImplementedError()

    def worker(self):
        """
        Waiting for the event from a main thread,
        then copy passed data to local var, which later encoded to JPEG
        and written to our streaming file
        """
        fn = self.get_streaming_file()
        while True:
            ret = self.pic_updated_event.wait(timeout=1)
            if self.closing:
                return
            if not ret:
                continue
            self.pic_updated_event.clear()
            self.pic_lock.acquire()
            pic_data = copy.deepcopy(self.picdata)
            self.pic_lock.release()

            if self.resize:
                pic_data = self.resize_pic(pic_data, self.resize)

            if self.debug:
                self.debug_n += 1
                if self.debug_ts == -1:
                    self.debug_ts = time.perf_counter()
                else:
                    if self.debug_n % 100 == 0:
                        b = time.perf_counter()-self.debug_ts
                        x = self.debug_n / b
                        print(f'DEBUG - streaming FPS: {x}')
                        self.debug_n = 0
                        self.debug_ts = time.perf_counter()

            if fn:
                self.live_udpate_ts = update_streaming_live(fn, self.live_udpate_ts)

            if self.data_transfer:
                pic_data = self.inject_data(pic_data)
                
            self.stream_image(pic_data)

    def write(self, img):
        """
        Store image in our shared data
        and then notify thread of a new arrived frame
        """
        self.pic_lock.acquire()
        self.picdata = copy.deepcopy(img)
        self.pic_lock.release()
        self.pic_updated_event.set()

    def prepare_data_transfer(self, info):
        if not info:
            self.data_transfer = False
            return

        self.data_transfer = True
        self.data_transfer_info = info

        # try one QR code generation and get w and h
        qr = self.data_to_qr('')
        w, h, _ = qr.shape
        if self.data_transfer_info['width'] < w or self.data_transfer_info['height'] < h:
            raise ValueError('Generated QR code dimentions exceeds what we expect')
        
        self.injection_qr_w = self.data_transfer_info['width']
        self.injection_qr_h = self.data_transfer_info['height']

        if not self.data_transfer_info.get('get_data'):
            raise ValueError('get_data is not set')

        self.injection_data_generator = self.data_transfer_info['get_data']

    def data_to_qr(self, text):        
        import qrcode
        qr = qrcode.QRCode(version = self.data_transfer_info.get('qr_version', 1),
                           box_size = self.data_transfer_info.get('qr_boxsize', 2),
                           border = self.data_transfer_info.get('qr_border', 1))
        qr.add_data(text)        
        qr.make(fit = False)
        img = qr.make_image(fill_color = '#010101', back_color = 'white')
        qrimg = np.array(img)
        return qrimg

    def inject_data(self, pic_data):        
        from PIL import Image

        # get the data
        text = self.injection_data_generator.generate_data()

        # generate new image
        in_h, in_w, _ = pic_data.shape
        w = in_w + self.injection_qr_w
        h = max(in_h, self.injection_qr_h)
        fakeimg = np.array(Image.new('RGB', (w, h), color = 'white'))
        
        # copy input image
        fakeimg[:in_h,self.injection_qr_w:,:] = pic_data[:,:,:]

        # copy qr code
        qr = self.data_to_qr(text)
        qr_w, qr_h, _ = qr.shape
        fakeimg[:qr_w,:qr_h,:] = qr[:,:,:]

        return fakeimg
        
    def resize_pic(self, pic_data, resize):
        return cv2.resize(pic_data, dsize=(resize['width'], resize['height']), interpolation=cv2.INTER_LINEAR)

    def get_streaming_file(self):
        return None
    

class StreamImagesToFile(StreamImagesTo):
    """
    Send image to file and not wasting too much time from main thread
    when encoding and writing JPG
    """
    def __init__(self, fileid: int, data_transfer = False, resize = False,
                 quality = 95, use_cv2 : bool = False):
        """
        Initilize our thread for writing frames to a streaming file

        fileid - the number of the port we are streaming from file
        defined in streaming_file_from_id(fileid)
        """
        self.file = streaming_file_from_id(fileid)
        super().__init__(data_transfer=data_transfer, resize=resize, quality=quality)        
        self.use_cv2 = use_cv2

    def stream_image(self, pic_data):
        if self.use_cv2:
            cv2.imwrite(self.file, pic_data)
        else:
            with open(self.file, 'wb') as f:
                f.write(img2jpg(pic_data, quality=self.quality))

    def get_streaming_file(self):
        return self.file


class StreamImagesToSocket(StreamImagesTo):
    """
    Send image to file and not wasting too much time from main thread
    when encoding and writing JPG
    """
    def __init__(self, sockid: int, data_transfer = False, resize = False, quality=95):
        """
        Initilize our thread for writing frames to a streaming socket

        sockid - the number of the port we are streaming from socket
        defined in streaming_sock_from_id(sockid)
        """
        self.sock_file = streaming_sock_from_id(sockid)
        super().__init__(data_transfer=data_transfer, resize=resize, quality=quality)
        self.sock = False

    def __del__():
        super().__del__()
        if self.sock:
            self.sock.close()

    def open_socket(self):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.sock_file)
        except:
            self.kill_socket()            
            return False
        return True

    def kill_socket(self):
        if self.sock:
            self.sock.close()
            self.sock = False
        
    def check_sock(self):
        if not self.sock:
            return self.open_socket()
        return True
   
    def stream_image(self, pic_data):
        if not self.check_sock():
            return
        jpg = img2jpg(pic_data, quality=self.quality)
        size = len(jpg)
        crc = 0
        header = bytearray(struct.pack('<ccLL', b'R', b'K', socket.htonl(size), socket.htonl(crc)))
        try:
            self.sock.send(header)
            self.sock.send(jpg)
        except:
            self.kill_socket()

    def get_streaming_file(self):
        return self.sock_file

