import copy
import cv2
import fcntl, sys, os
import numpy as np
import threading
import time

from .perception.sensor_system import SensorSystem
from .robot import Camera
from functools import wraps
from ruka.util.compression import img2jpg
from ruka_os.globals import streaming_file_from_id, get_infra_setup
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

                    obj.debug_visualizer = get_image_streamer(clonecam.get('fileid'))

                if obj.debug_visualizer:
                    t = extract(result, obj)
                    if t is not None:
                        obj.debug_visualizer.write(t)
                return result
            return inner

        setattr(cls, capture_method, method_decorator(getattr(cls, capture_method)))

        return cls
    return decorator


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

    def visualize(self, frame, sn):
        pass


streamers = {}
def get_image_streamer(id: int):
    """
    Resturns an object of StreamImagesToFile
    if the object was already created returns it

    So we do not re-run any thread for each launch of the camera streaming
    """
    global streamers
    
    sid = f'{id}'
    if streamers.get(sid):
        return streamers.get(sid)

    streamers[sid] = StreamImagesToFile(id)

    return streamers[sid]


class StreamImagesToFile():
    """
    Send image to file and not wasting too much time from main thread
    when encoding and writing JPG
    """
    def __init__(self, fileid: int, use_cv2 : bool = False):
        """
        Initilize our thread for writing frames to a streaming file

        fileid - the number of the port we are streaming from file
        defined in streaming_file_from_id(fileid)
        """
        self.file = streaming_file_from_id(fileid)

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

        self.use_cv2 = use_cv2
    
    def __del__(self):
        """
        Carefully stops thread
        """
        self.closing = True
        self.pic_updated_event.set()
        self.t1.join()

    def worker(self):
        """
        Waiting for the event from a main thread,
        then copy passed data to local var, which later encoded to JPEG
        and written to our streaming file
        """
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
            if self.use_cv2:
                cv2.imwrite(self.file, pic_data)
            else:
                with open(self.file, 'wb') as f:
                    f.write(img2jpg(pic_data))

    def write(self, img):
        """
        Store image in our shared data
        and then notify thread of a new arrived frame
        """
        self.pic_lock.acquire()
        self.picdata = copy.deepcopy(img)
        self.pic_lock.release()
        self.pic_updated_event.set()

