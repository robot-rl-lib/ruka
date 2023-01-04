import atexit
import copy
import cv2
import fcntl, sys, os
import numpy as np
import socket
import struct
import threading
import time

from .robot import Camera
from enum import Enum
from functools import wraps
from pathlib import Path
from ruka.util.compression import img2jpg
from ruka_os.globals import streaming_file_from_id, get_infra_setup, \
                            streaming_sock_from_id, streaming_info_from_file, \
                            LIVE_STREAMING_TIME_DELTA, DEFAULT_STREAM_JPG_QUALITY
from typing import Dict, Tuple, Any
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


class MasterCamera(Camera):

    # keep slave as a class variable
    def __init__(self, slave):        
        self.slave = slave
        self.begin()

    def begin(self):
        # MT thread
        self.started = False
        self.closing = False

        # event + lock
        self.pic_lock = threading.Lock()
        self.pic_updated_event = threading.Event()
        self.pic_updated_took = threading.Event()

        # shared data object so we should lock when reading
        self.last_pic = [[],[]]
        self.cur_pic_index = 0 # index of the actual pic is located
        self.wants = False
 
        # create and launch thread
        self.t1 = threading.Thread(target=self.worker)
        self.t1.daemon = True
        

    def worker(self):
        """
        Waiting for the event from a main thread,
        then copy passed data to local var, which later encoded to JPEG
        and written to our streaming file
        """
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

    # stop on emregncy when deleting MasterCamera - ouch - never!
    def __del__(self):
        #Carefully stops thread
        if self.started:
            self.closing = True
            self.t1.join()
            if self.slave:
                self.slave.stop()

    def force_stop(self):
        if self.started:
            self.closing = True
            self.t1.join()
            if self.slave:
                self.slave.stop()

    def reload(self, slave):
        if self.started:
            self.closing = True
            self.t1.join()
            self.slave.stop()
        self.slave = slave
        self.begin()

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
            # TODO: rebuild this later when we get the FPS info
            time.sleep(0.001)
        
        return picdata

    @property
    def width(self):
        return self.slave.width

    @property
    def height(self):
        return self.slave.height

instances = {}
def slaveize_camera(sn_field: str, config_field: str):
    """
    Decorator for the camera to slaveize it under master

    Master reads at 30 fps, but slave may be slower!
    """
    def decorator(cls):        
        def getinstance(*args, **kwargs):
            global instances
            cam = cls(*args, **kwargs)
            sn = getattr(cam, sn_field)
            config = getattr(cam, config_field)
            uid = (cls, sn)
            if uid in instances:
                if instances[uid]['config'] != config:
                    instances[uid]['camera'].reload(cam)
                    instances[uid]['config'] = config
            if uid not in instances:
                instances[uid] = {
                    'config': config,
                    'camera': MasterCamera(cam)
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


@slaveize_camera('sn', 'config')
@visualize_camera('sn')
class MockCamera(Camera):
    def __init__(self, config):
        self.sn = '111'
        self.config = config

    def start(self):
        pass

    def stop(self):
        pass

    def capture(self) -> np.ndarray:
        from PIL import Image
        time.sleep(0.02)
        return np.array(Image.new('RGB', (300, 300), color = 'red'))

    @property
    def width(self):
        return 200

    @property
    def height(self):
        return 200


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

