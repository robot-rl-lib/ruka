import fcntl, sys, os
import numpy as np

from .robot import Camera
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

    def __init__(self, orig_cam, devid):
        self._orig_cam = orig_cam
        width = self._orig_cam.width
        height = self._orig_cam.height

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
        buff = ConvertToYUYV(self.format.fmt.pix.sizeimage, self.format.fmt.pix.bytesperline, img)
        self.device.write(buff)