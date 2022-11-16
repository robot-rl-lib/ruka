import cv2
import os
import numpy as np

from ruka.util.compression import img2jpg, jpg2img, depth2png, png2depth


depth_img_hwc = None
rgb_img_hwc = None
gray_img_hwc = None


def setup_module(module):
    global depth_img_hwc, rgb_img_hwc, gray_img_hwc

    cur_path = os.path.dirname(os.path.abspath(__file__))
    depth_img_hwc = cv2.imread(os.path.join(cur_path, 'depth.png'), cv2.IMREAD_ANYDEPTH)[:,:,None]
    rgb_img_hwc = cv2.imread(os.path.join(cur_path, 'rgb.jpg'), cv2.IMREAD_ANYCOLOR)[:,:,::-1]
    gray_img_hwc = cv2.cvtColor(rgb_img_hwc, cv2.COLOR_RGB2GRAY)[:,:,None]

def PSNR(original, compressed, max_pixel=255.0):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def test_jpeg_quality95():
    buf = img2jpg(rgb_img_hwc.copy(), quality=95)
    img_enc = jpg2img(buf)
    assert PSNR(rgb_img_hwc, img_enc) > 39, PSNR(rgb_img_hwc, img_enc)

def test_jpeg_quality100():
    buf = img2jpg(rgb_img_hwc.copy(), quality=100)
    img_enc = jpg2img(buf)
    assert PSNR(rgb_img_hwc, img_enc) > 43, PSNR(rgb_img_hwc, img_enc)

def test_jpeg_quality5():
    buf = img2jpg(rgb_img_hwc.copy(), quality=5)
    img_enc = jpg2img(buf)
    assert PSNR(rgb_img_hwc, img_enc) < 30, PSNR(rgb_img_hwc, img_enc)

def test_gray_quality95():
    buf = img2jpg(gray_img_hwc.copy(), quality=95)
    img_enc = jpg2img(buf)
    assert PSNR(gray_img_hwc, img_enc) > 44, PSNR(gray_img_hwc, img_enc)

def test_depth_quality1():
    buf = depth2png(depth_img_hwc.copy(), quality=1)
    depth_enc = png2depth(buf)
    assert PSNR(depth_img_hwc, depth_enc, max_pixel=2**16-1) == 100, \
                    PSNR(depth_img_hwc, depth_enc, max_pixel=2**16-1)
