import numpy as np

from ruka.cv.util import AxisAlignedBBox, OrientedBBox


def check(a, b):
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_axis_aligned_bbox():
    box = AxisAlignedBBox(np.array([100, 150, 200, 350]))

    check(box.to_xyxy(), np.array([100, 150, 200, 350]))
    check(box.to_cxcywh(), np.array([150, 250, 100, 200]))
    check(box.corners, np.array([[100, 150], [100, 350], [200, 350], [200, 150]]))

    box = AxisAlignedBBox(np.array([[100, 150, 200, 350]]))

    check(box.to_xyxy(), np.array([[100, 150, 200, 350]]))
    check(box.to_cxcywh(), np.array([[150, 250, 100, 200]]))
    check(box.corners, np.array([[[100, 150], [100, 350], [200, 350], [200, 150]]]))

    box = AxisAlignedBBox(np.array([[100, 150, 200, 350], [200, 50, 300, 400]]))

    check(box.to_xyxy(), np.array([[100, 150, 200, 350], [200, 50, 300, 400]]))
    check(box.to_cxcywh(), np.array([[150, 250, 100, 200], [250, 225, 100, 350]]))
    check(
        box.corners,
        np.array(
            [[[100, 150], [100, 350], [200, 350], [200, 150]],
            [[200, 50], [200, 400], [300, 400], [300, 50]],
        ]))


def test_oriented_bbox():
    box = OrientedBBox(np.array([100, 150, 200, 350, 20]))

    check(
        box.to_cxcywhr(),
        np.array([100, 150, 200, 350, 20]))
    check(
        box.to_axis_aligned().to_xyxy(),
        np.array([-53.82278, -48.64822, 253.82278, 348.64822,]))
    check(
        box.corners,
        np.array([
            [-53.82278, 19.7558],
            [65.88426, 348.64822],
            [253.82278, 280.24419],
            [134.11573, -48.64822],
        ]))

    box = OrientedBBox(np.array([[100, 150, 200, 350, 20]]))

    check(
        box.to_cxcywhr(),
        np.array([[100, 150, 200, 350, 20]]))
    check(
        box.to_axis_aligned().to_xyxy(),
        np.array([[-53.82278, -48.64822, 253.82278, 348.64822,]]))
    check(
        box.corners,
        np.array([[
            [-53.82278, 19.7558],
            [65.88426, 348.64822],
            [253.82278, 280.24419],
            [134.11573, -48.64822],
        ]]))

    box = OrientedBBox(np.array([[330, 460, 70, 50, -45], [240, 180, 200, 150, 30]]))

    check(
        box.to_cxcywhr(),
        np.array([[330, 460, 70, 50, -45], [240, 180, 200, 150, 30]]))
    check(
        box.to_axis_aligned().to_xyxy(),
        np.array([
            [287.57359, 417.57359, 372.4264, 502.4264,],
            [115.89745, 65.04809, 364.10254, 294.9519,],
        ]))
    check(
        box.corners,
        np.array([
            [
                [322.92893219, 417.57359313],
                [287.57359313, 452.92893219],
                [337.07106781, 502.42640687],
                [372.42640687, 467.07106781],
            ],
            [
                [115.89745962, 165.04809472],
                [190.89745962, 294.95190528],
                [364.10254038, 194.95190528],
                [289.10254038, 65.04809472],
            ],
        ]))
