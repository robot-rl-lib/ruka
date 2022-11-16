import numpy as np


def cut_box_and_make_square(image, box, expand_k=1.0, min_side=None):
    half_side = int(max(box[2] - box[0], box[3] - box[1]) * expand_k / 2)
    if min_side is not None:
        half_side = max(half_side, min_side // 2)

    c_x, c_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x_min, y_min = max(0, c_x - half_side), max(0, c_y - half_side)
    x_max, y_max = min(image.shape[1], c_x + half_side + 1), min(image.shape[0], c_y + half_side + 1)

    result = image[y_min:y_max, x_min:x_max]

    sides_diff = abs(result.shape[0] - result.shape[1])

    if sides_diff > 0:
        pad = (sides_diff // 2, sides_diff - sides_diff // 2,)
        if result.shape[0] > result.shape[1]:
            result = np.pad(result, ((0, 0,), pad, (0, 0,)), mode='edge')
            x_min -= pad[0]
        else:
            result = np.pad(result, (pad, (0, 0,), (0, 0,),), mode='edge')
            y_min -= pad[0]

    assert result.shape[0] == result.shape[1]

    return result, (x_min, y_min)
