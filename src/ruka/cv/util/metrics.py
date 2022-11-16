def intersect_boxes(box1, box2):
    '''
    Not vectorized version of box intersection.

    Args:
        box1, box2: bounding boxes to intersect in XYXY format

    Returns:
        intersection in XYXY format or zeros if boxes do not intersect
    '''

    res = (max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3]))
    if res[0] > res[2] or res[1] > res[3]:
        return (0, 0, 0, 0)
    return res


def box_area(box):
    '''
    Not vectorized version of box area.

    Args:
        box: bounding box in XYXY format

    Returns:
        box area
    '''

    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    '''
    Not vectorized version of box iou.

    Args:
        box1, box2: bounding boxes to intersect in XYXY format

    Returns:
        box1 and box2 iou: float in [0, 1]
    '''

    intersection = intersect_boxes(box1, box2)
    return box_area(intersection) / (box_area(box1) + box_area(box2) - box_area(intersection))
