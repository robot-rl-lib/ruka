import copy
import cv2
import torch
import numpy as np
from typing import Callable, Optional, Dict, Union


def viz_batch_img(batch: Dict,
                  process_act_fn: Callable = lambda x: x,
                  process_bbox_fn: Optional[Callable] = None,
                  process_point_fn: Optional[Callable] = None,
                  process_pos_fn: Optional[Callable] = None,
                  process_gripper_fn: Optional[Callable] = None,
                  resize: int = 256,
                  img_num: int = 9,
                  obs_k: str = 'observation',
                  act_k: str = 'action',
                  img_k: str = 'rgb',
                  gripper_k: str = 'gripper',
                  pos_k: str = 'robot_pos',
                  bbox_k: str = 'tracker_object_bbox',
                  point_k: str = 'bb_center',
                 ) -> np.ndarray:
    """ Visualize batch in one image. 
        For each modality need to use preprocess function for scale values
        Can plot:
            - actions
            - bbox
            - point
            - robot_pos
            - gripper
        Return image (H,W,3) with img_num tiles
    """

    # imgs prepara
    imgs = make_nparray(batch[obs_k][img_k][:img_num, 0])
    if imgs.shape[-1] == 1:
        # grayscale -> fake rgb
        imgs = np.tile(imgs, (1, 1, 1, 3))
    imgs = [img for img in (imgs * 255).astype(np.uint8).copy()]
    
    
    # draw bbox
    if process_bbox_fn is not None:
        bboxs = make_nparray(batch[obs_k][bbox_k][:img_num, 0])
        for img, bbox in zip(imgs, bboxs):
            bbox = process_bbox_fn(bbox)
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[-2:]), (255, 0, 0), thickness=2)

    # draw point
    if process_point_fn is not None:
        points = make_nparray(batch[obs_k][point_k][:img_num, 0])
        for img, point in zip(imgs, points):
            point = process_point_fn(point)
            cv2.circle(img, tuple(point), 2,  (255, 0, 0), thickness=1)
            
    if resize:
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (resize, resize), interpolation=cv2.INTER_LANCZOS4)
    
    # draw actions
    acts = make_nparray(batch[act_k][:img_num, 0])
    for i in range(len(imgs)):
        act = process_act_fn(acts[i])
        imgs[i] = _draw_action(imgs[i], act)
        
    # draw pos and gripper
    if process_pos_fn is not None and process_gripper_fn is not None:
        poss = make_nparray(batch[obs_k][pos_k][:img_num, 0])
        grps = make_nparray(batch[obs_k][gripper_k][:img_num, 0])
        for i in range(len(imgs)):
            pos = process_pos_fn(poss[i])
            grp = process_gripper_fn(grps[i])
            imgs[i] = _draw_pos_gripper(imgs[i], pos, grp)
    
    return _make_grid(imgs)

def get_batch_statistics(batch: Dict,
                  obs_k: str = 'observation',
                  act_k: str = 'action',
                  pos_k: str = 'robot_pos',
                  nbins: int = 20,
                 ) -> Dict[str, np.ndarray]:
    """ Computes basic batch statistics
    returns keys of three types:
    key_img - image-like key
    key_mean, key_std - scalars
    key_hist - data for histogram plotting
    """
    out_dict = dict()
    batch = copy.deepcopy(batch)
    if '_sequence' in obs_k:
        for k, v in batch[obs_k].items():
            batch[obs_k][k] = v[:, 0, ...]
        batch[act_k] = batch[act_k][:, 0, ...]
    if pos_k is not None:
        pos = make_nparray(batch[obs_k][pos_k])
        out_dict.update(_make_pos_vis(pos, nbins))
    for k, v in batch[obs_k].items():
        out_dict.update(_vector_batch_stats(k, v, nbins))
    out_dict.update(_vector_batch_stats('action', batch[act_k], nbins))
    return out_dict

def make_nparray(v: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """ Make np array from array or tensor """

    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    raise ValueError(f'Not supported type {type(v)}')


def resize_bbox(bbox: np.ndarray, in_size: tuple, out_size: tuple) -> np.ndarray:
    """Resize bounding boxes according to image resize.
    Args:
        bbox: XYXY
        in_size: HW
        out_size HW
    Returns:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[..., 0] = x_scale * bbox[..., 0]
    bbox[..., 1] = y_scale * bbox[..., 1]
    bbox[..., 2] = x_scale * bbox[..., 2]
    bbox[..., 3] = y_scale * bbox[..., 3]
    return bbox


def resize_point(point: np.ndarray, in_size: tuple, out_size: tuple) -> np.ndarray:
    """Resize point according to image resize.
    Args:
        point: XY
        in_size: HW
        out_size HW
    Returns:
        point rescaled according to the given image shapes.
    """
    point = point.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    point[..., 0] = x_scale * point[..., 0]
    point[..., 1] = y_scale * point[..., 1]
    return point


def _frmt(a):
    if abs(a) > 1:
        return f'{int(a):<3}'
    else:
        return f'{float(a):.1f}'

    
def _draw_action(img, action):
    pad_img = np.zeros((128, 256, 3), dtype=np.uint8)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (200, 200, 200)
    thickness = 1
    pad_img = cv2.putText(pad_img, 'acts:', (100, 20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        
    xyz_act = action['tool_vel']['xyz']
    xyz_act_txt = f"x:{_frmt(xyz_act[0])} y:{_frmt(xyz_act[1])} z:{_frmt(xyz_act[2])}"
    pad_img = cv2.putText(pad_img, xyz_act_txt, (20, 50), font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    
    rpy_act = action['tool_vel']['rpy']
        
    rpy_act_txt = f"r:{_frmt(rpy_act[0])} p:{_frmt(rpy_act[1])} y:{_frmt(rpy_act[2])}"
    pad_img = cv2.putText(pad_img, rpy_act_txt, (20, 80), font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    
    gipper_txt = f"g: {action['gripper']:<4}"
    pad_img = cv2.putText(pad_img, gipper_txt, (100, 110), font, 
                      fontScale, color, thickness, cv2.LINE_AA)
    
    if img.shape[1] != pad_img.shape[1]:
        pad_img = cv2.resize(pad_img, (img.shape[1], img.shape[0] // 2), interpolation=cv2.INTER_LANCZOS4)
        
    return np.concatenate((img, pad_img), axis=0)


def _draw_pos_gripper(img, pos, gipper):
    side_img = np.zeros((256 + 128, 128, 3), dtype=np.uint8)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (200, 200, 200)
    thickness = 1
    side_img = cv2.putText(side_img, 'pos:', (20, 20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        
    for i, (p, name) in enumerate(zip(pos,'xyzrpy')):        
        txt = f"{name}:{_frmt(p)}"
        side_img = cv2.putText(side_img, txt, (20, 60 + i * 40), font, 
                           fontScale, color, thickness, cv2.LINE_AA)

    txt = f"g:{_frmt(gipper)}"
    side_img = cv2.putText(side_img, txt, (20, 60 + 6 * 40), font, 
                       fontScale, color, thickness, cv2.LINE_AA)        
        
    if img.shape[1] != side_img.shape[1] * 2:
        side_img = cv2.resize(side_img, (img.shape[1] // 2, img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
    return np.concatenate((img, side_img), axis=1)    


def _make_grid(imgs):
    rows = int(np.sqrt(len(imgs)))
    cols = len(imgs) // rows
    if rows * cols < len(imgs):
        cols += 1
    pad = rows * cols - len(imgs)

    lines = []
    for r in range(rows):
        lines.append(imgs[cols * r: cols * (r + 1)])
        if r == rows-1 and pad:
            pad_img = np.zeros(imgs[0].shape, dtype=np.uint8)
            lines[-1] = lines[-1] + [pad_img] * pad 
    
        lines[-1] = np.concatenate(lines[-1], axis=1)
    return np.concatenate(lines, axis=0)
    

def _vector_batch_stats(k, v, nbins) -> Dict[str, np.ndarray]:
    out_dict = dict()
    shape = v.shape
    if len(shape) == 2:
        size = shape[1]
        if size == 1:
            out_dict[k + '_hist'] = v
            out_dict[k + '_mean'] = np.mean(v)
            out_dict[k + '_std'] = np.std(v)
        else:
            if size==2:
                out_dict[k + '_img'] = np.histogram2d(v[:, 0], v[:, 1], bins=(nbins, nbins))[0]
            for i in range(size):
                new_k = f'{k}[{i}]'
                out_dict[new_k + '_hist'] = v[:, i]
                out_dict[new_k + '_mean'] = np.mean(v[:, i])
                out_dict[new_k + '_std'] = np.std(v[:, i])
    return out_dict

def _make_pos_vis(pos, nbins) -> Dict[str, np.ndarray]:
    out = dict()
    x_s = pos[:, 0]
    y_s = pos[:, 1]
    z_s = pos[:, 2]
    out['xy_dots_img'] = np.histogram2d(x_s, y_s, (nbins, nbins))[0]
    out['yz_dots_img'] = np.histogram2d(y_s, z_s, (nbins, nbins))[0]
    out['xz_dots_img'] = np.histogram2d(x_s, z_s, (nbins, nbins))[0]
    return out
