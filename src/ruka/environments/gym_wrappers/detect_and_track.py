import cv2
import gym
import numpy as np
import torch

from copy import deepcopy
from ruka.cv.detect_and_track import DetectAndTrack, TargetObject
from ruka.cv.util import plot_bounding_box
from ruka.observation import Observe
from ruka.util.debug import smart_shape
from typing import Tuple


class DetectAndTrackWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        img_key: str,
        ref_img_condition_key: str,
        detect_and_track: DetectAndTrack,
        save_bbox_img: bool = False,
        device: torch.device = torch.device('cuda:0')):

        self.env = env
        self._detect_and_track = detect_and_track.to(device).eval()
        self._img_key = img_key
        self._save_bbox_img = save_bbox_img
        self._ref_img_condition_key = ref_img_condition_key
        self._ref_img = None

        self.observation_space = deepcopy(self.env.observation_space)

        assert self._img_key in self.observation_space.keys(), (self._img_key, list(self.observation_space.keys()))
        assert self.observation_space[self._img_key].shape == (480, 640, 3), self.observation_space[self._img_key].shape

        max_size = max(*self.observation_space[self._img_key].shape)
        self.observation_space[Observe.TRACKER_OBJECT_BBOX.value] = gym.spaces.Box(low=0, high=max_size, shape=(4,))
        if self._save_bbox_img:
            self.observation_space["tracker_debug_bbox_img"] = deepcopy(self.observation_space[self._img_key])

    def reset(self):
        return self._update_observation(self.env.reset())

    def get_observation(self):
        return self._update_observation(self.env.get_observation())

    def _init_tracker(self):
        self._detect_and_track.reset(TargetObject(reference_image=self._ref_img))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._update_observation(obs), rew, done, info

    def _update_observation(self, obs):
        ref_img = obs[self._ref_img_condition_key]
        if self._ref_img is not ref_img:
            self._ref_img = ref_img
            self._init_tracker()

        assert obs[self._img_key].shape == (480, 640, 3), obs[self._img_key].shape

        img_rgb_hwc = obs[self._img_key]
        box = self._detect_and_track.find(img_rgb_hwc)

        obs[Observe.TRACKER_OBJECT_BBOX.value] = np.array(box.to_axis_aligned().item().to_xyxy(), dtype=np.int32)
        if self._save_bbox_img:
            obs["tracker_debug_bbox_img"] = plot_bounding_box(img_rgb_hwc, box, (255, 0, 0,), 2)

        return obs
