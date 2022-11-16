import gym
import cv2
import numpy as np
from copy import deepcopy
from typing import Tuple
from ruka.observation import Observe
from ruka.cv.tracking.single_object_swintrack import SingleObjectSwinTrack
from ruka.util.debug import smart_shape


class SingleObjectSwinTrackWrapper(gym.Wrapper):

    def __init__(self, env,
                       ref_img_rgb_hwc: np.ndarray, 
                       ref_img_box: Tuple[float],
                       img_key: str = Observe.RGB.value,
                       save_bbox_img: bool = False):
        self.env = env
        self._tracker = SingleObjectSwinTrack().cuda()
        self._ref_img_rgb_hwc = ref_img_rgb_hwc
        self._ref_img_box = ref_img_box
        assert len(ref_img_box) == 4, len(ref_img_box)
        self._img_key = img_key
        self._save_bbox_img = save_bbox_img

        self._step = 0
        
        self.observation_space = deepcopy(self.env.observation_space)

        assert self._img_key in self.observation_space.keys(), (self._img_key, list(self.observation_space.keys()))
        assert self.observation_space[self._img_key].shape == (3, 480, 640), self.observation_space[self._img_key].shape

        # add new fields to observation space
        max_size = max(*self.observation_space[self._img_key].shape)
        self.observation_space[Observe.TRACKER_OBJECT_BBOX.value] = gym.spaces.Box(low=0, high=max_size, shape=(4,))
        if self._save_bbox_img:
            self.observation_space["tracker_debug_bbox_img"] = deepcopy(self.observation_space[self._img_key])

    def _update_observation_with_tracker(self, obs):
        assert obs[self._img_key].shape == (3, 480, 640), obs[self._img_key].shape

        img_rgb_hwc = obs[self._img_key].transpose((1,2,0))
        if self._step == 0:
            self._tracker.initialize(self._ref_img_rgb_hwc, self._ref_img_box)
            box = self._tracker(img_rgb_hwc)
        else:
            box = self._tracker(img_rgb_hwc)

        obs[Observe.TRACKER_OBJECT_BBOX.value] = np.array(box, dtype=np.uint16)
        if self._save_bbox_img:
            img_bbox_rgb_hwc = cv2.rectangle(img_rgb_hwc.copy(), (box[0], box[1]),(box[2], box[3]), (255, 0, 0), thickness=2)
            obs["tracker_debug_bbox_img"] = img_bbox_rgb_hwc.transpose((2,0,1))   
        return obs


    def reset(self):
        self._step = 0
        obs = self.env.reset()
        return self._update_observation_with_tracker(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._step +=1
        return self._update_observation_with_tracker(obs), rew, done, info

