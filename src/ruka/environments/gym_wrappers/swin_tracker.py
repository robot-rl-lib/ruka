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
                       img_key: str,
                       ref_img_condition_key: str,
                       init_bbox: Tuple,
                       save_bbox_img: bool = False):

        self.env = env
        self._tracker = SingleObjectSwinTrack().cuda()
        self._img_key = img_key
        self._save_bbox_img = save_bbox_img
        self._ref_img_condition_key = ref_img_condition_key
        self._init_bbox = init_bbox
        self._step = 0

        self.observation_space = deepcopy(self.env.observation_space)

        assert self._img_key in self.observation_space.keys(), (self._img_key, list(self.observation_space.keys()))
        assert self.observation_space[self._img_key].shape == (480, 640, 3), self.observation_space[self._img_key].shape

        # add new fields to observation space
        max_size = max(*self.observation_space[self._img_key].shape)
        self.observation_space[Observe.TRACKER_OBJECT_BBOX.value] = gym.spaces.Box(low=0, high=max_size, shape=(4,))
        if self._save_bbox_img:
            self.observation_space["tracker_debug_bbox_img"] = deepcopy(self.observation_space[self._img_key])

    def _update_observation_with_tracker(self, obs):
        assert obs[self._img_key].shape == (480, 640, 3), obs[self._img_key].shape

        img_rgb_hwc = obs[self._img_key]
        box = self._tracker(img_rgb_hwc)

        obs[Observe.TRACKER_OBJECT_BBOX.value] = np.array(box, dtype=np.uint16)
        if self._save_bbox_img:
            obs["tracker_debug_bbox_img"] = cv2.rectangle(img_rgb_hwc.copy(), (box[0], box[1]),(box[2], box[3]), (255, 0, 0), thickness=2)
        return obs


    def reset(self):
        self._step = 0
        obs = self.env.reset()


        ref_imf = obs[self._ref_img_condition_key]

        #calc bbox. we scaled it on 1.8 before
        x_0 = int(ref_imf.shape[1] * 0.22)
        y_0 = int(ref_imf.shape[0] * 0.22)
        x_1 = ref_imf.shape[1] - x_0
        y_1 = ref_imf.shape[0] - y_0
        ref_box = (x_0, y_0, x_1, y_1)

        # init tracker from ref image
        self._tracker.initialize(ref_imf, ref_box)

        # cv2.imwrite(f't_ref.jpg', cv2.rectangle(ref_imf[:,:,::-1].copy(), tuple(ref_box[:2]), tuple(ref_box[-2:]), (255,0,0), thickness=2))

        img_rgb_hwc = obs[self._img_key]
        box = self._tracker(img_rgb_hwc, box=self._init_bbox)

        # cv2.imwrite(f'tt.jpg', cv2.rectangle(img_rgb_hwc[:,:,::-1].copy(), tuple(box[:2]), tuple(box[-2:]), (255,0,0), thickness=2))

        # reinit tracker from real frame
        self._tracker.initialize(img_rgb_hwc, box)

        return self._update_observation_with_tracker(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._step +=1
        return self._update_observation_with_tracker(obs), rew, done, info

