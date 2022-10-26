import os
import cv2
import gym
import numpy as np

class FrameSkip(gym.Wrapper):
    """ Skip frames """

    def __init__(self, env, frame_skip):
        self.env = env
        self.frame_skip = frame_skip

    def step(self, action):
        total_rew = 0
        for i in range(self.frame_skip):
            obs, rew, done, info = self.env.step(action)
            total_rew += rew
            if done:
                break

        return obs, total_rew, done, info


class RecordVideo(gym.Wrapper):
    
    def __init__(self, env, keys, size, path, prefix='', tb=None, each=1):
        self.env = env
        self.keys = keys if isinstance(keys, list) else [keys]
        self.size = size
        self.prefix = prefix
        self.path = path
        self.tb = tb
        self.each = each

        self._video = None
        self._frames = []
        self._ep_num = 0
        os.makedirs(self.path, exist_ok=True)

    def _make_video(self):
        name = os.path.join(self.path, f"{self.prefix}{self._ep_num:04d}.avi")
        print(f'Make video {name} with {len(self._frames)}...', flush=True)
        video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MJPG'), 10, self._frames[0].shape[:2][::-1])
        for frame in self._frames:
            video.write(frame)
        video.release()
        if self.tb is not None:
            self.tb.add_video('videos', np.array(self._frames).transpose(0, 3, 1, 2)[None].astype(np.uint8))     
        print('Make video done', flush=True)

    def reset(self):
        obs = self.env.reset()

        if self._frames:
            self._make_video()
        self._frames = []
        self._ep_num += 1
        self._add_frame(obs)
        return obs

    def _add_frame(self, obs):
        if self._ep_num % self.each != 0:
            return
        imgs = []
        for k in self.keys:
            img = cv2.resize(obs[k], self.size)
            if len(img.shape ) == 2:
                img = np.stack([img,img,img], axis=-1)
            imgs.append(img)
        frame = np.concatenate(imgs, axis=1)
        self._frames.append(frame)


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._add_frame(obs)
        return obs, rew, done, info
