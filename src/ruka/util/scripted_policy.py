import numpy as np

def get_supported_robot_env(env):
    if hasattr(env, '_actuator') and hasattr(env, '_last_pos'):
        return env
    if hasattr(env, 'envs'):
        return get_supported_robot_env(env.envs[0])
    if hasattr(env, 'env'):
        return get_supported_robot_env(env.env)
    return None


class ScriptedPolicy:
    """ 1. Position above target in XY plate
        2. Moving down to target
        3. Rotate while decrease distance between gripper and target
        4. Try grasp and move up
        5. Goto 3 if grasp fail """

    _stages = ['xy', 'down', 'rot', 'up']
    
    def __init__(self, env,
                xy_threshold: float,
                down_threshold: float,
                rot_threshold: float,
                clip_action: float,
                scale_action: float):
        
        self._xy_threshold = xy_threshold
        self._down_threshold = down_threshold
        self._rot_threshold = rot_threshold
        self._clip_action = clip_action
        self._scale_action = scale_action

        self._step = 0
        self._stage = 0
        self._stage_step = 0

        self._action_shape = env.action_space.shape
        assert self._action_shape == (5,), 'Support only (x, -y, -z, yaw_rot, open/close) action space'

        self._r = get_supported_robot_env(env)
        assert self._r is not None, 'Can not find supported robot env with _actuator and _last_pos attributes'
        
    def act_xy(self, diff):
        """ Position above target in XY plate """
        # (x, -y, -z, yaw_rot, open/close)
        act = np.zeros(self._action_shape, dtype=np.float32)
        # x,y,z
        act[:3] = diff
        act = self._r._actuator._action_scaler.transform([act])[0]
        # yaw rot and gripper
        act[3:] = 0
        # y
        act[1] = - act[1] # because env action ivertion bug
        # z
        act[2] = 0
        # gripper open
        act[-1] = 1
        return act

    def act_down(self, diff):
        """ Moving end-effector down to the target """
        # (x, -y, -z, yaw_rot, open/close)
        act = np.zeros(self._action_shape, dtype=np.float32)
        # x,y,z
        act[:3] = diff
        act = self._r._actuator._action_scaler.transform([act])[0]
        # yaw rot and gripper
        act[ 3:] = 0
        # x, y
        act[:2] = 0
        # z 
        act[2] = -act[2] # because env action ivertion bug
        # gripper open
        act[-1] = 1
        return act

    def act_up(self, diff):
        """ Grasp and moveing up """
        # (x, -y, -z, yaw_rot, open/close)
        act = np.zeros(self._action_shape, dtype=np.float32)
        # x,y,z
        act[:3] = diff
        act = self._r._actuator._action_scaler.transform([act])[0]
        # z
        act[ 2] = -1
        # close gripper
        act[-1] = -1
        return act
        
    def get_action(self, _):
        rob_xyz = np.array(self._r._last_pos['pose'][0])
        tgt_xyz = np.array(self._r._last_pos['target'][0])
        
        # distance between robot gripper and target object
        diff = tgt_xyz - rob_xyz
        
        if self._stages[self._stage] == 'xy':
            act = self.act_xy(diff)
            xy_norm = np.linalg.norm(diff[:2])
            if xy_norm < self._xy_threshold:
                self._stage +=1
                self._stage_step = 0

        elif self._stages[self._stage] == 'down':
            act = self.act_down(diff)
            if np.abs(diff[2]) < self._down_threshold:
                self._stage +=1
                self._stage_step = 0

        elif self._stages[self._stage] == 'rot':
            # z - down, rotation and gripper open
            act = np.array([0,0,1,1,1], dtype=np.float32)
            if np.abs(diff[2]) < self._rot_threshold:
                self._stage +=1
                self._stage_step = 0

        elif self._stages[self._stage] == 'up':
            act = self.act_up(diff)
            if self._stage_step > 1 and not self._r._actuator.object_detected():
                self._stage -= 1
                self._stage_step = 0
            
        self._stage_step += 1
        if self._clip_action:
            act = np.clip(-self._clip_action,self._clip_action,act)
        return act * self._scale_action
    
    def reset(self):
        self._step = 0
        self._stage = 0
        self._stage_step = 0