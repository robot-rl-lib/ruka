import os
import shutil
import tempfile

import pybullet as p
import pybullet_data
import ruka.logging.scene_log as scene_log
import ruka.logging.episode as episode_logger

from pybullet_utils import bullet_client


LOG_IN_CWD = True


"""
Build Video using PyBullet from stored scene
 - setup pybullet
 - load objects and positions
 - generate frame from camera
 - make a local epsidoe logger (to local folder)
 - put the frames to logger
 - build video
"""
class VideoUsingPyBullet:

    def __init__(self, debug = False):
        self.debug = debug
        self.set_camera_options()

    def load_scene(self, path):        
        self.path = path
        self.scene = scene_log.RukaSceneLog()
        if not self.scene.load(self.path):
            return False
        return True

    def set_camera_options(self, fov = None, near = None, far = None,
                           view_eye = None, view_target = None, view_up = None,
                           view_distance = None, view_yaw = None, view_pitch = None, view_roll = None, view_up_axis = None,
                           follow_object = None ):

        self.fov           = 60 if fov is None else fov
        self.near          = 0.02 if near is None else near
        self.far           = 1.5 if far is None else far

        self.view_eye      = [0.0, 0.0, 0.5] if view_eye is None else view_eye
        self.view_target   = [0.0, 0.0, 0.0] if view_target is None else view_target
        self.view_up       = [1,0,0] if view_up is None else view_up

        self.view_distance = 0 if view_distance is None else view_distance
        self.view_yaw      = 0 if view_yaw is None else view_yaw
        self.view_pitch    = 0 if view_pitch is None else view_pitch
        self.view_roll     = 0 if view_roll is None else view_roll
        self.view_up_axis  = 2 if view_up_axis is None else view_up_axis
        self.follow_object = follow_object
  
    def build_video(self, format, width, height, outfile, from_state, tot_frames = -1, framerate = -1):
        positions = self.scene.decompress_positions(tot_frames)
        joints    = self.scene.decompress_joints(tot_frames)
       
        logdir = tempfile.mkdtemp(dir=os.getcwd() if LOG_IN_CWD else False, prefix="ivd_")

        self.width = width
        self.height = height

        self.episode_logger = episode_logger.create_episode_logger(logdir, no_log=True)
        self.episode_logger.set_video_option('framerate',framerate if framerate!=-1 else self.scene.get_timerate())

        # iterate through frames and log them one-by-one
        i = 1
        tot = len(positions)
        for i in range(1,tot+1):
        
            print(f"Step {i} out of {tot} running")

            pos = positions[i-1]
            jl  = joints[i-1]

            # state if needed and was stored by logger
            state = False
            if from_state:  
                state = self.scene.decompress_file(f'scene_{i:05d}.bullet')
                #print(state)            

            # run the code
            self.setup_frame(pos, jl, state)
            print(f"Step {i} out of {tot} done")
            i += 1
        
        self.episode_logger.close(episode_steprate = self.scene.get_timerate())
        #print(self.episode_logger.store_data)
        videofile = os.path.join(self.episode_logger.path, self.episode_logger.DATA['']['video']['maincam']['astrotime']['video'])
        
        if not os.path.isfile(videofile):
            print(f'Video file not generated: {videofile}')
            return False
        shutil.copyfile(videofile, outfile)

        if not self.debug:
            shutil.rmtree(logdir)

        return True        
        
    def setup_frame(self, pos, jl, state = False):

        
        #cin = p.connect(p.SHARED_MEMORY)
        #if (cin < 0):
        #   cin = p.connect(p.GUI)
        #objects = [p.loadURDF("/home/technocore/arcadia/ytech/cobots/lib/ruka/src/ruka/data/models/table/table.urdf", 0.000000,0.000000,-0.820000,0.000000,0.000000,0.000000,1.000000)]            
        #p.setGravity(0.000000,0.000000,-9.810000)
        #p.stepSimulation()
        #p.disconnect()

        #physics_client = bullet_client.BulletClient(p.GUI)
        physics_client = bullet_client.BulletClient(p.DIRECT)
        physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())        
        
        follow_obj = False

        if state:
            raise NotImplementedError("pybullet load state is not working right")
            #physics_client.restoreState(fileName=state)
            #physics_client.loadBullet(bulletFileName=state)
        else:
            for x in pos:
                obj_num = f'{x[0]}'
                obj = self.scene.info['objects'][obj_num]           
                if obj['type'] == 'urdf':
                    
                    bp = x[1][0:3]
                    bo = x[1][3:]
                    #r = physics_client.loadURDF(obj['file'], basePosition=bp, baseOrientation=bo)
                    r = physics_client.loadURDF(os.path.join(self.path,obj['locfile']), basePosition=bp, baseOrientation=bo)                    


                    for j in range(physics_client.getNumJoints(int(obj_num))):
                        f = physics_client.getJointInfo(int(obj_num),j)
                        #print(f)

                    if self.follow_object == int(obj_num):
                        follow_obj = {
                            'pos': bp,
                            'or':  bo
                        }
                else:
                    raise NotImplementedError
                        
            if jl:
                for t in jl:
                    physics_client.resetJointState(t[0], t[1], t[2])
                    #physics_client.resetJointState(t['i'], t['j'], t['jp'])
        
        width = self.width
        height = self.height
        aspect = width / height

        if not follow_obj:
            if self.view_distance > 0:
                view_matrix = p.computeViewMatrixFromYawPitchRoll(self.view_target, self.view_distance, self.view_yaw, self.view_pitch, self.view_roll, self.view_up_axis)
            else:            
                view_matrix = p.computeViewMatrix(self.view_eye, self.view_target, self.view_up)
        else:

            eye =  [v_i + w_i for v_i, w_i in zip(self.view_eye,bp)]
            target =  [v_i + w_i for v_i, w_i in zip(self.view_target,eye)]            
            print(eye)
            print(target)
            print(self.view_up)
            view_matrix = p.computeViewMatrix(eye, target, self.view_up)

        projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
        pic = physics_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER)

        self.episode_logger.add_video_frame('maincam', pic, width=width, height=height)  
        self.episode_logger.step()

