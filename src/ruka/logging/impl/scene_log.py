import io
import json
import os
import shutil

from pybullet_utils import bullet_client
from pybullet_utils import urdfEditor as ed
import ruka.logging.video_and_scene_utils as scene_utils
from xml.dom import WrongDocumentErr


"""
Engine to dump the RUKA sceme to a folder
whcih later could be extracted and rebuilt:
 - for building a video
 - for taking a screenshot of a frame
 - re-setup the pybullet scene
"""
class RukaSceneLog:
    def __init__(self, physical_engine = False, timerate = False, dont_localize = False):
        self.objects          = {}
        self.object_num       = len(self.objects)
        self.positions        = []
        self.joints_and_links = []
        self.prev_pos = {
            'objects': {},
            'links': {}
        }
        self.magic            = 'RSLF'
        self.version          = 1        
        self.format           = 'text' # currently supported only this one
        self.physical_engine  = physical_engine
        self.timerate         = timerate
        self.dont_localize    = dont_localize

    def log_object(self, objfile, objtype, temp):
        """
        Log a new object in the list of all objects:
        - unique object identified by its file location
        - num of the object is incremented from 0 (same as for pybullet!)

        Returns the num of the object
        """
        if not self.objects.get(objfile):
            on = self.object_num
            self.objects[objfile] = {
                'file': objfile,
                'type': objtype,
                'num' : self.object_num,
                'temp': temp
            }
            self.object_num = len(self.objects)
            return on
        return self.objects[objfile]['num']
            
    def log_scene(self, objects, joints_and_links):
        """
        Log the "full" scene to list:
         - positions - new coordinates of each object
         - joints - new values for each joint

         Add new object if it didn't existed in previous frame
        """
        # log objects if they are not already logged
        map = {}
        for k in range(len(objects)):
            num      = self.log_object(objects[k]['file'], objects[k]['type'], objects[k].get('temp'))
            map[num] = k

        # check new positions
        new_ps = {
            'objects': {},
            'links': {}
        }
        store_ps = []

        # compute new positions and compress
        for k in map:
            changed = False
            if not self.prev_pos['objects'].get(k):
                changed = True
            else:
                if self.prev_pos['objects'][k]['pos'] != objects[map[k]]['pos']:
                    changed = True
            new_ps['objects'][k] = {
                'pos': objects[map[k]]['pos']
            } 
            if changed:
                v = [k]
                v.append(objects[map[k]]['pos'])
                store_ps.append(v)
            else:
                store_ps.append([k])        
        self.positions.append(store_ps)
        
        # compute new joint values and compress
        store_jl = []
        for x in joints_and_links:
            #print(x)
            k = x['i'],x['j']
            changed = False
            if not self.prev_pos['links'].get(k):
                changed = True
            else:
                if self.prev_pos['links'][k] != x['jp']:
                    changed = True
            new_ps['links'][k] = x['jp']
            if changed:
                store_jl.append([x['i'],x['j'],x['jp']])
            else:
                store_jl.append([x['i'],x['j']])
        self.joints_and_links.append(store_jl)

        # store new positions for next step
        self.prev_pos = new_ps

    def type_to_ext(self,type):
        """
        Obtain the file extention for a given type of the 3D model        
        """
        if type == 'urdf':
            return 'urdf'
        raise NotImplementedError

    def dump(self, path):
        """
        Dump everything to folder
        """
        info = {}
        info['magic']    = self.magic
        info['version']  = self.version
        info['format']   = self.format
        info['render']   = self.physical_engine
        info['timerate'] = self.timerate
 
        # store objects on the disk in the folder
        info['objects'] = {}
        d_obj = os.path.join(path, 'objects')
        os.mkdir(d_obj)
        for x in self.objects:            
            d = self.objects[x]
            filen = d['num']
            ext = self.type_to_ext(d['type'])            
            if self.dont_localize:
                shutil.copyfile(d['file'], os.path.join(d_obj,f'obj_{filen}.{ext}'))
            else:                
                scene_utils.localize_urdf(d['file'], d_obj, f'obj_{filen}.{ext}', f'geom_{filen}')
            info['objects'][filen] = d        
            info['objects'][filen]['locfile'] = f'objects/obj_{filen}.{ext}'            
            if d.get('temp'):
                os.remove(d['file'])
        self.info = info
        
        # store some common info about the scene
        with open(os.path.join(path, 'info.json'), 'w') as f:
            json.dump(info, f)
        
        # dump positions and joints
        if self.format == 'text':            
            # store directly JSON files
            # this is easier, but not so clean and may waste some place
            # enyway this is fast, add new format later when we will needed it
            with open(os.path.join(path, 'positions.json'), 'w') as f:
                json.dump(self.positions, f)

            with open(os.path.join(path, 'joints_and_links.json'), 'w') as f:
                json.dump(self.joints_and_links, f)
        else:
            raise NotImplementedError

    def get_dump_file(path, name):
        return os.path.join(path, name)

    def get_timerate(self):
        return self.timerate

    def load(self, path):

        # store path from load
        self.path = path

        # load config
        i = os.path.join(self.path, 'info.json')
        if not os.path.isfile(i):
            print(f"File not found: {i}")
            return False
        with open(i) as f:
            d = json.load(f)            
        self.info = d
        if not self.info.get('objects'):
            print("Bad objects engine")
            return False

        if self.info['magic'] != 'RSLF':
            raise WrongDocumentErr

        if self.info['format'] != 'text':
            raise NotImplementedError

        if self.info['version'] != 1:
            raise NotImplementedError

        # Currently work onl with text format
        self.timerate = self.info['timerate']
        
        # load positions
        p = os.path.join(self.path, 'positions.json')
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            return False
        with open(p) as f:
            d = json.load(f)            
        self.positions = d

        # load joints and state
        p = os.path.join(self.path, 'joints_and_links.json')
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            return False
        with open(p) as f:
            d = json.load(f)            
        self.joints_and_links = d
        
        # check objects if file exists and etc
        for x in self.info['objects']:
            r = self.info['objects'][x]
            self.info['objects'][x]['thefile'] = os.path.join(self.path, r['locfile'])
            if not os.path.isfile(self.info['objects'][x]['thefile']):
                rr = self.info['objects'][x]['thefile']
                print(f'File not found: {rr}')
                return False

        return True

    def decompress_position(self, prev, pos):
        new_pos = {}
        the_pos = []
        for t in pos:
            obj_num  = t[0]
            pp = False
            if len(t)>1:
                pp = t[1]
            if prev.get(obj_num):
                if len(t) == 1:
                    pp = prev[obj_num]['pos']            
            if pp is False:
                raise BadPositionCompression
            the_pos.append([obj_num, pp])
            new_pos[obj_num] = {
                'num': obj_num,
                'pos': pp
            }        
        return new_pos, the_pos

    def decompress_joint(self, prev, jl):
        if not self.joints_and_links:
            return False
        new_jl = {}
        the_jl = []

        for t in jl:
            key  = t[0],t[1]
            pp = False
            if len(t)>2:
                pp = t[2]
            if prev.get(key):
                if len(t) == 2:
                    pp = prev[key]            
            if pp is False:
                print(t)
                print(prev)
                raise BadJointLinkCompression
            the_jl.append([t[0],t[1],pp])
            new_jl[key] = pp        
        return new_jl, the_jl

    def decompress_positions(self, limit = -1):
        positions = []
        prev_pos = {}        
        i = 1
        tot = len(self.positions) if limit == -1 else limit       
        for x in self.positions:      
            if i > tot:
                break
            prev_pos, pos = self.decompress_position(prev_pos, x) 
            positions.append(pos)
            i += 1
        return positions
        
    def decompress_joints(self, limit = -1):
        if not self.joints_and_links:
            return []
        prev_js = {} 
        joints = []       
        i = 1
        tot = len(self.joints_and_links) if limit == -1 else limit       
        for x in self.joints_and_links:      
            if i > tot:
                break
            prev_js, jl = self.decompress_joint(prev_js, x) 
            joints.append(jl)
            i += 1
        return joints

    def decompress_file(self, name):
        return os.path.join(self.path, name)


#
# Get objects and joints
#
def phys_get_objects_and_joins(phys, path):
    if not isinstance(phys, bullet_client.BulletClient):
        raise NotImplementedError
    obj    = pybullet_objects_to_ruka_scene_log(path, phys)
    joints = pybullet_joint_and_links_to_ruka_scene_log(path, phys)
    return obj, joints

#
# Some debug info to store
#
def phys_store_debug(phys, file):
    if not isinstance(phys, bullet_client.BulletClient):
        raise NotImplementedError    
    if False:
        with open(f, 'a') as file:            
            for i in range(phys.getNumBodies()):
                pos, orn = phys.getBasePositionAndOrientation(i)
                linVel, angVel = phys.getBaseVelocity(i)
                txtPos = "  pos=" + str(pos) + "\n"
                txtOrn = "  orn=" + str(orn) + "\n"
                txtLinVel = "    linVel" + str(linVel) + "\n"
                txtAngVel = "    angVel" + str(angVel) + "\n"
                infod = phys.getVisualShapeData(i)
                info = print_to_string(infod)
                file.write("\n\n===============\n")
                file.write(txtPos)
                file.write(txtOrn)
                file.write(txtLinVel)
                file.write(txtAngVel)
                file.write(info)
        phys.saveBullet(f)

#
# Text name of the physical engine
#
def phys_name(phys):
    if not isinstance(phys, bullet_client.BulletClient):
        raise NotImplementedError    
    return "pybullet"

#
# Text name of the physical engine
#
def phys_timerate(phys):
    return 240

#
# Pybullet methods
#
def pybullet_scene_to_ruka_scene_log(phys, memfile, lines):
    """
    Old method through saveWorld!
    Not working as we dont have joint values
    """
    phys.saveWorld(memfile)
    with open(memfile) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    sc = []
    for x in lines:
        m = re.search(r'load([A-Z0-9]+)\([\'"]?([^\'"]+)[\'"]\s?,\s?([^\)]+)\)', x)
        if m:
            #print('TYPE', m.group(1))
            #print('FILE', m.group(2))
            #print('POS', m.group(3))
            pos = []
            e = m.group(3).split(',')
            for a in e:
                pos.append(float(a))

            sc.append({
                'file': m.group(2),
                'type': m.group(1).lower(),
                'pos': pos
            })
    return sc

def pybullet_objects_to_ruka_scene_log(path, p):
    """
    Get pybullet objects and the positions
    """
    sc = []        
    for i in range(p.getNumBodies()):
        pos, orn = p.getBasePositionAndOrientation(i)        
        t = os.path.join(path, f'step_shape_{i}.urdf')
        tempfile = True            
        if not os.path.isfile(t):
            ed0 = ed.UrdfEditor()
            ed0.initializeFromBulletBody(i,p._client)
            ed0.saveUrdf(t)
        sc.append({
                'file': t,
                'type': 'urdf',
                'pos': [pos[0],pos[1],pos[2],orn[0],orn[1],orn[2],orn[3]],
                'temp': tempfile
            })                     
        
    return sc
    
def pybullet_joint_and_links_to_ruka_scene_log(path, p):
    """
    Get pybullet joints values
    """
    jl = []                
    for i in range(p.getNumBodies()):
        for j in range(p.getNumJoints(i)):
            jj = p.getJointState(i,j)
            jp = jj[0]
            #ll = p.getLinkState(i,j)
            #lp = ll[0:3]
            #lo = ll[3:7]
            jl.append({
                    'i': i,
                    'j': j,
                    'jp': jp
                    #'lp': lp,
                    #'lo': lo
                })                             
    return jl
    
    

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents
