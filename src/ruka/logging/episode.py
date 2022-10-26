import datetime
import glob
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import uuid
import urllib

import numpy as np
import ruka.logging.scene_log as scene_log
import ruka.logging.video_and_scene_utils as video_and_scene_utils

from io import FileIO
from PIL import Image
from ruka_os import s3
from xml.dom import WrongDocumentErr


class EpisodeLogger:
    """
    Interface to the EPISODE LOGGER ENGINE.
    """
    def close(self, episode_time=-1, episode_steprate=-1):
        """
        Call this when you are done with the episode!
        """
        raise NotImplementedError()

    def add_scalar(self, tag, value, frame_no=-1, step_no=-1):
        """
        Store scalar.
        """
        raise NotImplementedError()

    def add_scalars(self, tag, value, frame_no=-1, step_no=-1):
        """
        Store several scalars.
        """
        raise NotImplementedError()

    def add_text(self, tag, value, frame_no=-1, step_no=-1):
        """
        Store a text.
        """
        raise NotImplementedError()

    def add_image(self, tag, value):
        """
        Store an image (may be used the store some critical images).
        This could be one shot of a "specific camera" or something.
        """
        raise NotImplementedError()

    def add_video_frame(self, tag, value, width=640, height=480,
                        frame_no=-1, step_no=-1, framerate=-1):
        """
        Store video frame (same as above, but we build videos/gifs from them later).
        We may combine them for a video.
        """
        raise NotImplementedError()

    def add_scene(self, tag, phys, step_no=-1):
        """
        Store physical scene.
        """
        raise NotImplementedError()

    def add_metadata(self, name='data', data={}):
        """
        Add some meta data - packed as JSON.
        """
        raise NotImplementedError()

    def step(self, step_no=-1):
        """
        Switch to next step of simulation.
        Or set explicit to passed one.
        """
        raise NotImplementedError()

    def add_frame_number(self, frame_no):
        """
        Set main-frame number.
        This may provide you with non-linear time abilities.
        """
        raise NotImplementedError()

    def set_video_option(self, k, v):
        """
        Misc to store the video options.
        """
        raise NotImplementedError()


def create_episode_logger(path=False, no_log=False):
    return _EpisodeLoggerImpl(path=path, no_log=no_log)


class _EpisodeLoggerImpl(EpisodeLogger):

    def __init__(self, path=False, no_log=False) -> None:
        self.local_dir = False
        if not path:
            path = tempfile.mkdtemp(prefix="epl_")
            self.local_dir = True

        self.path          = path
        self.framerate     = -1
        self.video_options = {}
        self.localize      = False if os.getenv('NO_LOCALIZE_SCENE_OBJECTS') else True
        self.no_log        = no_log
        self.step_no       = 1
        self.frame_no      = 1    # main frame
        self.X             = {}
        self.DATA          = {}
        self.frame_numbers = {}
        self.episode_time  = -1
        self.min_step_no   = -1
        self.max_step_no   = -1

    # drop local dir
    def __del__(self):
        if self.local_dir:
            shutil.rmtree(self.path)

    # 1. build video or something
    # 2. store some info the episode (success or not and etc)
    # 3. store some JSON's where needed:q
    def __finalize_episode(self, store_data):

        store_data['video'] = {}
        store_data['scene'] = {}

        step_time = -1
        if self.episode_time>0:
            step_time = self.episode_time/(self.max_step_no-self.min_step_no)

        # store simple values
        simple = ['scalar', 'scalars', 'text']
        for x in simple:
            store_data[x] = {}
            if self.X.get(x):
                dp = os.path.join(self.path, x)
                os.makedirs(dp, exist_ok=True)
                for tag in self.X[x]:
                    fn = os.path.join(dp, f'{tag}.json')
                    with open(fn, 'w') as f:
                        json.dump(self.X[x][tag]['value'], f)

                    fn = os.path.join(dp, f'{tag}_fm.json')
                    with open(fn, 'w') as f:
                        json.dump(self.X[x][tag]['frames_map'], f)

                    store_data[x][tag] = True

        # convert to video
        if self.X.get('video'):
            for tag in self.X['video']:

                # render one internal clock time
                internal = False
                internal_framerate = -1
                final_internal_framerate = -1
                average_frame_time = -1
                if self.has_internal_frame_time(self.X['video'][tag]['frames_map']):
                    internal = 'internal.mp4'
                    internal_framerate = self.X['video'][tag]['framerate'] if self.X['video'][tag]['framerate'] > 0  else -1
                    if internal_framerate<0:
                        internaltime = step_time * (self.X['video'][tag]['last_stepno']-self.X['video'][tag]['first_stepno'])
                        internal_framerate = round(len(self.X['video'][tag]['frames_map']) / internaltime)
                        average_frame_time = internaltime / len(self.X['video'][tag]['frames_map'])

                    final_internal_framerate = video_and_scene_utils.images_in_dir_to_video(
                        self.X['video'][tag]['dir'],
                        self.X['video'][tag]['format'],
                        os.path.join(self.X['video'][tag]['dir'], internal),
                        framerate=internal_framerate)


                astro = 'astro.mp4'
                # render astro clock time!
                # lets fill the missing steps
                todel = []
                r = range(self.X['video'][tag]['first_stepno'],self.X['video'][tag]['last_stepno']+1)
                st = self.X['video'][tag]['first_stepno']
                prev = os.path.join(self.X['video'][tag]['dir'], f'frame_{st:06d}.png')
                for st in r:
                    f = os.path.join(self.X['video'][tag]['dir'], f'frame_{st:06d}.png')
                    if not os.path.isfile(f):
                        shutil.copyfile(prev, f)
                    else:
                        prev = f
                # now compute the astro frame rate
                astro_framerate = round(1.0/step_time)
                final_astro_framerate = video_and_scene_utils.images_in_dir_to_video(
                        self.X['video'][tag]['dir'],
                        self.X['video'][tag]['format'],
                        os.path.join(self.X['video'][tag]['dir'], astro),
                        framerate=astro_framerate)
                if BE_VERBOSE:
                    print('=========')
                    print('Episode time        ', self.episode_time)
                    print('=========')
                    print('Astro frames        ', len(r))
                    print('Astro framerate in  ', astro_framerate)
                    print('Astro framerate out ', final_astro_framerate)
                    print('=========')
                    print('Inter frames        ', len(self.X['video'][tag]['frames_map']))
                    print('Inter framerate in  ', internal_framerate)
                    print('Inter framerate out ', final_internal_framerate)
                    print('Average frame time  ', average_frame_time)

                # clean frames
                if not NO_DELETE_FRAME_PICS:
                    for f in glob.glob(os.path.join(self.X['video'][tag]['dir'], self.X['video'][tag]['format'])):
                        os.remove(f)
                        #print(f)
                    for f in todel:
                        os.remove(f)

                # keep data
                store_data['video'][tag] = {
                        'tag': tag,
                        'astrotime': {
                            'video': os.path.join('video',tag, astro),
                            'framerate': final_astro_framerate,
                            'nextframe': average_frame_time
                            },
                        'internaltime': False if not internal else {
                            'video': os.path.join('video',tag, "" if not internal else internal),
                            'framerate': final_internal_framerate,
                            'nextframe': 1.0/final_internal_framerate
                            },
                        'width': self.X['video'][tag]['width'],
                        'height': self.X['video'][tag]['height'],
                        'first_stepno': self.X['video'][tag]['first_stepno'],
                        'first_stepno': self.X['video'][tag]['first_frameno'],
                        'last_stepno': self.X['video'][tag]['last_stepno'],
                        'last_frameno': self.X['video'][tag]['last_frameno'] }

        # convert to scenes
        if self.X.get('scene'):
            for tag in self.X['scene']:
                if self.X['scene'][tag].get('memfile'):
                    os.remove(self.X['scene'][tag]['memfile'])
                self.X['scene'][tag]['log'].dump(self.X['scene'][tag]['dir'])
                store_data['scene'][tag] = {'tag': tag}

        # track arbirtary data
        for k in self.DATA:
            store_data['data'][k] = True

        # some arbirtary data to store
        self.add_metadata('', store_data)

        # store datas
        if self.DATA:
            dp = os.path.join(self.path, 'data')
            os.makedirs(dp, exist_ok=True)
            for k in self.DATA:
                fn = os.path.join(dp, f'{k}.json')
                if k == '':
                    fn = os.path.join(self.path, 'info.json')
                with open(fn, 'w') as f:
                    json.dump(self.DATA[k], f)

    def add_metadata(self, name = 'data', data = {}):
        """
        Method to store an JSON for the epsiode:
        - name is nthe name of the file
        """
        if name != '' and not re.match(r'^[a-zA-Z0-9-]*$', name):
            raise NameError(f'Bad name for metadata: {name}')
        try:
            json.dumps(data)
        except (TypeError, OverflowError):
            raise WrongDocumentErr(f'Not a JSON serializable data')
        self.DATA[name] = data

    def close(self, episode_time = -1, episode_steprate = -1):

        self.episode_time = -1
        if episode_time > 0:
            self.episode_time = episode_time
        else:
            if episode_steprate> 0:
                self.episode_time = ((self.max_step_no-self.min_step_no) / episode_steprate)

        if self.episode_time < 0:
            raise NotImplementedError

        store_data = {
            'date':          datetime.datetime.now().strftime("%B %d, %Y %I:%M%p"),
            'frame_mapping': self.frame_numbers,
            'episode_time' : episode_time,
            'min_step'     : self.min_step_no,
            'max_step'     : self.max_step_no,
        }

        # finalize everything
        self.__finalize_episode(store_data)

        # save episode to S3 or LOCAL folder
        if not self.no_log:
            packer = EpisodePacker()
            id = packer.pack_and_store(self.path)
            print('Episode Logged to:', id)
            url = packer.view_episode_url(id)
            if url:
                print('View episode:', url)

    # generate step and frame numbers
    def step_and_frame_no(self, step_no, frame_no, map):
        step_no  = self.step_no if step_no < 0 else step_no
        frame_no = self.frame_no if frame_no < 0 else frame_no
        map[step_no] = frame_no
        return step_no, frame_no

    # check whether this scalar/video has its own internal frame timeline
    def has_internal_frame_time(self, map):
        val = False
        for x in map:
            if val is False:
                val = map[x]
                continue
            if val != map[x]:
                return True
        return False

    # store scalar
    def add_scalar(self, tag, value, frame_no = -1, step_no = -1):
        if not self.X.get('scalar'):
            self.X['scalar'] = {}
        if not self.X['scalar'].get(tag):
            self.X['scalar'][tag]               = {}
            self.X['scalar'][tag]['value']      = {}
            self.X['scalar'][tag]['frames_map'] = {}
        step_no, frame_no = self.step_and_frame_no(step_no, frame_no, self.X['scalar'][tag]['frames_map'])
        self.X['scalar'][tag]['value'][step_no] = value

    # store seceral scalars
    def add_scalars(self, tag, value, frame_no = -1, step_no = -1):
        if not self.X.get('scalars'):
            self.X['scalars'] = {}
        if not self.X['scalars'].get(tag):
            self.X['scalars'][tag]               = {}
            self.X['scalars'][tag]['value']      = {}
            self.X['scalars'][tag]['frames_map'] = {}
        step_no, frame_no = self.step_and_frame_no(step_no, frame_no, self.X['scalars'][tag]['frames_map'])
        self.X['scalars'][tag]['value'][step_no] = value

    # store seceral scalars
    def add_text(self, tag, value, frame_no = -1, step_no = -1):
        if not self.X.get('text'):
            self.X['text'] = {}
        if not self.X['text'].get(tag):
            self.X['text'][tag]               = {}
            self.X['text'][tag]['value']      = {}
            self.X['text'][tag]['frames_map'] = {}
        step_no, frame_no = self.step_and_frame_no(step_no, frame_no, self.X['text'][tag]['frames_map'])
        self.X['text'][tag]['value'][step_no] = value

    # store an image (may be used the store some critical images)
    # this could one shot of a "specific camera" or something
    def add_image(self, tag, value):
        raise NotImplemented

    # store video frame (same as above , but we build videos/gifs from them later)
    # we may combine them for a video
    def add_video_frame(self, tag, value, width = 640, height = 480, frame_no = -1, step_no = -1, framerate = -1):

        first = False
        if not self.X.get('video'):
            self.X['video'] = {}
        if not self.X['video'].get(tag):
            self.X['video'][tag]           = {}
            dir = os.path.join(self.path, 'video', tag)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            self.X['video'][tag]['dir']    = dir
            self.X['video'][tag]['format'] = 'frame_*.png'
            self.X['video'][tag]['width'] = width
            self.X['video'][tag]['height'] = height
            self.X['video'][tag]['framerate'] = framerate
            self.X['video'][tag]['frames_map']  = {}
            first = True

        step_no, frame_no = self.step_and_frame_no(step_no, frame_no, self.X['video'][tag]['frames_map'])

        if first:
            self.X['video'][tag]['first_stepno']   = step_no
            self.X['video'][tag]['first_frameno']  = frame_no
            self.X['video'][tag]['last_stepno']   = step_no
            self.X['video'][tag]['last_frameno']  = frame_no
        if step_no < self.X['video'][tag]['first_stepno']:
            self.X['video'][tag]['first_stepno'] = step_no
        if frame_no < self.X['video'][tag]['first_frameno']:
            self.X['video'][tag]['first_frameno'] = frame_no
        if step_no > self.X['video'][tag]['last_stepno']:
            self.X['video'][tag]['last_stepno'] = step_no
        if frame_no > self.X['video'][tag]['last_frameno']:
            self.X['video'][tag]['last_frameno'] = frame_no

        f = os.path.join(self.X['video'][tag]['dir'], f'frame_{step_no:06d}.png')
        # we may log over created file!
        #if os.path.isfile(f):
        #    print(f)
        #    raise FileExistsError

        rgb = np.asarray(value[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
        rgbim = Image.fromarray(rgb)
        rgbim_no_alpha = rgbim.convert('RGB')
        rgbim_no_alpha.save(f)

    # store physical scene
    def add_scene(self, tag, phys, step_no = -1):

        # for scene we dont support currently the excplcit value
        # as this depends on poistion set from previous step and etc
        if step_no >= 0:
            raise NotImplementedError

        if not self.X.get('scene'):
            self.X['scene'] = {}
        if not self.X['scene'].get(tag):
            self.X['scene'][tag] = {}
            dir = os.path.join(self.path, 'scene', tag)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            self.X['scene'][tag]['log'] = scene_log.RukaSceneLog(physical_engine=scene_log.phys_name(phys), timerate=scene_log.phys_timerate(phys), dont_localize = not self.localize, )
            self.X['scene'][tag]['dir'] = dir
            if not self.X['scene'][tag]['dir']:
                raise FileIO

        scene_log.phys_store_debug(phys,
                                   scene_log.RukaSceneLog.get_dump_file(self.X['scene'][tag]['dir'],f'scene_{self.step_no:05d}.state' ))
        objects, joints = scene_log.phys_get_objects_and_joins(phys, self.X['scene'][tag]['dir'])

        self.X['scene'][tag]['log'].log_scene(objects, joints)

    # switch to next step of simulation
    # or set explicit to passed one
    def step(self, step_no = -1):
        if step_no < 0:
            self.step_no += 1
        else:
            self.step_no = step_no
        if self.min_step_no < 0:
            self.min_step_no = self.step_no
        if self.max_step_no < 0:
            self.max_step_no = self.step_no
        if step_no < self.min_step_no:
            self.min_step_no = step_no
        if step_no > self.max_step_no:
            self.max_step_no = step_no

    # set main-frame number
    def add_frame_number(self, frame_no):
        self.frame_no = frame_no
        self.frame_map[self.step_no] = frame_no

    # misc to store the video options
    def set_video_option(self, k, v):
        self.video_options[k] = v


class EpisodePacker:
    def __init__(self, config = False):
        self.config = self.parse_config(config if config else os.getenv('RUKA_EPISODE_LOGGER_CONFIG'))

    # parse config
    def parse_config(self, str):
        if not str:
            return {'type': 'local', 'folder': os.getcwd()}
        if str.startswith('local:'):
            return {'type': 'local', 'folder': str[6:]}
        if str.startswith('s3:'):
            return {'type': 's3', 'folder': str[3:], 'as_url': False}
        if str.startswith('s3u:'):
            return {'type': 's3', 'folder': str[4:], 'as_url': True}
        raise NotImplementedError

    # parse passed id
    def parse_id(self, id):
        if id.startswith('file://'):
            return {'type': 'local', 'file': id[7:]}
        if id.startswith('http://') or id.startswith('https://'):
            return {'type': 'url', 'url': id}
        if id.startswith('s3://'):
            return {'type': 's3', 'key': id[5:]}
        raise NotImplementedError

    # prefix length
    def prefix_len(self):
        return 12

    # format magic
    def prefix_magic(self):
        return 'RUKAEPLG'

    # generate prefix for a file
    def gen_prefix(self):
        MAGIC   = bytes(self.prefix_magic(), "utf-8")
        VERSION = 1
        FORMAT  = 1
        return struct.pack('bbbbbbbbHH', MAGIC[0],MAGIC[1],MAGIC[2],MAGIC[3],MAGIC[4],MAGIC[5],MAGIC[6],MAGIC[7],VERSION, FORMAT)


    # parse prefix of length 12
    def parse_prefix(self, str):
        m0, m1, m2, m3, m4, m5, m6, m7, VERSION, FORMAT = struct.unpack('ccccccccHH', str)
        m0 = m0.decode('utf-8')
        m1 = m1.decode('utf-8')
        m2 = m2.decode('utf-8')
        m3 = m3.decode('utf-8')
        m4 = m4.decode('utf-8')
        m5 = m5.decode('utf-8')
        m6 = m6.decode('utf-8')
        m7 = m7.decode('utf-8')
        str = f'{m0}{m1}{m2}{m3}{m4}{m5}{m6}{m7}'
        if str != self.prefix_magic():
            return False
        return {'version': VERSION, 'format': FORMAT}

    # pack and send the episode
    def pack_to(self, path, loc):
        ep_path = loc
        with tempfile.TemporaryDirectory() as dir:
           # Pack.
            tar_path = f'{dir}/archive.tar.gz'
            with tarfile.open(tar_path, 'x:gz') as tar:
                if os.path.isdir(path):
                    for p in os.listdir(path):
                        tar.add(os.path.join(path, p), arcname=p)

            file_add_prefix(tar_path, ep_path, self.gen_prefix())

    # pack and send the episode
    def pack_and_store(self, path):

        id = False
        ep_path = False

        if self.config['type'] == 'local':
            folder = self.config['folder']
            ufn = uuid.uuid4().hex + '.rep'
            while os.path.isfile(os.path.join(folder, ufn)):
                ufn = uuid.uuid4().hex + '.rep'
            ep_path = os.path.join(folder, ufn)
            id = 'file://' + ep_path

            self.pack_to(path, ep_path)

        elif self.config['type'] == 's3':

            with tempfile.NamedTemporaryFile() as tmp:
                self.pack_to(path, tmp.name)
                key, url = s3.upload_file(tmp.name, bucket='episodes', folder = self.config['folder'])
                if key is False:
                    return False

            if self.config['as_url']:
                id = url
            else:
                id = 's3://'+key



        return id

    # unpack local file
    def unpack_file_to(self, file, path):
        with tempfile.NamedTemporaryFile() as tmp:
            with open(file, mode='rb') as f:
                prefix_str = f.read(self.prefix_len())
                prefix = self.parse_prefix(prefix_str)
                if not prefix:
                    return 'Bad format of the episode file'
                if prefix['version'] != 1:
                    raise NotImplementedError
                if prefix['format'] != 1:
                    raise NotImplementedError

                tmp.write(f.read())
                # TODO: do this chunked read/weite later
                # not working for somehow?!
                #for chunk in iter(lambda: f.read(1024), b""):
                #    tmp.write(chunk)
            tar_name = tmp.name
            #shutil.copyfile(tar_name, 'aaa.tar.gz')
            #print(tar_name)
            with tarfile.open(tar_name,'r:gz') as tar:
                tar.extractall(path)
        return True

    # process the remove or local file
    def fetch_and_unpack(self, id, path = False):
        t = self.parse_id(id)

        if not path:
            return 'No path set'

        loc_file = False
        if t['type'] == 'local':
            loc_file = t['file']
            if not os.path.isfile(loc_file):
                return 'File not found'
            ret = self.unpack_file_to(loc_file, path)
            if not (ret is True):
                return ret

        if t['type'] == 'url':
            loc_file = t['url']
            with tempfile.NamedTemporaryFile() as tmp:
                try:
                    urllib.request.urlretrieve(t['url'], tmp.name)
                except:
                    return "Can't download URL: "+t['url']
                ret = self.unpack_file_to(tmp.name, path)
                if not (ret is True):
                    return ret

        if t['type'] == 's3':
            key = t['key']
            with tempfile.NamedTemporaryFile() as tmp:
                if not s3.download_file('episodes', key, tmp.name):
                    return 'S3 key not found'
                ret = self.unpack_file_to(tmp.name, path)
                if not (ret is True):
                        return ret

        return True

    # show view URL
    def view_episode_url(self, id):
        url = os.getenv('RUKA_EPISODE_LOGGER_VIEW_URL')
        if not url:
            return False
        return url.replace('{ID}', urllib.parse.quote_plus(id))


class EpisodeReader():
    def __init__(self):
        self.info = {}

    def load(self, path):
        self.path = path
        if not os.path.isdir(path):
            return 'Bad location'
        info_file = os.path.join(path,'info.json')
        if not os.path.isfile(info_file):
            return 'info.js not found in episode storage'
        with open(info_file) as f:
            d = json.load(f)
        self.info = d
        return True

    def get_videos(self):
        if not self.info.get('video'):
            return []
        ret = []
        for tag in self.info['video']:
            ret.append(self.info['video'][tag])
        return ret

    def get_scalar(self):
        if not self.info.get('scalar'):
            return []
        ret = []
        for tag in self.info['scalar']:
            scname = os.path.join(self.path, 'scalar',f'{tag}.json')
            with open(scname, 'r') as fp:
                data = json.load(fp)
            ret.append({'tag': tag, 'data' : data})
        return ret

    def has_scene(self):
        return True if self.info.get('scene') else False

    def get_scene(self, tag):
        return os.path.join('scene', tag)


DEFAULT_FRAME_RATE_INTERNALTIME = 24
NO_DELETE_FRAME_PICS = False
BE_VERBOSE = False

def generate_debug_episode_folder(base = False, tag = False):
    if not tag:
        tag = 'main'
    if not base:
        base = os.path.join(os.getenv('HOME'), 'episodes')
    base_path = os.path.join(base, tag)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    euid = 1
    bad = True
    while bad:
        bad = False
        try:
            os.mkdir(os.path.join(base_path, f'ep_{euid:03d}'))
        except FileExistsError:
            bad = True
        if bad:
            euid += 1
    return os.path.join(base_path, f'ep_{euid:03d}')

def file_add_prefix(oldfn, newfn, prefix):
    with open(oldfn, "rb") as old, open(newfn, "wb") as new:
        new.write(prefix)
        #for chunk in iter(lambda: old.read(1024), b""):
        #    new.write(chunk)
        new.write(old.read())

