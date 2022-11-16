import glob
import os
import re
import shutil
import tempfile

from subprocess import run, DEVNULL, check_output


FFMPEG             = "ffmpeg"
LIMIT_FRAMERATE    = -1

def do_images_in_dir_to_video(path, img_pattern, out_file, framerate):
    # ~/ffmpeg/ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4    
    p = run(
        [FFMPEG, '-framerate', f'{framerate}', '-pattern_type', 'glob', '-i', img_pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_file],
        #stdout=DEVNULL, stderr=DEVNULL,
        capture_output=True,
        cwd=path
    )
    if p.returncode != 0:
        print("FFMPEG failed:")
        print (p.stdout)
        print (p.stderr)            
        return False    
    return True
        

def images_in_dir_to_video(path, img_pattern, out_file, framerate = -1):
    framerate = 24 if framerate<0 else framerate
    reduce_rate = False
    if LIMIT_FRAMERATE >= 0:
        if framerate > 0 and framerate>LIMIT_FRAMERATE: 
            reduce_rate = True
    if not reduce_rate:        
        ret = do_images_in_dir_to_video(path=path, img_pattern=img_pattern, out_file=out_file, framerate=framerate)
        if ret:
            return framerate
        return False

    target_framerate = LIMIT_FRAMERATE
    _, file_extension = os.path.splitext(img_pattern)

    with tempfile.TemporaryDirectory() as dir:
        num = 0
        prev = -1
        # files = []        
        for f in sorted(glob.glob(os.path.join(path, img_pattern))):            
            i = round(num * target_framerate / framerate)
            if i != prev:
                prev = i
                shutil.copyfile(f, os.path.join(dir, f"framex_{i:06d}.{file_extension}"))                
            num += 1
        #print('here')
        #print(target_framerate)
        #print(framerate)
        #print(prev)
        #print(num)

        ret = do_images_in_dir_to_video(path=dir, img_pattern=f"framex_*.{file_extension}", out_file=out_file, framerate=target_framerate)

    if ret:
        return target_framerate
    return False


def mktemp_file_in_ram():
    p = run(
        ['mktemp', '-p', '/dev/shm/'],
        #stdout=DEVNULL, stderr=DEVNULL,
        capture_output=True
    )
    if p.returncode != 0:        
        print("mktemp failed:")
        print (p.stdout)
        print (p.stderr)            
        return False
    
    return p.stdout.decode('utf-8').strip()

def localize_urdf_obj(objfile, path, fout):
    import pybullet_data
    with open(objfile) as file:
        lines = file.readlines()
        for line in lines:
            m = re.search(r'mtllib\s+([^\s]+)', line)
            if m:
                fn = m.group(1)
                fullp = fn
                if not os.path.isabs(fullp):
                    fullp = os.path.join(os.path.dirname(objfile), fn)                
                if not os.path.isfile(fullp):
                    fullp = os.path.join(pybullet_data.getDataPath(),fn)
                shutil.copyfile(fullp, os.path.join(path, fn))            
    shutil.copyfile(objfile, os.path.join(path, fout))

def localize_urdf(urdf_in, path, urdf_out, subfolder = "geom"):
    import pybullet_data
    sub = os.path.join(path,subfolder)
    if not os.path.isdir(sub):
        os.mkdir(sub)
    already = {}
    num = 0    
    pd = pybullet_data.getDataPath()
    with open(os.path.join(path, urdf_out), 'w') as the_file:        
        with open(urdf_in) as file:
            lines = file.readlines()
            for line in lines:
                m = re.search(r'filename=[\'"]?([^\'"]+)[\'"]', line)
                if m:
                    fn = m.group(1)
                    fullp = fn
                    if not os.path.isabs(fullp):
                        fullp = os.path.join(os.path.dirname(urdf_in), fn)

                    if os.path.commonpath([pd, fullp]) == pd:
                        # print('in bullled data')                        
                        pass
                    else:
                        if not already.get(fn):
                            #filename, file_extension = os.path.splitext(fn)                        
                            #new_fn = f'g_{num:03d}{file_extension}'
                            #tor = already[fn] = os.path.join(subfolder,new_fn)                         
                            #if file_extension == '.obj':
                            #    localize_urdf_obj(fullp, sub, new_fn)
                            #else:
                            #    shutil.copyfile(fullp, os.path.join(path, tor))
                            bn = os.path.basename(fn)
                            new_dn = f'g_{num:03d}'
                            new_fn = f'{new_dn}/{bn}'
                            tor = already[fn] = os.path.join(subfolder,new_fn)                         
                            #print('========')
                            #print(os.path.dirname(fullp))
                            #print(os.path.join(sub,new_dn))
                            shutil.copytree(os.path.dirname(fullp), os.path.join(sub,new_dn))                                
                            num += 1
                        else:
                            tor = already.get(fn)
                        
                        line = line.replace(fn,tor)
                the_file.write(line)

