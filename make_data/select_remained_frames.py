import os
import shutil

source_dir = '/home/yaboliu/data/cvae/corridor/training/frames_step1'
dst_dir = '/home/yaboliu/data/cvae/corridor/training/frames_step2'
exists_frames_dir = '/home/yaboliu/data/cvae/corridor/training/pose_step1/vis_results'

exists_frames_ls = []
for exists_frames_name in os.listdir(exists_frames_dir):
    exists_frames_ls.append(exists_frames_name.split('.')[0])
    
for one_dir in os.listdir(source_dir):
    if one_dir not in exists_frames_ls:
        shutil.move(os.path.join(source_dir, one_dir), os.path.join(dst_dir, one_dir))