import os
from re import S
import shutil

# ucsd
def move_ucsd():
    for one_dir in os.listdir(source_dir):
        dst_name = one_dir.replace('Test', '')
        dst_name = dst_name.replace('Train', '')
        dst_name = '01_' + dst_name
        shutil.move(os.path.join(source_dir, one_dir), os.path.join(source_dir, dst_name))
        
# avenue
def move_avenue():
    for one_dir in os.listdir(source_dir):
        dst_name = '01_' + one_dir
        shutil.move(os.path.join(source_dir, one_dir), os.path.join(source_dir, dst_name))
        
# source_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/mot_yolov5_deepsort'
# move_ucsd()

# source_dir = '/home/yaboliu/data/cvae/avenue/training/frames'
# move_avenue()

# source_dir = '/home/yaboliu/data/cvae/corridor/training/mot_yolov5_deepsort'
# for i in os.listdir(source_dir):
#     shutil.move(os.path.join(source_dir, i), os.path.join(source_dir, '01_' + i))


frame_dir = '/home/yaboliu/data/cvae/corridor/testing/frames'
source_dir = '/home/yaboliu/data/cvae/corridor/testing/mot_yolov5_deepsort'
frame_name_dict = {}
for frame_name in os.listdir(frame_dir):
    sence_id, vid_id = frame_name.split('_')
    frame_name_dict[vid_id] = sence_id
source_name_dict = {}
for source_name in os.listdir(source_dir):
    sence_id, vid_id = source_name.split('_')
    new_name = '{}_{}'.format(frame_name_dict[vid_id.split('.')[0]], vid_id)
    # print(new_name)
    shutil.move(os.path.join(source_dir, source_name), os.path.join(source_dir, new_name))