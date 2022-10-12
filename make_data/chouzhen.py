import os
import shutil
import subprocess

def chouzhen(video_path, save_dir):
    strcmd = "ffmpeg -i " + video_path + " {}/%07d.jpg".format(save_dir)
    subprocess.call(strcmd, shell=True)
    
video_dir = '/home/yaboliu/data/cvae/corridor/training/videos'
save_dir = '/home/yaboliu/data/cvae/corridor/training/frames'

# video_dir = '/home/yaboliu/data/cvae/corridor/testing/videos'
# save_dir = '/home/yaboliu/data/cvae/corridor/testing/frames'

# video_ls = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
video_ls = os.listdir(video_dir)
prefix = ''

for video_name in video_ls:
    video_path = os.path.join(video_dir, video_name)
    save_frame_dir = os.path.join(save_dir, prefix + video_name.split('.')[0])
    if os.path.exists(save_frame_dir):
        shutil.rmtree(save_frame_dir)
    os.mkdir(save_frame_dir)
    chouzhen(video_path, save_frame_dir)