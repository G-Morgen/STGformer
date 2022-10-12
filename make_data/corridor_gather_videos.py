import os
import shutil


def main_gather_videos():
    for sub_dir_name in os.listdir(source_dir):
        sub_dir = os.path.join(source_dir, sub_dir_name)
        video_name = os.listdir(sub_dir)[0]
        video_path = os.path.join(sub_dir, video_name)
        shutil.move(video_path, os.path.join(dst_dir, video_name))
    
if __name__ == '__main__':
    # source_dir = '/home/yaboliu/data/cvae/corridor/Train_IITB_Corridor/Train'
    # dst_dir = '/home/yaboliu/data/cvae/corridor/training/videos'
    # source_dir = '/home/yaboliu/data/cvae/corridor/Test_IITB-Corridor/Test'
    # dst_dir = '/home/yaboliu/data/cvae/corridor/testing/videos'
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)
    main_gather_videos()