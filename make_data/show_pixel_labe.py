import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# from matplotlib.pyplot import colorbar
# from numpy.testing.nose_tools.utils import import_nose
# from scipy.ndimage.filters import laplace

def compress_video(input_path, output_path):
    cmd_str = 'ffmpeg -i {} {}'.format(input_path, output_path)
    os.system(cmd_str)

def main():
    clip_ls = os.listdir(data_dir)
    for clip in clip_ls:
        img_dir = os.path.join(data_dir, clip)
        label_path = os.path.join(pixel_label_dir, clip + '.npy')
        frame_label_path = os.path.join(frame_label_dir, clip + '.npy')
        label = np.load(label_path)
        frame_label = np.load(frame_label_path)
        img_ls = sorted(os.listdir(img_dir))

        write_fps = 24
        video_save_path = os.path.join(save_dir, clip + '.avi')
        compress_video_save_path = os.path.join(save_dir, clip + '.mp4')
        one_img_name = os.listdir(os.path.join(data_dir, clip))[0]
        one_img= cv2.imread(os.path.join(data_dir, clip, one_img_name))
        height, width = one_img.shape[:2]
        writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), write_fps, (width, height))

        for i in range(len(img_ls)):
            img_name = img_ls[i]
            img_path = os.path.join(img_dir, img_name)
            img_np = cv2.imread(img_path)
            label_one = label[i]
            frame_label_one = frame_label[i]
            color_img = np.ones(img_np.shape)
            color_img[:, :, 2] = 255
            img_np[label_one > 0] = img_np[label_one > 0] * 0.5 + color_img[label_one > 0] * 0.5
            if frame_label_one > 0:
                cv2.rectangle(img_np, (0, 0), (width, height), color=(0, 0, 255),thickness=30)
            writer.write(img_np)
        writer.release()
        compress_video(video_save_path, compress_video_save_path)
        os.remove(video_save_path)
        

if __name__ == '__main__':
    data_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/frames'
    pixel_label_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/test_pixel_mask'
    frame_label_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/test_frame_mask'
    save_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/gt_vis'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    main()

