from random import shuffle
import shutil
import scipy.io as scio
import os
import numpy as np
from tqdm import tqdm
import cv2

def main():
    for one_dir in tqdm(os.listdir(input_dir)):
        if one_dir + '_gt' in os.listdir(input_dir):
            continue
        label_dir = os.path.join(input_dir, one_dir)
        if '_gt' not in one_dir:
            for one in os.listdir(label_dir):
                if 'DS' in one:
                    # os.remove(os.path.join(label_dir, one))
                    print('here', one_dir, one)
            label_ls = [0] * len(os.listdir(label_dir))
        else:
            label_ls = []
            for label_img_name in sorted(os.listdir(label_dir)):
                # print(label_img_name)
                if 'DS' in label_img_name:
                    # os.remove(os.path.join(label_dir, label_img_name))
                    print('there', one_dir, label_img_name)
                    continue
                label_img = cv2.imread(os.path.join(label_dir, label_img_name))
                if np.max(label_img) > 0:
                    label_ls.append(1)
                else:
                    label_ls.append(0)
        label = np.array(label_ls)
        npy_name = '01_' + str(one_dir.split('_gt')[0])
        npy_name = npy_name.replace('Test', '')
        np.save(os.path.join(save_dir, npy_name), label)

if __name__ == '__main__':
    input_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/frames'
    save_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/labels'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    main()



