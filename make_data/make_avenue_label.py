from random import shuffle
import shutil
import scipy.io as scio
import os
import numpy as np
from tqdm import tqdm

def main():
    for mat_name in tqdm(os.listdir(input_dir)):
        mat_path = os.path.join(input_dir, mat_name)
        mat = scio.loadmat(mat_path)['volLabel'][0]
        label_ls = []
        for i in range(len(mat)):
            mat_i = mat[i]
            if np.max(mat_i) > 0:
                label_ls.append(1)
            else:
                label_ls.append(0)
        label = np.array(label_ls)
        npy_name = str(mat_name.split('_')[0])
        npy_name = '01_' + '0' * (2 - len(npy_name)) + npy_name
        np.save(os.path.join(save_dir, npy_name), label)

if __name__ == '__main__':
    input_dir = '/home/yaboliu/data/cvae/avenue/testing/testing_label_mask'
    save_dir = '/home/yaboliu/data/cvae/avenue/testing/labels'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    main()

