from random import shuffle
import shutil
import scipy.io as scio
import os
import numpy as np
from tqdm import tqdm
import cv2

idxes = {'1': [60, 152], 
         '2': [50, 175], 
         '3': [91, 200], 
         '4': [31, 168], 
         '5': [[5, 90], [140,200], 0], 
         '6': [[1,100], [110,200], 0], 
         '7': [1,175], 
         '8': [1,94], 
         '9': [1,48], 
         '10': [1,140], 
         '11': [70,165], 
         '12': [130,200], 
         '13': [1,156], 
         '14': [1,200], 
         '15': [138,200], 
         '16': [123,200], 
         '17': [1,47], 
         '18': [54,120], 
         '19': [64,138], 
         '20': [45,175], 
         '21': [31,200], 
         '22': [16,107], 
         '23': [8,165], 
         '24': [50,171], 
         '25': [40,135], 
         '26': [77,144], 
         '27': [10,122], 
         '28': [105,200], 
         '29': [[1,15], [45,113], 0], 
         '30': [175,200], 
         '31': [1,180], 
         '32': [[1,52], [65,115], 0], 
         '33': [5,165], 
         '34': [1,121], 
         '35': [86,200], 
         '36': [15,108], }

def main():
    for i in range(36):
        count = i + 1
        save_name = '01_0' + (2 - len(str(count))) * '0' + str(count)
        label_np = np.zeros(200)
        print(idxes[str(count)])
        if len(idxes[str(count)]) > 2:
            start_idx, end_idx = idxes[str(count)][0]
            label_np[start_idx - 1: end_idx - 1] = 1
            start_idx, end_idx = idxes[str(count)][1]
            label_np[start_idx - 1: end_idx - 1] = 1
        else:
            start_idx, end_idx = idxes[str(count)]
            label_np[start_idx - 1: end_idx - 1] = 1
        
        np.save(os.path.join(save_dir, save_name), label_np)

if __name__ == '__main__':
    save_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/labels'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    main()



