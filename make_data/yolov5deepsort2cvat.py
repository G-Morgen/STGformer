from genericpath import exists
import os
import shutil
import sunau

def main():
    for txt_file_name in os.listdir(source_dir):
        txt_file_path = os.path.join(source_dir, txt_file_name)
        save_txt_file_path = os.path.join(save_dir, txt_file_name)
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        with open(save_txt_file_path, 'a') as f:  
            for line in lines:
                parts = line.strip().split(' ')[:-4] + ['1', '1', '1.0']
                f.write(','.join(parts) + '\n')
    
if __name__ == '__main__':
    source_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/mot_yolov5_deepsort'
    save_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/mot_yolov5_deepsort_cvat_ori'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    main()

