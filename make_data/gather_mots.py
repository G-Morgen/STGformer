import os
import shutil
from glob import glob
from tqdm import tqdm

# if __name__ == '__main__':
#     input_dir = '/home/yaboliu/work/frameworks/yolov5_deepsort/runs/corridor/track_corridor_train'
#     save_dir = '/home/yaboliu/data/cvae/corridor/training/mot_yolov5_deepsort'

#     add_single_sence_id = True
#     single_sence_id = '01'

#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)

#     sub_dirs = os.listdir(input_dir)
#     for sub_dir in tqdm(sub_dirs):
#         sub_path = os.path.join(input_dir, sub_dir)
#         txt_files = glob(os.path.join(sub_path, '*.txt'))
#         if len(txt_files) < 1:
#             continue
#         txt_file = os.path.basename(txt_files[0])
#         if add_single_sence_id:
#             save_txt_name = single_sence_id + '_' + txt_file
#         else:
#             save_txt_name = txt_file
#         shutil.copy(os.path.join(sub_path, txt_file), os.path.join(save_dir, save_txt_name))
        
if __name__ == '__main__':
    input_dir = '/home/yaboliu/data/cvae/corridor/Test_IITB-Corridor/Test_Annotation/Annotation'
    save_dir = '/home/yaboliu/data/cvae/corridor/testing/labels'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    sub_dirs = os.listdir(input_dir)
    for sub_dir in tqdm(sub_dirs):
        sub_path = os.path.join(input_dir, sub_dir)
        txt_files = glob(os.path.join(sub_path, '*.npy'))
        txt_file = os.path.basename(txt_files[0])

        save_txt_name = txt_file
        shutil.copy(os.path.join(sub_path, txt_file), os.path.join(save_dir, save_txt_name))