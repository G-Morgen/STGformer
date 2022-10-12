from pickle import NONE
import shutil
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from utils.scoring_utils import get_frame_dataset_scores

def compress_video(input_path, output_path):
    cmd_str = 'ffmpeg -i {} {}'.format(input_path, output_path)
    os.system(cmd_str)

def get_draw_bar_data(img_size, clip_ids, scores, scores_all):
    height, width = img_size
    left_border = 50
    down_border = 50
    score_hight = 100
    num_imgs = clip_ids.shape[0]
    
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = left_border, height - down_border - score_hight, \
        left_border + num_imgs, height - down_border
        
    scores_normal = (scores / np.max(scores_all)) * score_hight
    scores_normal = bbox_ymax - scores_normal
    bbox = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
    return bbox, scores_normal
    
def draw_progress_bar(img, score_ids, clip_gt, bbox, scores, scores_normal, currunt_count):
    ab_scores = np.ones(clip_gt.shape) * -1
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    cv2.rectangle(img, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), color=(128, 0, 0),thickness=1)
    current_score = None
    lines = []
    # print(score_ids.shape, clip_gt.shape)
    for i in range(len(clip_gt)):
        gt = clip_gt[i]        
        if gt == 0:
            gt_color = (0, 255, 0)
        else:
            gt_color = (0, 0, 255)
        cv2.circle(img, (int(bbox_xmin + i), int(bbox_ymax)), 1, gt_color, -1)
        if score_ids[i] == 1:
            x = bbox_xmin + i
            y = scores_normal[0]
            scores_normal = scores_normal[1:]
            cv2.circle(img, (int(x), int(y)), 1, (128, 0, 0), -1)
            
            ab_scores[i] = scores[0]
            scores = scores[1:]
        else:
            x = bbox_xmin + i
            y = bbox_ymax
        lines.append([int(x), int(y)])
    for i in range(len(lines) - 1):
        point1 = lines[i]
        point2 = lines[i + 1]
        cv2.line(img, point1, point2, (128, 0, 0), 1)
            
    cv2.line(img, (int(bbox_xmin + currunt_count), int(bbox_ymin)), (int(bbox_xmin + currunt_count), int(bbox_ymax)), (255, 255, 255), 1)
    current_gt = 'AB' if clip_gt[currunt_count] else 'NM'
    current_color = (0, 0, 255) if clip_gt[currunt_count] else (0, 255, 0)
    current_score = ab_scores[currunt_count]
    
    cv2.putText(img, '{}:{}_{:3f}'.format(currunt_count, current_gt, current_score), (int(bbox_xmin + currunt_count - 3), bbox_ymin - 5), 
                0, 0.5, current_color, thickness=1, lineType=cv2.LINE_AA)
    return img

def main(score, metadata, dataset):
    gt_arr, scores_arr, score_ids_arr, metadata_arr, clip_gt_arr = get_frame_dataset_scores(score, metadata, dataset=dataset)
    scores_all = np.concatenate(scores_arr)
    for i in range(len(metadata_arr)):
        clip = metadata_arr[i]
        score_ids = score_ids_arr[i]
        clip_gt = clip_gt_arr[i]
        scores = scores_arr[i]
        
        img_dir = os.path.join(data_dir, clip)
        img_ls = sorted(os.listdir(img_dir))
        
        write_fps = 24
        video_save_path = os.path.join(save_dir, clip + '.avi')
        compress_video_save_path = os.path.join(save_dir, clip + '.mp4')
        one_img_name = os.listdir(os.path.join(data_dir, clip))[0]
        one_img= cv2.imread(os.path.join(data_dir, clip, one_img_name))
        height, width = one_img.shape[:2]
        writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), write_fps, (width, height))
        # for i in scores:
        #     print(len(i))
        bbox, scores_normal = get_draw_bar_data([height, width], clip_gt, scores, scores_all)
        # print(bbox, scores_normal)

        for j in range(len(clip_gt)):
            img_np = cv2.imread(os.path.join(img_dir, img_ls[j]))
            img_np = draw_progress_bar(img_np, score_ids, clip_gt, bbox, scores, scores_normal, currunt_count=j)
            writer.write(img_np)
        
        writer.release()
        compress_video(video_save_path, compress_video_save_path)
        os.remove(video_save_path)
            

# shanghai
# if __name__ == '__main__':
#     score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/01/high/stc/Feb19_0154/checkpoints/48_reco_loss.npy')
#     metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/01/high/stc/Feb19_0154/checkpoints/test_dataset_meta.npy')
#     save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/01/high/stc/Feb19_0154/checkpoints/48_reco_loss_results_show'
    
    
    
#     # score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/06/low/stc/Mar19_0254/checkpoints/1_reco_loss.npy')
#     # metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/06/low/stc/Mar19_0254/checkpoints/test_dataset_meta.npy')
#     # save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/06/low/stc/Mar19_0254/checkpoints/1_reco_loss_results_show'
    
    
    
    
    
    
#     data_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/frames'
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
    
#     main(score, metadata, dataset='HR_shanghai')
    
# # avenue
# if __name__ == '__main__':
#     score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_avenue/01/high/stc/Feb14_1310/checkpoints/3_reco_loss.npy')
#     metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_avenue/01/high/stc/Feb14_1310/checkpoints/test_dataset_meta.npy')
#     save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_avenue/01/high/stc/Feb14_1310/checkpoints/3_reco_loss_results_show'
#     data_dir = '/home/yaboliu/data/cvae/avenue/testing/frames'
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
    
#     main(score, metadata, dataset='HR_avenue')

# corrdior
if __name__ == '__main__':
    score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/corridor/04_05_11/high/stc/Feb17_1227/checkpoints/11_reco_loss.npy')
    metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/corridor/04_05_11/high/stc/Feb17_1227/checkpoints/test_dataset_meta.npy')
    save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/corridor/04_05_11/high/stc/Feb17_1227/checkpoints/11_reco_loss_results_show'
    data_dir = '/home/yaboliu/data/cvae/corridor/testing/frames'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    main(score, metadata, dataset='corridor')
    
    
# ucsdped1
# if __name__ == '__main__':
#     score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped1/01/high/stc/Feb13_1738/checkpoints/5_reco_loss.npy')
#     metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped1/01/high/stc/Feb13_1738/checkpoints/test_dataset_meta.npy')
#     save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped1/01/high/stc/Feb13_1738/checkpoints/5_reco_loss_results_show'
#     data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/frames'
#     dataset = 'ucsdped1'
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
    
#     main(score, metadata, dataset)
    
# ucsdped2
# if __name__ == '__main__':
    # score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped2/01/high/stc/Feb13_1745/checkpoints/5_reco_loss.npy')
    # metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped2/01/high/stc/Feb13_1745/checkpoints/test_dataset_meta.npy')
    # save_dir = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/ucsdped2/01/high/stc/Feb13_1745/checkpoints/5_reco_loss_results_show'
    # data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/frames'
    # dataset = 'ucsdped2'
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.mkdir(save_dir)
    
    # main(score, metadata, dataset)