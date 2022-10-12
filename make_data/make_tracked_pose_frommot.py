from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import IncrementalNewlineDecoder
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from torch._C import *
# from torch._C import preserve_format

# from detector import Detector
from sort import Sort



import os.path as osp
import sys

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

# img_postfix = 'tif'
img_postfix = 'jpg'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('/home/yaboliu/work/frameworks/hrnet/lib')
# mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
# add_path(mm_path)
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

def compress_video(input_path, output_path):
    cmd_str = 'ffmpeg -i {} {}'.format(input_path, output_path)
    os.system(cmd_str)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def track(seq_dets, max_age=3, min_hits=3, iou_threshold=0.3, display=False, img_dir=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold) #create instance of the SORT tracker
    track_result = []
    # with open(save_path, 'w') as out_file:
    # print('####')
    # print(int(seq_dets[:,0].max()))
    # print(seq_dets)
    for frame in range(int(seq_dets[:,0].min()), int(seq_dets[:,0].max())):
        dets = seq_dets[seq_dets[:, 0]==frame, 1:6]
        # print(dets.shape)
        if(display):
            img_path = os.path.join(img_dir, '{:03d}.jpg'.format(frame))
            img_np = cv2.imread(img_path)
            ax1.imshow(img_np)
            plt.title('Tracked Targets')
        # print(dets.shape)
        trackers = mot_tracker.update(dets)
        for d in trackers:
            # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
            track_result.append([frame, d[4], d[0], d[1], d[2], d[3]])
        if(display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()
        frame += 1 #detection and frame numbers begin at 1
        # print(len(trackers), len(track_result))
                
    track_result = np.array(track_result)
    # print(track_result)
    return track_result

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    # parser.add_argument('--cfg', type=str, default='inference-config.yaml')
    parser.add_argument('--cfg', type=str, default='/home/yaboliu/work/frameworks/hrnet/demo/inference-config48.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image',type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args
    

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return [], []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]
    pred_score = pred_score[:pred_t+1]

    person_boxes = []
    person_scores = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)
            person_scores.append(pred_score[idx])

    return person_boxes, person_scores


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale
    
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DetectPose():
    def __init__(self, real_score=False) -> None:
        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        args = parse_args()
        update_config(cfg, args)

        box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        box_model.to(CTX)
        box_model.eval()

        pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

        if cfg.TEST.MODEL_FILE:
            print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')

        pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
        pose_model.to(CTX)
        pose_model.eval()
        
        self.box_model = box_model
        self.pose_model = pose_model
        self.real_score = real_score

    def detect(self, image_bgr):
        image = image_bgr[:, :, [2, 1, 0]]
        # image = image_bgr

        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # print(img.shape)
        img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
        input.append(img_tensor)

        # object detection box
        pred_boxes, pred_scores = get_person_detection_boxes(self.box_model, input, threshold=0.1)
        # print(pred_boxes)
        bboxes = []
        for i in range(len(pred_boxes)):
            bbox = pred_boxes[i]
            score = pred_scores[i]
            # print(score)
            # print(np.array(bbox).reshape(1, 4).tolist())
            bbox = np.array(bbox).reshape(1, 4).tolist()[0] + [score]
            # bbox = np.array(bbox)
            # print(bbox.shape)
            bboxes.append(bbox)
        # if len(bboxes) > 0:
        #     bboxes = np.concatenate(bboxes)

        # else:
        bboxes = np.array(bboxes)
        return bboxes
    
    def pose(self, image_bgr, pred_boxes):
        image = image_bgr[:, :, [2, 1, 0]]
        results = []
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                # image_pose = image_bgr
                pose_preds = get_pose_estimation_prediction(self.pose_model, image_pose, center, scale)
                pose_preds_np = np.array(pose_preds)
                pose_preds_np = pose_preds_np.reshape((-1))
                box = [box[0][0], box[0][1], box[1][0], box[1][1], 1]
                one_result = np.concatenate((np.array(box), pose_preds_np))
                results.append(one_result)
        results = np.array(results)
        return results
      
SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17
      
def draw_vis(img, results):
    for result in results:
        xmin, ymin, xmax, ymax = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        keypoints = result[5:39].reshape((-1, 2))
        person_id = int(result[39])
        cover_flag = int(result[41])
        independent_flag = int(result[42])
        
        if independent_flag == 1:
            rect_color = (0, 255, 0)
        elif cover_flag == 0:
            rect_color = (0, 180, 0)
        else:
            rect_color = (128, 128, 128)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=rect_color,thickness=1)
        cv2.putText(img, str(person_id), (xmin, ymin), 0, 0.5, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img, '{}|{}'.format(cover_flag, independent_flag), (xmin + 15, ymin), 0, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        assert keypoints.shape == (NUM_KPTS,2)
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
            cv2.circle(img, (int(x_a), int(y_a)), 2, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 2, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 1)
    return img


def draw_pose(keypoints_all,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    # print(keypoints_all.shape)
    keypoints_all = keypoints_all[:, 5:].reshape((keypoints_all.shape[0], -1, 2))
    # print(keypoints.shape)
    for i in range(keypoints_all.shape[0]):
        keypoints = keypoints_all[i]
        assert keypoints.shape == (NUM_KPTS,2)
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
            cv2.circle(img, (int(x_a), int(y_a)), 2, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 2, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 1)
    return img

def result2json(results, name_length=3):
    result_json = {}
    print(results.shape)
    person_ids = sorted(list(set(results[:, 39].reshape(-1).tolist())))
    # print(person_ids)
    for person_id in person_ids:
        result_json[str(int(person_id))] = {}
        frames_catched = sorted(list(set(results[results[:, 39]==person_id, 40].reshape(-1).tolist())))
        # frames_catched = [int(i) for i in frames_catched]
        for frame_id in frames_catched:
            if filter_first_frame:
                if frame_id == 1:
                    continue
                frame_name = (name_length - len(str(int(frame_id - 1)))) * '0' + str(int(frame_id - 1))
            else:
                frame_name = (name_length - len(str(int(frame_id)))) * '0' + str(int(frame_id))
            # if num_length == 3:
            #     result_json[str(int(person_id))]["{0:03d}".format(int(frame_id))] = {}
            # elif num_length == 4:
            result_json[str(int(person_id))][frame_name] = {}
            one_person_results = results[results[:, 39]==person_id, :]
            keypoints = one_person_results[one_person_results[:, 40]==frame_id, 5:39]
            bbox = one_person_results[one_person_results[:, 40]==frame_id, :4].tolist()
            covered_flag = one_person_results[one_person_results[:, 40]==frame_id, 41].tolist()[0]
            independent_flag = one_person_results[one_person_results[:, 40]==frame_id, 42].tolist()[0]
            # iofs = one_person_results[one_person_results[:, 40]==frame_id, 43:].tolist()
            # print(keypoints.shape)
            keypoints = keypoints.reshape((-1, 2))
            scores = np.ones((keypoints.shape[0], 1))
            keypoints = np.concatenate((keypoints, scores), axis=1).reshape(-1).tolist()
            result_json[str(int(person_id))][frame_name]['keypoints'] = keypoints
            result_json[str(int(person_id))][frame_name]['bbox'] = bbox
            result_json[str(int(person_id))][frame_name]['covered_flag'] = covered_flag
            result_json[str(int(person_id))][frame_name]['independent_flag'] = independent_flag
            # result_json[str(int(person_id))][frame_name]['iofs'] = iofs
            # result_json[str(int(person_id))]["{0:03d}".format(int(frame_id))]['keypoints'] = keypoints
    return result_json



def mot2np(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    result = []
    for line in lines:
        line = line.strip().split(' ')
        # print(line)
        line = [int(float(i)) for i in line]
        result.append(line)
    result = np.array(result)
    # xmin, ymin, w, h --> xmin, ymin, xmax, ymax
    result[:, 4] += result[:, 2]
    result[:, 5] += result[:, 3]
    return result


def compute_iou(box1, box2, mode='iou'):
 
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
 
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
 
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    if mode == 'iou':
        iou = area_inter / area_union
    elif mode == 'iof':
        iou = area_inter / area_1
    return iou

def filter_cover(bbox_np):
    # print('>>>>')
    # print(bbox_np.shape)
    cover_flag = []
    independent_flag = []
    # ious = []
    if bbox_np.shape[0] <= 1:
        cover_flag = [0]
        independent_flag = [1]
        ious_res = np.array([[-1]])
    else:
        ious = compute_iou(bbox_np, bbox_np, mode='iof')
        ious_res = ious.copy()
        # print('ious, ', ious)
        for i in range(ious.shape[0]):
            ious_res[i][i] = -1
            # print('for, ', i)
            iou = ious.copy()[i]
            iou[i] = -1
            # print(iou)
            max_iou = np.max(iou)
            if max_iou < 0.1:
                independent_flag.append(1)
            else:
                independent_flag.append(0)
            if max_iou > iou_thresh:
                # indexes = np.where(iou > iou_thresh)
                # # print(indexes)
                # current_ymax = bbox_np[0][3]
                # cover_ymax = 0
                # for index in indexes:
                #     index = int(index[0])
                #     if index == i:
                #         continue
                #     cover_y = bbox_np[index][3]
                #     if cover_y > cover_ymax:
                #         cover_ymax = cover_y
                # if current_ymax > cover_ymax:
                #     cover_flag.append(0)    #未被遮挡
                # else:
                #     cover_flag.append(1)
                    
                cover_flag.append(1)
            else:
                cover_flag.append(0)
    cover_flag, independent_flag = np.array(cover_flag), np.array(independent_flag)
    cover_flag, independent_flag = cover_flag[:, None], independent_flag[:, None]
    return cover_flag, independent_flag, ious_res

            
        

def main():
    print('main')
    clip_ls = os.listdir(data_dir)
    # clip_ls = clip_ls[:5]
    for clip in clip_ls:
        print('>>>>> Processing clip {}'.format(clip))
        det_results = []
        clip_dir = os.path.join(data_dir, clip)
        mot_path = os.path.join(mot_dir, clip + '.txt')
        mot_np = mot2np(mot_path)
        img_ls = os.listdir(clip_dir)
        name_length = len(img_ls[0].split('.')[0])
        # for i in img_ls:
        #     print(i, i.split('.')[0])
        #     print(int(i.split('.')[0]))
        frame_count = sorted([int(i.split('.')[0]) for i in img_ls])
        # continue
        if frame_count[0] == 0:
            count_from_0 = True
        else:
            count_from_0 = False
            
        
        video_save_path = os.path.join(vis_save_dir, clip + '.avi')
        compress_video_save_path = os.path.join(vis_save_dir, clip + '.mp4')
        write_fps = 24
        img_name = (name_length - len('1')) * '0' + '1.' + img_postfix
        img_path_one = os.path.join(data_dir, clip, img_name)
        print(img_path_one)
        img_np_one = cv2.imread(img_path_one)
        height, width = img_np_one.shape[:2]
        writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), write_fps, (width, height))
        for img_id in range(len(img_ls)):
            # print('one>>>>')
            if not count_from_0:
                frame_id = img_id + 1
            else:
                frame_id = img_id

            img_name = (name_length - len(str(frame_id))) * '0' + str(frame_id) + '.' + img_postfix
            img_path = os.path.join(clip_dir, img_name)
            img_np = cv2.imread(img_path)
            height, width = img_np.shape[:2]
            bboxes_np = mot_np[mot_np[:, 0]==(frame_id + 1), 2:6]
            persons_np = mot_np[mot_np[:, 0]==(frame_id + 1), 1]
            if bboxes_np.shape[0] > 0:
                covered_flag, independent_flag, iofs = filter_cover(bboxes_np)
                bboxes = []
                for i in range(bboxes_np.shape[0]):
                    bbox_np = bboxes_np[i]
                    bbox = [(float(bbox_np[0]), float(bbox_np[1])), (float(bbox_np[2]), float(bbox_np[3]))]
                    bboxes.append(bbox)
                time0 = time.time()
                pose_result = detector.pose(img_np, bboxes) # bbox score keypoints 39
                time1 = time.time()
                # print(time1 - time0, len(bboxes), ((time1 - time0) /  len(bboxes)))
                # print(pose_result.shape)
                frame_idx_np = np.ones((bboxes_np.shape[0], 1)) * frame_id
                # print(frame_idx_np.shape, cover_flag.shape, independent_flag.shape)
                det_result = np.concatenate((pose_result, persons_np[:, None], frame_idx_np, covered_flag, independent_flag), axis=1) # bbox score keypoints person frame 41
                # print(det_result.shape)
                det_results.append(det_result)
                if vis:
                    img_np = draw_vis(img_np, det_result)
            writer.write(img_np)
            
            # if img_id == 200:
            #     break

        print(len(det_results))
        det_results = np.concatenate(det_results, axis=0)
        print(det_results.shape)
        # track_results = track(det_results)
        # clip_result = []
        # for img_id in range(len(img_ls)):
        #     if not count_from_0:
        #         frame_id = img_id + 1
        #     else:
        #         frame_id = img_id
        #     track_result = track_results[track_results[:, 0]==frame_id, :]
        #     if track_result.shape[0] > 0:
        #         bboxes_np = track_result[:, 2:]
        #         bboxes = []
        #         for i in range(bboxes_np.shape[0]):
        #             bbox_np = bboxes_np[i]
        #             bbox = [(float(bbox_np[0]), float(bbox_np[1])), (float(bbox_np[2]), float(bbox_np[3]))]
        #             bboxes.append(bbox)
        #         img_name = (name_length - len(str(frame_id))) * '0' + str(frame_id) + '.jpg'  
        #         img_path = os.path.join(clip_dir, img_name)
        #         img_np = cv2.imread(img_path)
        #         pose_result = detector.pose(img_np, bboxes)
                
        #         frame_idx_np = np.ones((track_result.shape[0], 1)) * frame_id
        #         person_idx_np = track_result[:, 1]
        #         if len(person_idx_np.shape) == 1:
        #             person_idx_np = person_idx_np[:, None]
        #         one_result = np.concatenate((pose_result, person_idx_np, frame_idx_np), axis=1)
        #         # print(one_result.shape, pose_result.shape, person_idx_np.shape, frame_idx_np.shape)
        #         clip_result.append(one_result)    
        #         if vis:
        #             img_np_vis = draw_vis(img_np, one_result)
        #             # vis_save_path = os.path.join(vis_save_dir, '{}_{}_vis.jpg'.format(clip, frame_id))
        #             # cv2.imwrite(vis_save_path, img_np_vis)
        #             writer.write(img_np_vis)
        #     else:
        #         pass
            
        # det_results = np.concatenate(det_results)
        clip_json = result2json(det_results, name_length)
        json.dump(clip_json, open(os.path.join(json_save_dir, '{}_alphapose_tracked_person.json'.format(clip)),"w"), indent=4, ensure_ascii=False)
        compress_video(video_save_path, compress_video_save_path)
        os.remove(video_save_path)
        writer.release()
        # if vis:
        #     pass
    # track_results = np.array(track_results)  
    # print(track_results.shape)  
            
    
    return


'''
# shanghai
if __name__ == '__main__':
    detector = DetectPose()
    vis = True
    iou_thresh = 1 / 3.5
    filter_first_frame = False

    
    print('###### test ######')
    data_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/frames'
    mot_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/mot_yolov5_deepsort'
    save_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/pose_hrnet48'
    
    json_save_dir = os.path.join(save_dir, 'json_results')
    det_save_dir = os.path.join(save_dir, 'det_results')
    vis_save_dir = os.path.join(save_dir, 'vis_results')
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(json_save_dir)
    os.mkdir(det_save_dir)
    os.mkdir(vis_save_dir)
    main()
    
    print('###### train ######')
    data_dir = '/home/yaboliu/data/cvae/shanghaitech/training/frames'
    mot_dir = '/home/yaboliu/data/cvae/shanghaitech/training/mot_yolov5_deepsort'
    save_dir = '/home/yaboliu/data/cvae/shanghaitech/training/pose_hrnet48'
    
    json_save_dir = os.path.join(save_dir, 'json_results')
    det_save_dir = os.path.join(save_dir, 'det_results')
    vis_save_dir = os.path.join(save_dir, 'vis_results')
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(json_save_dir)
    os.mkdir(det_save_dir)
    os.mkdir(vis_save_dir)
    main()
''' 

# avenue
# if __name__ == '__main__':
#     detector = DetectPose()
#     vis = True
#     iou_thresh = 1 / 3.5
    # filter_first_frame = False

    
#     print('###### test ######')
#     data_dir = '/home/yaboliu/data/cvae/avenue/testing/frames_5'
#     mot_dir = '/home/yaboliu/data/cvae/avenue/testing/mot_yolov5_deepsort'
#     save_dir = '/home/yaboliu/data/cvae/avenue/testing/pose_refine'
    
#     json_save_dir = os.path.join(save_dir, 'json_results')
#     det_save_dir = os.path.join(save_dir, 'det_results')
#     vis_save_dir = os.path.join(save_dir, 'vis_results')
    
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
#     os.mkdir(json_save_dir)
#     os.mkdir(det_save_dir)
#     os.mkdir(vis_save_dir)
#     main()
    
    # print('###### train ######')
    # data_dir = '/home/yaboliu/data/cvae/avenue/training/frames'
    # mot_dir = '/home/yaboliu/data/cvae/avenue/training/mot_yolov5_deepsort'
    # save_dir = '/home/yaboliu/data/cvae/avenue/training/pose'
    
    # json_save_dir = os.path.join(save_dir, 'json_results')
    # det_save_dir = os.path.join(save_dir, 'det_results')
    # vis_save_dir = os.path.join(save_dir, 'vis_results')
    
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.mkdir(save_dir)
    # os.mkdir(json_save_dir)
    # os.mkdir(det_save_dir)
    # os.mkdir(vis_save_dir)
    # main()
    

# ucsdped1
# if __name__ == '__main__':
#     detector = DetectPose()
#     vis = True
#     iou_thresh = 1 / 3.5
    # filter_first_frame = False
    
#     print('###### test ######')
#     data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/frames'
#     mot_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/mot_yolov5_deepsort'
#     save_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/pose'
    
#     json_save_dir = os.path.join(save_dir, 'json_results')
#     det_save_dir = os.path.join(save_dir, 'det_results')
#     vis_save_dir = os.path.join(save_dir, 'vis_results')
    
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
#     os.mkdir(json_save_dir)
#     os.mkdir(det_save_dir)
#     os.mkdir(vis_save_dir)
#     main()
    
#     print('###### train ######')
#     data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Train/frames'
#     mot_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Train/mot_yolov5_deepsort'
#     save_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Train/pose'
    
#     json_save_dir = os.path.join(save_dir, 'json_results')
#     det_save_dir = os.path.join(save_dir, 'det_results')
#     vis_save_dir = os.path.join(save_dir, 'vis_results')
    
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
#     os.mkdir(json_save_dir)
#     os.mkdir(det_save_dir)
#     os.mkdir(vis_save_dir)
#     main()
    

# # # corridor
if __name__ == '__main__':
     detector = DetectPose()
     vis = True
     iou_thresh = 1 / 3.5
     filter_first_frame = True
    
     print('###### test ######')
     data_dir = '/home/yaboliu/data/cvae/corridor/testing/frames'
     mot_dir = '/home/yaboliu/data/cvae/corridor/testing/mot_yolov5_deepsort'
     save_dir = '/home/yaboliu/data/cvae/corridor/testing/pose_hrnet48'
   
     json_save_dir = os.path.join(save_dir, 'json_results')
     det_save_dir = os.path.join(save_dir, 'det_results')
     vis_save_dir = os.path.join(save_dir, 'vis_results')
    
     if os.path.exists(save_dir):
         shutil.rmtree(save_dir)
     os.mkdir(save_dir)
     os.mkdir(json_save_dir)
     os.mkdir(det_save_dir)
     os.mkdir(vis_save_dir)
     main()
    
     print('###### train ######')
     data_dir = '/home/yaboliu/data/cvae/corridor/training/frames'
     mot_dir = '/home/yaboliu/data/cvae/corridor/training/mot_yolov5_deepsort'
     save_dir = '/home/yaboliu/data/cvae/corridor/training/pose_hrnet48'
    
     json_save_dir = os.path.join(save_dir, 'json_results')
     det_save_dir = os.path.join(save_dir, 'det_results')
     vis_save_dir = os.path.join(save_dir, 'vis_results')
    
     if os.path.exists(save_dir):
         shutil.rmtree(save_dir)
     os.mkdir(save_dir)
     os.mkdir(json_save_dir)
     os.mkdir(det_save_dir)
     os.mkdir(vis_save_dir)
     main()