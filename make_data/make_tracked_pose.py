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
    parser.add_argument('--cfg', type=str, default='inference-config.yaml')
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
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0),thickness=1)
        cv2.putText(img, str(person_id), (xmin, ymin), 0, 0.5, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
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
    person_ids = sorted(list(set(results[:, -2].reshape(-1).tolist())))
    # print(person_ids)
    for person_id in person_ids:
        result_json[str(int(person_id))] = {}
        frames_catched = sorted(list(set(results[results[:, -2]==person_id, -1].reshape(-1).tolist())))
        # frames_catched = [int(i) for i in frames_catched]
        for frame_id in frames_catched:
            frame_name = (name_length - len(str(int(frame_id)))) * '0' + str(int(frame_id))
            # if num_length == 3:
            #     result_json[str(int(person_id))]["{0:03d}".format(int(frame_id))] = {}
            # elif num_length == 4:
            result_json[str(int(person_id))][frame_name] = {}
            one_person_results = results[results[:, -2]==person_id, :]
            keypoints = one_person_results[one_person_results[:, -1]==frame_id, 5:-2]
            # print(keypoints.shape)
            keypoints = keypoints.reshape((-1, 2))
            scores = np.ones((keypoints.shape[0], 1))
            keypoints = np.concatenate((keypoints, scores), axis=1).reshape(-1).tolist()
            result_json[str(int(person_id))][frame_name]['keypoints'] = keypoints
            # result_json[str(int(person_id))]["{0:03d}".format(int(frame_id))]['keypoints'] = keypoints
    return result_json

def main():
    clip_ls = os.listdir(data_dir)
    # clip_ls = clip_ls[:5]
    for clip in clip_ls:
        # print('>>>>')
        print('>>>>> Processing clip {}'.format(clip))
        
        det_results = []
        clip_dir = os.path.join(data_dir, clip)
        img_ls = os.listdir(clip_dir)
        name_length = len(img_ls[0].split('.')[0])
        frame_count = sorted([int(i.split('.')[0]) for i in img_ls])
        if frame_count[0] == 0:
            count_from_0 = True
        else:
            count_from_0 = False
        # print('count_from_0', count_from_0)
        for img_id in range(len(img_ls)):
            if not count_from_0:
                frame_id = img_id + 1
            else:
                frame_id = img_id
            img_name = (name_length - len(str(frame_id))) * '0' + str(frame_id) + '.jpg'
            img_path = os.path.join(clip_dir, img_name)
            img_np = cv2.imread(img_path)
            height, width = img_np.shape[:2]
            # print(img_path)
            det_result = detector.detect(img_np)
            # print(det_result)
            if det_result.shape[0] < 1:
                continue
            
            frame_idx_np = np.ones((det_result.shape[0], 1)) * frame_id
            # print(frame_idx_np.shape, det_result.shape)
            det_result = np.concatenate((frame_idx_np, det_result), axis=1)
            det_results.append(det_result)
            
            if img_id == 200:
                break

        video_save_path = os.path.join(vis_save_dir, clip + '.avi')
        compress_video_save_path = os.path.join(vis_save_dir, clip + '.mp4')
        write_fps = 24
        writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), write_fps, (width, height))
        det_results = np.concatenate(det_results, axis=0)
        track_results = track(det_results)
        clip_result = []
        for img_id in range(len(img_ls)):
            if not count_from_0:
                frame_id = img_id + 1
            else:
                frame_id = img_id
            track_result = track_results[track_results[:, 0]==frame_id, :]
            if track_result.shape[0] > 0:
                bboxes_np = track_result[:, 2:]
                bboxes = []
                for i in range(bboxes_np.shape[0]):
                    bbox_np = bboxes_np[i]
                    bbox = [(float(bbox_np[0]), float(bbox_np[1])), (float(bbox_np[2]), float(bbox_np[3]))]
                    bboxes.append(bbox)
                img_name = (name_length - len(str(frame_id))) * '0' + str(frame_id) + '.jpg'  
                img_path = os.path.join(clip_dir, img_name)
                img_np = cv2.imread(img_path)
                pose_result = detector.pose(img_np, bboxes)
                
                frame_idx_np = np.ones((track_result.shape[0], 1)) * frame_id
                person_idx_np = track_result[:, 1]
                if len(person_idx_np.shape) == 1:
                    person_idx_np = person_idx_np[:, None]
                one_result = np.concatenate((pose_result, person_idx_np, frame_idx_np), axis=1)
                # print(one_result.shape, pose_result.shape, person_idx_np.shape, frame_idx_np.shape)
                clip_result.append(one_result)    
                if vis:
                    img_np_vis = draw_vis(img_np, one_result)
                    # vis_save_path = os.path.join(vis_save_dir, '{}_{}_vis.jpg'.format(clip, frame_id))
                    # cv2.imwrite(vis_save_path, img_np_vis)
                    writer.write(img_np_vis)
            else:
                pass
            
        clip_result = np.concatenate(clip_result)
        clip_json = result2json(clip_result, name_length)
        json.dump(clip_json, open(os.path.join(json_save_dir, '{}_alphapose_tracked_person.json'.format(clip)),"w"), indent=4, ensure_ascii=False)
        compress_video(video_save_path, compress_video_save_path)
        os.remove(video_save_path)
        writer.release()
        # if vis:
        #     pass
    # track_results = np.array(track_results)  
    # print(track_results.shape)  
            
    
    return




if __name__ == '__main__':
    data_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/frames'
    save_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/pose2'  
    vis = True
    
    json_save_dir = os.path.join(save_dir, 'json_results')
    det_save_dir = os.path.join(save_dir, 'det_results')
    vis_save_dir = os.path.join(save_dir, 'vis_results')
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(json_save_dir)
    os.mkdir(det_save_dir)
    os.mkdir(vis_save_dir)
    detector = DetectPose()
    main()
    
    data_dir = '/home/yaboliu/data/cvae/shanghaitech/training/frames'
    save_dir = '/home/yaboliu/data/cvae/shanghaitech/training/pose2'
    
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
