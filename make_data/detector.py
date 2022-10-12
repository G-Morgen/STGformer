# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import sys

sys.path.append('/home/yaboliu/work/frameworks/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync

class Detector:
    def __init__(self,
                 model_path,  # model.pt path(s)
                 imgsz=1280,  # inference size (pixels)
                 device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 half=False,  # use FP16 half-precision inference):
                 ):
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        torch.no_grad()
        model = attempt_load(model_path, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        if half:
            model.half()  # to FP16
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        self.device = device
        self.model = model
        self.imgsz = imgsz
        self.half = half
        self.stride = stride

    def inference(self,
                  img0,
                  conf_thres=0.6,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  max_det=1000,  # maximum detections per image
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False,  # class-agnostic NMS
                  ):
        dt = [0, 0, 0]
        t1 = time_sync()
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(img)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            return det
        else:
            return []

def plot(img, bboxes, line_width=3):
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for bbox in bboxes:
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        conf, cls = bbox[4], int(bbox[5])
        color = palette[cls]
        # color = (0, 0, 255)
        txt_color = (255, 255, 255)
        cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
        tf = max(line_width - 1, 1)  # font thickness
        label = '{}_{:.2f}'.format(cls, conf)
        w, h = cv2.getTextSize(label, 0, fontScale=line_width/ 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return img

if __name__ == "__main__":
    img_dir = '/home/yaboliu/data/projects/shengao/images'
    save_dir = '/home/yaboliu/data/projects/shengao/images_vis'
    img_size = [1280, 1920]
    model_path = 'pretrain_models/yolov5s.pt'
    detector = Detector(model_path=model_path, imgsz=img_size, device='0', half=False)

    img_list = os.listdir(img_dir)
    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        img_np = cv2.imread(img_path)
        results = detector.inference(img_np, classes=0)
        vis_img = plot(img_np, results)
        cv2.imwrite(os.path.join(save_dir, img_name), vis_img)
