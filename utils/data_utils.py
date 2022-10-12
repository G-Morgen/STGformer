import math
from pickle import NONE
from matplotlib.pyplot import axis
import torch
import numpy as np


def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x


ae_trans_list = [
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=True),  # 3
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=False),  # 6
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=True),  # 9
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=45, flip=False),  # 12
]

def distance(m, n):
    sum_temp = np.sum((m - n) ** 2, axis=-1)
    return np.sqrt(sum_temp[..., np.newaxis])

def normalize_pose(pose_data, bbox_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', True)
    # symm_range = kwargs.get('symm_range', False)
    sub_mean = kwargs.get('sub_mean', True)
    scale = kwargs.get('scale', False)
    scale_proportional = kwargs.get('scale_proportional', False)
    new_preprocess = kwargs.get('new_preprocess', False)
    only_normal = kwargs.get('only_normal', False)
    add_center = kwargs.get('add_center', False)
    add_scale = kwargs.get('add_scale', False)
    print('add_center: ', add_center)
    print('add_scale: ', add_scale)
    
    
    print('normalize:')
    print('vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess')
    print(vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess)
    with open('train.txt', 'a') as f:
        f.write('vid_res, symm_range, sub_mean, scale, scale_proportional\n')
        f.write('{} {} {} {} {}\n'.format(vid_res, symm_range, sub_mean, scale, scale_proportional))
        
    # print(pose_data.shape, bbox_data.shape)    
    if (new_preprocess and not only_normal):
        pose_data = pose_data[..., :2] # [-1, t, 18, 2]
        bbox_data = bbox_data[..., 0]   # [-1, t, 4]
        # openpose中 肩 2 5 胯 8 11
        # pose_centers = (pose_data[..., 11, :] + pose_data[..., 5, :] + pose_data[..., 8, :] + pose_data[..., 2, :]) / 4
        # pose_centers = pose_centers[:, :, None, :]
        

        # hip_shoulder_distance = (distance(pose_data[..., 8, :], pose_data[..., 2, :]) + 
        #                       distance(pose_data[..., 11, :], pose_data[..., 5, :])) / 2
        # hip_shoulder_scale_target = hip_shoulder_distance / hip_shoulder_distance.mean(axis=1)[..., None]
        # pose_data = pose_data * hip_shoulder_scale_target[..., None]
        
        # temp_center = np.array(vid_res) / 2
        # pose_diff = temp_center[None, None, None, :] - pose_centers
        # pose_data = pose_data + pose_diff
        
        min_kp_xy = np.min(pose_data, axis=(1, 2))
        pose_data = pose_data - min_kp_xy[:, None, None, :]
        
        if add_center:
            pose_centers = (pose_data[..., 11, :] + pose_data[..., 5, :] + pose_data[..., 8, :] + pose_data[..., 2, :]) / 4
            pose_centers = pose_centers[:, :, None, :]
            pose_data = np.concatenate([pose_data, pose_centers], axis=2)
        max_kp_xy = np.max(pose_data, axis=(1, 2))
        pose_data = pose_data / max_kp_xy[:, None, None, :]
        if add_scale:
            width_scale = (bbox_data[..., 2] - bbox_data[..., 0]) / vid_res[0]
            hight_scale = (bbox_data[..., 3] - bbox_data[..., 1]) / vid_res[1]
            width_scale = [width_scale[:, :, None, None] for _ in range(pose_data.shape[2])]
            hight_scale = [hight_scale[:, :, None, None] for _ in range(pose_data.shape[2])]
            width_scale = np.concatenate(width_scale, axis=2)
            hight_scale = np.concatenate(hight_scale, axis=2)
            print(width_scale.shape, hight_scale.shape, pose_data.shape)
            pose_data = np.concatenate([pose_data, width_scale, hight_scale], axis=3)

        return pose_data    # [-1, t, 18+1, 2]
    elif (not new_preprocess and only_normal):
        pose_data = pose_data[..., :2] # [-1, t, 18, 2]
        bbox_data = bbox_data[..., 0]   # [-1, t, 4]
        pose_data = pose_data / np.array(vid_res)  # t p 1 c
        # if add_scale:
        #     width_scale = (bbox_data[..., 2] - bbox_data[..., 0]) / vid_res[0]
        #     hight_scale = (bbox_data[..., 3] - bbox_data[..., 1]) / vid_res[1]
        #     width_scale = [width_scale[:, :, None, None] for _ in range(pose_data.shape[2])]
        #     hight_scale = [hight_scale[:, :, None, None] for _ in range(pose_data.shape[2])]
        #     width_scale = np.concatenate(width_scale, axis=2)
        #     hight_scale = np.concatenate(hight_scale, axis=2)
        #     print(width_scale.shape, hight_scale.shape, pose_data.shape)
        #     pose_data = np.concatenate([pose_data, width_scale, hight_scale], axis=3)

        return pose_data    # [-1, t, 18+1, 2]
    
            
    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    if sub_mean or scale or scale_proportional:  # Inner frame scaling requires mean subtraction
        pose_data_zero_mean = pose_data_centered
        mean_kp_val = np.mean(pose_data_zero_mean[..., :2], (1, 2))
        pose_data_zero_mean[..., :2] -= mean_kp_val[:, None, None, :]

    max_kp_xy = np.max(np.abs(pose_data_centered[..., :2]), axis=(1, 2))
    max_kp_coord = max_kp_xy.max(axis=1)

    # pose_data_scaled = pose_data_zero_mean
    pose_data_scaled = pose_data_centered
    if scale:
        print('scale>>>>>>>>>>>')
        # Scale sequence to maximize the [-1,1] frame
        # Removes average position from all keypoints, than scales in x and y to fill the frame
        # Loses body proportions
        pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_xy[:, None, None, :]

    elif scale_proportional:
        # Same as scale but normalizes by the same factor
        # (smaller axis, i.e. divides by larger fraction value)
        # Keeps propotions
        pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_coord[:, None, None, None]

    return pose_data_scaled


def make_high_level_point(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', True)
    # symm_range = kwargs.get('symm_range', False)
    sub_mean = kwargs.get('sub_mean', True)
    scale = kwargs.get('scale', False)
    scale_proportional = kwargs.get('scale_proportional', False)
    new_preprocess = kwargs.get('new_preprocess', False)
    add_center = kwargs.get('add_center', False)
    print('add_center: ', add_center)
    
    
    print('normalize:')
    print('vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess')
    print(vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess)
    with open('train.txt', 'a') as f:
        f.write('vid_res, symm_range, sub_mean, scale, scale_proportional\n')
        f.write('{} {} {} {} {}\n'.format(vid_res, symm_range, sub_mean, scale, scale_proportional))
        
    norm_factor = np.array(vid_res)
    high_level_ls = []
    bbox_ls = []
    for one_pose_data in pose_data:
        kp_np = np.array(one_pose_data) # t p v c'
        # print(kp_np.shape)
        bbox = kp_np[..., -4:, 0]
        kp_np = kp_np[..., :-4, :]
        # print(bbox)
        # height = bbox[..., 3, 0] - bbox[..., 1, 0]
        # height = height[:, :, None, None] / vid_res[1]
        # print('kp_Np: ', kp_np.shape)
        neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
        kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
        opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        opp_order = np.array(opp_order, dtype=np.int)
        kp_coco18 = kp_np[..., opp_order, :]
            
        # pose_centers = (kp_coco18[..., 11, :] + kp_coco18[..., 5, :] + kp_coco18[..., 8, :] + kp_coco18[..., 2, :]) / 4
        pose_centers = (kp_coco18[..., 11, :] + kp_coco18[..., 8, :]) / 2
        pose_centers = pose_centers[..., None, :2] / norm_factor  # t p 1 c
        # print(pose_centers.shape, bbox.shape)
        # pose_centers = np.concatenate([pose_centers, bbox], axis=2)
        bbox_ls.append(bbox)
        
        # print('>' * 10)
        # print(pose_centers.shape)
        pose_centers = np.transpose(pose_centers, (3, 1, 0, 2)).astype(np.float32) # c p t 1
        # print(pose_centers.shape)
        high_level_ls.append(pose_centers)
        
    return high_level_ls, bbox_ls

def use_bbox_make_high_level_point(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', True)
    # symm_range = kwargs.get('symm_range', False)
    sub_mean = kwargs.get('sub_mean', True)
    scale = kwargs.get('scale', False)
    scale_proportional = kwargs.get('scale_proportional', False)
    new_preprocess = kwargs.get('new_preprocess', False)
    add_center = kwargs.get('add_center', False)
    print('add_center: ', add_center)
    
    
    print('normalize:')
    print('vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess')
    print(vid_res, symm_range, sub_mean, scale, scale_proportional, new_preprocess)
    with open('train.txt', 'a') as f:
        f.write('vid_res, symm_range, sub_mean, scale, scale_proportional\n')
        f.write('{} {} {} {} {}\n'.format(vid_res, symm_range, sub_mean, scale, scale_proportional))
        
    norm_factor = np.array(vid_res)
    high_level_ls = []
    bbox_ls = []
    for one_pose_data in pose_data:
        kp_np = np.array(one_pose_data) # t p v c'
        # print(kp_np.shape)
        bbox = kp_np[..., -4:, 0]   # xmin ymin xmax ymax
        # kp_np = kp_np[..., :-4, :]
        # # print(bbox)
        # # height = bbox[..., 3, 0] - bbox[..., 1, 0]
        # # height = height[:, :, None, None] / vid_res[1]
        # # print('kp_Np: ', kp_np.shape)
        # neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
        # kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
        # opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # opp_order = np.array(opp_order, dtype=np.int)
        # kp_coco18 = kp_np[..., opp_order, :]
            
        # # pose_centers = (kp_coco18[..., 11, :] + kp_coco18[..., 5, :] + kp_coco18[..., 8, :] + kp_coco18[..., 2, :]) / 4
        # pose_centers = (kp_coco18[..., 11, :] + kp_coco18[..., 8, :]) / 2
        
        # print(pose_centers.shape, bbox.shape)
        # print(bbox)
        pose_centers_x = (bbox[..., 0] + bbox[..., 2]) / 2
        pose_centers_y = (bbox[..., 1] + bbox[..., 3]) / 2
        # print(bbox.shape, bbox[..., 0].shape, pose_centers_x.shape, pose_centers_y.shape)
        pose_centers = np.concatenate([pose_centers_x[..., None], pose_centers_y[..., None]], axis=2)
        # print(pose_centers.shape)
        pose_centers = pose_centers[..., None, :2] / norm_factor  # t p 1 c
        # print(pose_centers.shape, bbox.shape)
        # pose_centers = np.concatenate([pose_centers, bbox], axis=2)
        bbox_ls.append(bbox)
        
        # print('>' * 10)
        # print(pose_centers.shape)
        pose_centers = np.transpose(pose_centers, (3, 1, 0, 2)).astype(np.float32) # c p t 1
        # print(pose_centers.shape)
        high_level_ls.append(pose_centers)
        
    return high_level_ls, bbox_ls