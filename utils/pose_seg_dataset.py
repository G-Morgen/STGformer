import os
import json
from pickle import NONE
import lmdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import normalize_pose

from utils.patch_utils import get_seg_patches, gen_clip_seg_data_np, seg_patches_to_tensor, patches_from_db

# HR_shanghai_except = ['01_0130', '01_0135', '01_0136', '01_0054', '06_0144', '12_0152']
HR_shanghai_except = ['01_0130', '01_0135', '01_0136', '06_0144', '06_0145', '12_0152']
HR_avenue_except = ['04_01', '02_06', '02_09',  '02_15', '02_19', '02_16']


class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, an np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """
    def __init__(self, path_to_json_dir,
                 path_to_vid_dir=None, path_to_patches=None,
                 patch_size=None, transform_list=None,
                 return_indices=False, return_metadata=False, debug=False,
                 dataset_clips=None,
                 **dataset_args):
        super().__init__()
        self.path_to_json = path_to_json_dir
        

        self.patches_db = None
        self.use_patches = False
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.path_to_patches = path_to_patches
        if (path_to_vid_dir is not None) or (path_to_patches is not None):
            self.use_patches = True
            self.use_patches_db = False
            self.patch_size = patch_size
            if path_to_patches is not None:
                self.use_patches_db = True
                self.patches_db = lmdb.open(path_to_patches, subdir=False, lock=False,
                                            readahead=False, readonly=True, meminit=False,
                                            max_readers=2)
                print("Using pre-extracted patches from lmdb env")

        self.debug = debug
        num_clips = 5 if debug else None
        if dataset_clips is not None:
            num_clips = dataset_clips
        self.return_indices = return_indices
        self.return_metadata = return_metadata

        if transform_list is None or transform_list == []:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(transform_list)
        self.transform_list = transform_list
        self.in_channels = dataset_args.get('in_channels', 2)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        
        self.segs_data_np, self.segs_meta, self.person_keys, self.bbox = gen_dataset(path_to_json_dir, num_clips=num_clips,
                                                                          ret_keys=True, **dataset_args)
        # Convert person keys to ints
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.segs_meta = np.array(self.segs_meta)
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = index // self.num_samples
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = data_numpy = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        seg_metadata = self.segs_meta[sample_index]
        ret_arr = [data_transformed, trans_index]
        bbox = np.array(self.bbox[sample_index])
        if self.return_metadata:
            ret_arr += [seg_metadata]
        if self.use_patches:  # Add patch data to loaded segments
            if self.use_patches_db:  # Use pre extracted, slice correct segment and cast ToTensor
                key = ('{:02d}_{:04d}_{:02d}'.format(*seg_metadata[:3]))
                person_patches_np, _ = patches_from_db(self.patches_db, key.encode('ascii'))
                person_keys_sorted = self.person_keys[key]
                start_ofst = person_keys_sorted.index(seg_metadata[-1])
                seg_patches_np = person_patches_np[:, start_ofst: start_ofst+self.seg_len, :, :, :]
                seg_patches_tensor = seg_patches_to_tensor(seg_patches_np)
            else:  # Extract patches from individual jpeg frames
                dir_len = len(os.listdir(self.path_to_vid_dir)[0])
                if dir_len == 6:
                    dirn = '{:02d}_{:03d}'.format(seg_metadata[0], seg_metadata[1])
                else:
                    dirn = '{:02d}_{:04d}'.format(seg_metadata[0], seg_metadata[1])
                seg_patches_tensor = self.get_set_patch_tensor(data_numpy, dirn, seg_metadata)

            if self.headless:
                seg_patches_tensor = seg_patches_tensor[:, :, :14]
            ret_arr = [seg_patches_tensor] + ret_arr[1:]  # Replace pose w/patches

        if self.return_indices:
            ret_arr += [index]
        ret_arr += [bbox]
        return ret_arr

    def get_set_patch_tensor(self, data_numpy, dirn, metadata):
        return get_seg_patches(os.path.join(self.path_to_vid_dir, dirn), data_numpy, metadata,
                               lmdb_env=self.patches_db, patch_size=self.patch_size)

    def __len__(self):
        return self.num_transform * self.num_samples


def gen_dataset(person_json_root, num_clips=None, normalize_pose_segs=True,
                kp18_format=True, ret_keys=False, **dataset_args):
    segs_data_np = []
    segs_meta = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 12)
    vid_res = dataset_args.get('vid_res', [856, 480])
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    add_center = dataset_args.get('add_center', False)
    filter_bbox = dataset_args.get('filter_bbox', -1)
    dataset = dataset_args.get('dataset', 'shanghai')
    dataset_sence = dataset_args.get('dataset_sence', None)
    in_channels= dataset_args.get('in_channels', 2)
    filter_border = dataset_args.get('filter_border', -1)
    filter_independent = dataset_args.get('filter_independent', False)
    filter_cover = dataset_args.get('filter_cover', False)

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('.json')])
    if num_clips:
        json_list = json_list[:num_clips]  # For debugging purposes
    for person_dict_fn in json_list:
        # if dataset == 'HR_shanghai':
        #     if person_dict_fn.split('.')[0] in HR_shanghai_except:
        #         continue
        # if dataset == 'HR_avenue':
        #     if person_dict_fn.split('.')[0] in HR_avenue_except:
        #         continue
        if dataset == 'HR_shanghai':
            if person_dict_fn.split('_alphapose')[0] in HR_shanghai_except:
                print('HR dataset exception:')
                print(person_dict_fn)
                continue
        # if dataset == 'HR_avenue':
        #     if person_dict_fn.split('_alphapose')[0] in HR_avenue_except:
        #         print('HR dataset exception:')
        #         print(person_dict_fn)
        #         continue
        scene_id, clip_id = person_dict_fn.split('_')[:2]
        if dataset_sence is not None and (scene_id not in dataset_sence):
            continue
        print('use sence: ', person_dict_fn)
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys = gen_clip_seg_data_np(clip_dict, start_ofst, seg_stride,
                                                                            seg_len, scene_id=scene_id,
                                                                            clip_id=clip_id, ret_keys=ret_keys, filter_bbox=filter_bbox,
                                                                            filter_border=filter_border, dataset=dataset, filter_independent=filter_independent,
                                                                            filter_cover=filter_cover)
        segs_data_np.append(clip_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}
    segs_data_np = np.concatenate(segs_data_np, axis=0) # curr_pose_segs_np [-1, t, 17, 3]
    # print('dataset000000 :', segs_data_np.shape)
    # print('dataloader:', segs_data_np)
    segs_bbox_np = segs_data_np[..., -4:, :]
    segs_data_np = segs_data_np[..., :-4, :]
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
    # print('dataset :', segs_data_np)
    if normalize_pose_segs:
        del dataset_args['vid_res']
        segs_data_np = normalize_pose(segs_data_np, segs_bbox_np, vid_res=vid_res, **dataset_args)
        print(segs_data_np.shape)
        
    # print('dataset :', segs_data_np)
    if headless:
        if add_center:
            segs_data_np = segs_data_np[:, :, :14, :]
            center = segs_data_np[:, :, -1, :]
            segs_data_np = np.concatenate([segs_data_np, center[:, :, None, :]], axis=2)
        else:
            segs_data_np = segs_data_np[:, :, :14, :]
        
        print('head less dataset out: ', segs_data_np.shape)
        # segs_data_graph = segs_data_np[:, :, :14, :]
        # segs_data_position = segs_data_np[:, :, -1, :]
        # segs_data_np = np.concatenate([segs_data_graph, segs_data_position[:, :, None, :]], axis=2)
        
    # segs_data_np [-1, t, kp, 3]
    # print('dataset :', segs_data_np.shape)
    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)  # segs_data_np [-1, 3, t, kp]
    
    # if seg_conf_th > 0.0:
    #     segs_data_np, segs_meta = seg_conf_th_filter(segs_data_np, segs_meta, seg_conf_th)
        
    # optional
    # if not dataset_args.new_preprocess:
    segs_data_np = segs_data_np[:, :in_channels, :, :]
    print('dataset out: ', segs_data_np.shape)
    segs_bbox_np = segs_bbox_np[..., 0]
    
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_bbox_np
    else:
        return segs_data_np, segs_meta


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, seg_conf_th=2.0):
    seg_len = segs_data_np.shape[2]
    conf_vals = segs_data_np[:, 2]
    sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    return seg_data_filt, seg_meta_filt

