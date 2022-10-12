from functools import partial, reduce
import os
from pickle import NONE
from posixpath import join
from shutil import Error
import numpy as np
from scipy.ndimage.interpolation import shift
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from sklearn import mixture
from joblib import dump, load
import cv2
from sklearn import metrics
from .pose_seg_dataset import HR_shanghai_except


def dpmm_calc_scores(model, train_dataset, eval_normal_dataset, eval_abn_dataset=None, args=None,
                     ret_metadata=False, pred_loss=False, dpmm_components=10, dpmm_downsample_fac=10, pt_dpmm_path=None):
    """
    Wrapper for extracting features for DNS experiment, given a trained DCEC models, a normal training dataset and two
    datasets for evaluation, a "normal" one and an "abnormal" one
    :param model: A trained model
    :param train_dataset: "normal" training dataset, for alpha calculation
    :param eval_normal_dataset: "normal" or "mixed" evaluation dataset
    :param eval_abn_dataset: "abnormal" evaluation dataset (optional)
    :param args - command line arguments
    :param ret_metadata:
    :param dpmm_components:  Truncation parameter for DPMM
    :param dpmm_downsample_fac: Downsampling factor for DPMM fitting
    :param pt_dpmm_path: Path to a pretrained DPMM model
    :return actual experiment done after feature extraction (calc_p)
    """
    # Alpha calculation and fitting
    train_p = calc_p(model, train_dataset, args, ret_metadata=False)
    print('>>>>>.', train_p.shape)
    eval_p_ret = calc_p(model, eval_normal_dataset, args, ret_metadata=ret_metadata, pred_loss=pred_loss)
    if ret_metadata:
        if pred_loss:
            eval_p_normal, metadata, reco_loss = eval_p_ret
            # reco_loss = np.concatenate(reco_loss)
            # print(loss.shape)
        else:
            eval_p_normal, metadata = eval_p_ret
    else:
        eval_p_normal = eval_p_ret

    p_vec = eval_p_normal
    eval_p_abn = None
    if eval_abn_dataset:
        eval_p_abn = calc_p(model, eval_abn_dataset, args, ret_metadata=ret_metadata)
        p_vec = np.concatenate([eval_p_normal, eval_p_abn])
    # print(train_p.shape)
    print("Started fitting DPMM")
    if pt_dpmm_path is None:
        dpmm_mix = mixture.BayesianGaussianMixture(n_components=dpmm_components,
                                                   max_iter=500, verbose=1, n_init=1)
        # a = train_p[::dpmm_downsample_fac]
        # print(len(a), a[0].shape)
        dpmm_mix.fit(train_p[::dpmm_downsample_fac])
    else:
        dpmm_mix = load(pt_dpmm_path)
    
    dpmm_scores = dpmm_mix.score_samples(p_vec)

    if eval_p_abn is not None:
        gt = np.concatenate([np.ones(eval_p_normal.shape[0], dtype=np.int),
                             np.zeros(eval_p_abn.shape[0], dtype=np.int)])
    else:
        gt = np.ones_like(dpmm_scores, dtype=np.int)

    try:  # Model persistence
        dpmm_fn = args.ae_fn.split('.')[0] + '_dpgmm.pkl'
        dpmm_path = os.path.join(args.ckpt_dir, dpmm_fn)
        dump(dpmm_mix, dpmm_path)
    except:
        print("Joblib missing, DPMM not saved")
        
    feature_vectors = [train_p, p_vec]

    if ret_metadata:
        if pred_loss:
            return dpmm_scores, gt, metadata, reco_loss
        else:
            return dpmm_scores, gt, metadata, feature_vectors
    else:
        if pred_loss:
            return dpmm_scores, gt, reco_loss
        else:
            return dpmm_scores, gt


def calc_p(model, dataset, args, ret_metadata=True, ret_z=False, pred_loss=False):
    """ Evalutates the models output over the data in the dataset. """
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, pin_memory=True)
    model = model.to(args.device)
    model.eval()
    if pred_loss:
        # print('here')
        p, loss = p_compute_features(loader, model, device=args.device, ret_z=ret_z, pred_loss=pred_loss)
    else:
        # print('there')
        p = p_compute_features(loader, model, device=args.device, ret_z=ret_z)
    # print('>>>>>****', p.shape)
    if ret_z:
        p, z = p
        if ret_metadata:
            # print(1)
            return p, z, dataset.metadata
        else:
            # print(2)
            return p, z
    else:
        if ret_metadata:
            if pred_loss:
                # print(3)
                return p, dataset.metadata, loss
            else:
                # print(4)
                return p, dataset.metadata
        else:
            if pred_loss:
                # print(5)
                return p, loss
            else:
                # print(6)
                return p


def p_compute_features(loader, model, device='cuda:0', ret_z=False, pred_loss=False):
    sfmax = []
    z_arr = []
    loss_ls = []
    loss_func = torch.nn.MSELoss(reduce=False).cuda()
    for itern, data_arr in enumerate(loader):
        data = data_arr[0]
        if itern % 100 == 0:
            print("Compute Features Iter {}".format(itern))
        with torch.no_grad():
            data = data.to(device)
            model_ret = model(data, ret_z=ret_z)
            cls_sfmax, reco, gt = model_ret
            # cls_sfmax = model_ret[0]
            if pred_loss:
                loss = loss_func(reco, gt)
                loss = loss.view(loss.shape[0], -1)
                loss = loss.sum(axis=1) / loss.shape[1]
                loss_ls.append(loss.to('cpu', non_blocking=True).numpy().astype('float32'))
                # print(loss)
            if ret_z:
                z = model_ret[-1]
                z_arr.append(z.to('cpu', non_blocking=True).numpy().astype('float32'))
            cls_sfmax = torch.reshape(cls_sfmax, (cls_sfmax.size(0), -1))
            # print(cls_sfmax.shape)
            sfmax.append(cls_sfmax.to('cpu', non_blocking=True).numpy().astype('float32'))
        # if itern == 50:
        #     break
            
    print(len(sfmax))
    sfmax = np.concatenate(sfmax)
    print(sfmax.shape)
    if ret_z:
        z_arr = np.concatenate(z_arr)
        if pred_loss:
            loss_np = np.concatenate(loss_ls)
            return sfmax, z_arr, loss_np
        else:
            return sfmax, z_arr
    else:
        if pred_loss:
            loss_np = np.concatenate(loss_ls)
            return sfmax, loss_np
        else:
            print('####', sfmax.shape)
            return sfmax


def score_dataset(score_vals, metadata, max_clip=None, scene_id=None, level='low', dataset='shanghai'):
    if level == 'low':
        gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, metadata, max_clip, scene_id, dataset)
    elif level == 'high':
        gt_arr, scores_arr, score_ids_arr, metadata_arr, clip_gt_arr = get_frame_dataset_scores(score_vals, metadata, max_clip, scene_id, dataset)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    print('score_dataset: ', scores_np.shape, gt_np.shape)
    # print(scores_np[:100], gt_np[:100])
    auc, shift, sigma, fpr_res, tpr_res = score_align(scores_np, gt_np)
    # print(auc)
    return auc, shift, sigma, fpr_res, tpr_res

def score_align0(scores_np, gt, seg_len=6, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    # shift = 6
    scores_shifted[shift:] = scores_np[:-shift]
    scores_smoothed = gaussian_filter1d(scores_shifted, sigma)
    # scores_smoothed = np.max(scores_smoothed) - scores_smoothed
    auc = roc_auc_score(gt, scores_smoothed)
    return auc, shift, sigma

def score_align(scores, labels):
    best_auc = -1
    best_sigma = 0
    best_shift = 0
    plot_material = None
    fpr_res = None
    tpr_res = None
    for shift in range(4, 5, 1):
    # for shift in range(4, 20, 1):
        scores[shift:] = scores[:-shift]
        # for sigma in range(1, 1050, 1):  # sigma从0.1到100,0.1为步长，寻找最佳sigma
        for sigma in range(1, 200, 100):  # sigma从0.1到100,0.1为步长，寻找最佳sigma
            sigma = sigma/10
            scores_gau = gaussian_filter1d(scores, sigma)
            print(labels.shape, scores_gau.shape)
            fpr, tpr, threshold = metrics.roc_curve(labels.tolist(), scores_gau.tolist())
            auc_gau = metrics.auc(fpr, tpr)
            # auc_gau = roc_auc_score(labels, scores_gau)
            if auc_gau > best_auc:
                best_auc = auc_gau
                best_sigma = sigma
                best_shift = shift
                fpr_res = fpr
                tpr_res = tpr
    return best_auc, best_shift, best_sigma, fpr_res, tpr_res

def score_align1(scores, labels):
    best_auc = -1
    best_sigma = 0
    best_shift = 0
    for shift in range(4, 20, 1):
        scores[shift:] = scores[:-shift]
        for sigma in range(1, 1050, 1):  # sigma从0.1到100,0.1为步长，寻找最佳sigma
            sigma = sigma/10
            scores_gau = gaussian_filter1d(scores, sigma)
            auc_gau = roc_auc_score(labels, scores_gau)
            if auc_gau > best_auc:
                best_auc = auc_gau
                best_sigma = sigma
                best_shift = shift
    return best_auc, best_shift, best_sigma

def shift_gaussian_filter(scores, shift=0, sigma=33):
    scores[shift:] = scores[:-shift]
    scores = gaussian_filter1d(scores, sigma)
    return scores


def score_dataset_mix(dp_scores_high, metadata_high, loss_high, dp_scores_low, metadata_low, loss_low, ratios, normalize=False):
    gt_arr_low, scores_arr_low, score_ids_arr, metadata_arr = get_dataset_scores(dp_scores_low, metadata_low)
    gt_arr_low, loss_arr_low, score_ids_arr, metadata_arr = get_dataset_scores(loss_low, metadata_low)
    gt_arr_high, scores_arr_high, score_ids_arr, metadata_arr_high = get_frame_dataset_scores(dp_scores_high, metadata_high)
    
    # print(len(loss_high))
    # print(len(loss_low))
    # print(len(metadata_low))
    gt_np_low = np.concatenate(gt_arr_low)
    scores_np_low = np.concatenate(scores_arr_low)
    loss_low = np.concatenate(loss_arr_low)
    gt_np_high = np.concatenate(gt_arr_high)
    scores_np_high_ori = np.concatenate(scores_arr_high)
    loss_high_ori = np.array(loss_high)
    metadata_arr_high = np.concatenate(metadata_arr_high)
    
    print(normalize)
    if normalize:
        print(scores_np_low.shape, loss_low.shape)
        # scores_np_low = min_max_normalize(scores_np_low)
        # loss_low = min_max_normalize(loss_low)
        loss_low = loss_low / np.max(loss_low)
        scores_np_low = scores_np_low / np.max(scores_np_low)
    
    # shift, sigma = 15, 33
    # scores_np_low = shift_gaussian_filter(scores_np_low, shift=shift, sigma=sigma)
    # loss_low = shift_gaussian_filter(loss_low, shift=shift, sigma=sigma)
    
    aucs = []
    shifts_scores_low = []
    sigmas_scores_low = []
    shifts_loss_low = []
    sigmas_loss_low = []
    output_ratios = []

    # shifts = list(range(4, 20, 1))
    # sigmas = list(range(1, 1050, 1))
    shifts = list(range(4, 7, 1))
    sigmas = list(range(20, 35, 1))
    
    # shifts = list(range(4, 6, 1))
    # sigmas = list(range(1, 11, 5))
    
    for ratio in ratios:
        for shifts_scores_low_one in shifts:
            for sigmas_scores_low_one in sigmas:
                for shifts_loss_low_one in shifts:
                    for sigmas_loss_low_one in sigmas:
                        scores_np_low = shift_gaussian_filter(scores_np_low, shift=shifts_scores_low_one, sigma=sigmas_scores_low_one)
                        loss_low = shift_gaussian_filter(loss_low, shift=shifts_loss_low_one, sigma=sigmas_loss_low_one)
                        scores_np = ratio[0] * loss_low + ratio[1] * scores_np_low
                        auc = roc_auc_score(gt_np_low, scores_np)
        
                        aucs.append(auc)
                        shifts_scores_low.append(shifts_scores_low_one)
                        sigmas_scores_low.append(sigmas_scores_low_one)
                        shifts_loss_low.append(shifts_loss_low_one)
                        sigmas_loss_low.append(sigmas_loss_low_one)
                        output_ratios.append(ratio)
                        print(auc, shifts_scores_low_one, sigmas_scores_low_one, shifts_loss_low_one, sigmas_loss_low_one, ratio)
    # print(auc)
    return aucs, shifts_scores_low, sigmas_scores_low, shifts_loss_low, sigmas_loss_low, output_ratios

def score_dataset_mix0(dp_scores_high, metadata_high, loss_high, dp_scores_low, metadata_low, loss_low, ratios, normalize=False):
    gt_arr_low, scores_arr_low, score_ids_arr, metadata_arr = get_dataset_scores(dp_scores_low, metadata_low)
    gt_arr_low, loss_arr_low, score_ids_arr, metadata_arr = get_dataset_scores(loss_low, metadata_low)
    gt_arr_high, scores_arr_high, score_ids_arr, metadata_arr_high = get_frame_dataset_scores(dp_scores_high, metadata_high)
    
    # print(len(loss_high))
    # print(len(loss_low))
    # print(len(metadata_low))
    gt_np_low = np.concatenate(gt_arr_low)
    scores_np_low = np.concatenate(scores_arr_low)
    loss_low = np.concatenate(loss_arr_low)
    gt_np_high = np.concatenate(gt_arr_high)
    scores_np_high_ori = np.concatenate(scores_arr_high)
    loss_high_ori = np.array(loss_high)
    metadata_arr_high = np.concatenate(metadata_arr_high)
    
    print(normalize)
    if normalize:
        print(scores_np_low.shape, loss_low.shape)
        scores_np_low = min_max_normalize(scores_np_low)
        loss_low = min_max_normalize(loss_low)
    #     # print(scores_arr_low.shape, scores_arr_high.shape)
    #     loss_high_ori = loss_high_ori / np.max(loss_high_ori)
    #     loss_low = loss_low / np.max(loss_low)
    #     scores_np_low = scores_np_low / np.max(scores_np_low)
    #     scores_np_high_ori = scores_np_high_ori / np.max(scores_np_high_ori)
    
    shift, sigma = 8, 33
    scores_np_low = shift_gaussian_filter(scores_np_low, shift=shift, sigma=sigma)
    loss_low = shift_gaussian_filter(loss_low, shift=shift, sigma=sigma)
    
    # print('gt_np_low:{}, scores_np_low:{}, gt_np_high:{}, scores_np_high:{}, metadata_arr_high:{} {}'.format(gt_np_low.shape, scores_np_low.shape, 
    #                                                         gt_np_high.shape, scores_np_high_ori.shape, metadata_arr_high.shape, sum(metadata_arr_high)))
    aucs = []
    for ratio in ratios:
        scores = []
        scores_np_high = scores_np_high_ori
        loss_high = loss_high_ori
        # print(gt_np_low.shape[0])
        # print(metadata_arr_high)
        
        # for i in range(gt_np_low.shape[0]):
        #     # print('****** {}'.format(i))
        #     # print(metadata_arr_high[i])
        #     if int(metadata_arr_high[i]) == 1:
        #         # print('>>>>>>')
        #         # print(scores_np_high.shape)
        #         # print(scores_np_high[i], scores_np_low[i])
        #         score = ratio[3] * loss_low[i] + ratio[1] * scores_np_low[i]# + ratio[2] * loss_high[0] + ratio[0] * scores_np_high[0]
        #         scores_np_high = scores_np_high[1:]
        #         loss_high = loss_high[1:]
        #         # print(scores_np_high.shape)
        #         # print('<<<<<<<')
        #     else:
        #         # print('######')
        #         # print(i)
        #         score = ratio[3] * loss_low[i] + ratio[1] * scores_np_low[i]
            
        #     scores.append(score)
        # scores_np = np.array(scores)
        # auc, shift, sigma = score_align(scores_np, gt_np_low)
        
        scores_np = ratio[0] * loss_low + ratio[1] * scores_np_low
        auc = roc_auc_score(gt_np_low, scores_np)
        
        aucs.append(auc)
    # print(auc)
    return aucs, shift, sigma

def filter_hr_avenue0(input, data_id):
    if data_id == '04_01':
        output = np.concatenate([input[:75], input[120:390], input[436:864], input[910:931], input[1000:]])
    elif data_id == '01_02':
        output = np.concatenate([input[:272], input[319:723], input[763:]])
    elif data_id == '01_03':
        output = np.concatenate([input[:293], input[340:]])
    elif data_id == '02_06':
        output = np.concatenate([input[:561], input[624:814], input[1006:]])
    elif data_id == '02_16':
        output = np.concatenate([input[:728], input[739:]])
    else:
        output = input
    return output

def filter_hr_avenue(input, data_id):
    if data_id == '04_01':
        output = np.concatenate([input[:75], input[120:390], input[436:864], input[910:931], input[1000:]])
    elif data_id == '01_02':
        output = np.concatenate([input[:272], input[319:723], input[763:]])
    elif data_id == '05_03':
        output = np.concatenate([input[:293], input[340:]])
    elif data_id == '04_06':
        output = np.concatenate([input[:561], input[624:814], input[1006:]])
    elif data_id == '04_16':
        output = np.concatenate([input[:728], input[739:]])
    else:
        output = input
    return output
    

def get_dataset_scores(scores, metadata, max_clip=None, scene_id=None, dataset='shanghai'):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    if dataset in ['shanghai', 'HR_shanghai']:
        per_frame_scores_root = '../gepc/data/testing/test_frame_mask/'
    elif dataset in ['avenue', 'HR_avenue']:
        per_frame_scores_root = '/home/yaboliu/data/cvae/avenue/testing/labels/'
    elif dataset == 'ucsdped1':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/labels/'
    elif dataset == 'ucsdped2':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/labels/'
    elif dataset == 'corridor':
        per_frame_scores_root = '/home/yaboliu/data/cvae/corridor/testing/labels'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')]
        if dataset == 'HR_shanghai':
            # if scene_id + '_' + clip_id in HR_shanghai_except:
            
            #     print('filter_shagnhai_hr_1')
            #     continue
            if clip.split('.')[0] in HR_shanghai_except:
                print('filter_shagnhai_hr_2')
                continue
        
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        scores_zeros = np.zeros(clip_gt.shape[0])
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        if len(list(clip_person_scores_dict.values())) < 1:
            continue
        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        
        
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        # clip_score = np.amin(clip_ppl_score_arr, axis=0)
        
        #在此加入HR_avenue的判定
        if dataset == 'HR_avenue':
            clip_gt = filter_hr_avenue(clip_gt, clip.split('.')[0])
            clip_score = filter_hr_avenue(clip_score, clip.split('.')[0])
            if clip_gt is None:
                continue

        
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr

def get_dataset_scores_low(scores, metadata, metadata_high, max_clip=None, scene_id=None, dataset='shanghai'):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    # metadata_np = np.array(metadata)
    if dataset in ['shanghai', 'HR_shanghai']:
        per_frame_scores_root = '../gepc/data/testing/test_frame_mask/'
    elif dataset in ['avenue', 'HR_avenue']:
        per_frame_scores_root = '/home/yaboliu/data/cvae/avenue/testing/labels/'
    elif dataset == 'ucsdped1':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/labels/'
    elif dataset == 'ucsdped2':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/labels/'
    elif dataset == 'corridor':
        per_frame_scores_root = '/home/yaboliu/data/cvae/corridor/testing/labels'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')]
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        scores_zeros = np.zeros(clip_gt.shape[0])
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        
        
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        # clip_score = np.amin(clip_ppl_score_arr, axis=0)
        
        
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr

def get_frame_dataset_scores(scores, metadata, max_clip=None, scene_id=None, dataset='shanghai'):
    # np.save('scores.npy', scores)
    # np.save('metadata.npy', metadata)
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    clip_gt_arr = []
    metadata_np = np.array(metadata)
    if dataset in ['shanghai', 'HR_shanghai']:
        per_frame_scores_root = '../gepc/data/testing/test_frame_mask/'
    elif dataset in ['avenue', 'HR_avenue']:
        per_frame_scores_root = '/home/yaboliu/data/cvae/avenue/testing/labels/'
    elif dataset == 'ucsdped1':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped1/Test/labels/'
    elif dataset == 'ucsdped2':
        per_frame_scores_root = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/labels/'
    elif dataset == 'corridor':
        per_frame_scores_root = '/home/yaboliu/data/cvae/corridor/testing/labels'
    
    
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        # print('>' * 10)
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn) 
        gt_indexes = np.zeros(clip_gt.shape[0])
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')]
        if dataset == 'HR_shanghai':
            # if scene_id + '_' + clip_id in HR_shanghai_except:
            #     print('filter_shagnhai_hr_1')
            #     continue
            if clip.split('.')[0] in HR_shanghai_except:
                print('filter_shagnhai_hr_2')
                continue
        # print(clip_gt[:3])
        # print(metadata[:3])
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        # print(clip_metadata_inds.shape)
        clip_metadata = metadata[clip_metadata_inds]
        clip_fig_idxs = sorted([arr[3] for arr in clip_metadata])
        # print('>>>>')
        # print(clip_gt.shape)
        
        temp_gt = clip_gt[clip_fig_idxs]
        #在此加入HR_avenue的判定
        if dataset == 'HR_avenue':
            temp_gt = filter_hr_avenue(temp_gt, clip.split('.')[0])
            if temp_gt is None:
                continue
        clip_gt_arr.append(clip_gt)
        dataset_gt_arr.append(temp_gt)
        gt_indexes[clip_fig_idxs] = 1
        dataset_score_ids_arr.append(gt_indexes)
        # print(sum(gt_indexes))
        # for i in clip_fig_idxs:
        #     dataset_gt_arr.append(np.array(clip_gt[i]))
        # scores_zeros = np.zeros(clip_gt.shape[0])
        # clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        clip_scores = []
        for frame_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 3] == frame_id))[0]
            pid_scores = scores[person_metadata_inds]
            clip_scores.append(pid_scores)
            
        clip_scores = np.array(clip_scores).reshape(-1)
        # print(clip_scores.shape)
        #在此加入HR_avenue的判定
        if dataset == 'HR_avenue':
            clip_scores = filter_hr_avenue(clip_scores, clip.split('.')[0])
        dataset_scores_arr.append(clip_scores)
            # pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            # clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        # clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        
        
        # clip_score = np.amax(clip_ppl_score_arr, axis=0)
        # clip_score = np.amin(clip_ppl_score_arr, axis=0)
        
        
        # fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        # dataset_gt_arr.append(clip_gt)
        # dataset_scores_arr.append(pid_scores)
        # dataset_score_ids_arr.append(fig_score_id)
        # dataset_metadata_arr.append([scene_id, clip_id])
        dataset_metadata_arr.append(clip.split('.')[0])
    # dataset_gt_arr = np.array(dataset_gt_arr)
    # dataset_scores_arr = np.array(dataset_scores_arr)
    # print(dataset_gt_arr.shape)
    # print(dataset_scores_arr.shape)
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr, clip_gt_arr

def score_align3(scores, labels):
    best_auc = -1
    best_sigma = 0
    best_shift = 0
    for shift in range(4, 20, 1):
        scores[shift:] = scores[:-shift]
        for sigma in range(1, 1050, 1):  # sigma从0.1到100,0.1为步长，寻找最佳sigma
            sigma = sigma/10
            scores_gau = gaussian_filter1d(scores, sigma)
            # fpr_gau, tpr_gau, thresholds_gau = metrics.roc_curve(
            #     labels, scores_gau, pos_label=0)
            # auc_gau = metrics.auc(fpr_gau, tpr_gau)
            auc_gau = roc_auc_score(labels, scores_gau)
            if auc_gau > best_auc:
                best_auc = auc_gau
                best_sigma = sigma
                best_shift = shift
    return best_auc, best_shift, best_sigma

def min_max_normalize(arr):
    max_a = np.max(arr)
    min_a = np.min(arr)
    if min_a == max_a:
        return np.ones(shape=arr.shape)
    return (arr - min_a) / (max_a - min_a)

def score_align0(scores_np, gt, seg_len=6, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    scores_shifted[shift:] = scores_np[:-shift]
    
    # scores_shifted = scores_shifted / np.max(scores_shifted)
    # scores_shifted = min_max_normalize(scores_shifted)
    scores_smoothed = gaussian_filter1d(scores_shifted, sigma)
    auc = roc_auc_score(gt, scores_smoothed)
    
    # auc = roc_auc_score(gt, scores_np)
    return auc, shift, sigma


def avg_scores_by_trans(scores, gt, num_transform=5, ret_first=False):
    score_mask, scores_by_trans, scores_tavg = dict(), dict(), dict()
    gti = {'normal': 1, 'abnormal': 0}
    for k, gt_val in gti.items():
        score_mask[k] = scores[gt == gt_val]
        scores_by_trans[k] = score_mask[k].reshape(-1, num_transform)
        scores_tavg[k] = scores_by_trans[k].mean(axis=1)
        print(k, score_mask[k].shape, scores_by_trans[k].shape, scores_tavg[k].shape)

    gt_trans_avg = np.concatenate([np.ones_like(scores_tavg['normal'], dtype=np.int),
                                   np.zeros_like(scores_tavg['abnormal'], dtype=np.int)])
    scores_trans_avg = np.concatenate([scores_tavg['normal'], scores_tavg['abnormal']])
    if ret_first:
        scores_first_trans = dict()
        for k, v in scores_by_trans.items():
            scores_first_trans[k] = v[:, 0]
        scores_first_trans = np.concatenate([scores_first_trans['normal'], scores_first_trans['abnormal']])
        return scores_trans_avg, gt_trans_avg, scores_first_trans
    else:
        return scores_trans_avg, gt_trans_avg
    
def draw_graph(joints_input, label=None, img_size=[500], img=None, color=(0, 0, 255)):
    if len(img_size) == 1:
        img_size = [img_size[0], img_size[1]]
    neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                     (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]# + [(15, 0), (14, 0), (17, 15), (16, 14)]
    if img is None:
        img = np.ones((img_size[1], img_size[0], 3), np.uint8) * 255
    joints = []
    # print(label)
    # print(joints_input)
    for joint in joints_input:
        # print(joint)
        i, j = joint
        # print(i, j)
        
        joints.append([int(i * img_size[0]), int(j * img_size[1])])
    # print(joints)
    # print(label)
        
    for joint in joints:
        cv2.circle(img, joint, 4, color, -1)
    for link in neighbor_link:
        cv2.line(img, joints[link[0]], joints[link[1]], color, 2)
    if label:
        cv2.putText(img, label, (10, 50), 0, 1, color, 2)
    # cv2.line(img, [img_size[0] - 2, 0], [img_size[0] - 2, img_size[1]], (0, 0, 0), 2)
    return img

def make_show(data, reco_data, loss, img_size=[856, 480], graph_img_size=[856, 480], img_ls=None, resize=0.5):
    # graph_img_size=[500, 500]
    show_img_ls = []
    # print(data.shape)
    # print('draw: ', data.shape)
    data = data.permute(0, 2, 3, 1).contiguous().tolist()   # n t v c
    data = data[0]
    # print(data[:][-1][:])
    # for one_data in data:
    img_pred = img_ls[-1].copy()
    for i in range(len(data)):
        one_data = data[i]
        img_np = img_ls[i]
        # print(one_data)
        center = [int(one_data[-1][0] * img_size[0]), int(one_data[-1][1] * img_size[1])]
        # one_img = draw_graph(one_data[:-1][:], label=str(center), img_size=graph_img_size)
        one_img = draw_graph(one_data[:][:], label=str(center), img_size=graph_img_size, img=img_np)
        show_img_ls.append(one_img)
    
    reco_data = reco_data.permute(0, 3, 2, 1).contiguous()
    # print('------->>>>>')
    # print(reco_data.shape)
    reco_data = reco_data.squeeze(2).tolist()[0]
    # print(reco_data)
    pred_position = [reco_data[-1][0] * img_size[0], reco_data[-1][1] * img_size[1]]
    # gt_position = [data[-1][-1][0] * img_size[0], data[-1][-1][1] * img_size[1]]
    # print(pred_position)
    # print(gt_position)
    pred_position = [int(i) for i in pred_position]
    # gt_position = [int(i) for i in gt_position]
    # pred_img = draw_graph(reco_data[:-1][:], label=str(pred_position), img_size=graph_img_size)
    pred_img = draw_graph(reco_data[:][:], label=str(pred_position), img_size=graph_img_size, img=img_pred)
    cv2.line(pred_img, [1, 0], [1, graph_img_size[1]], (0, 255, 0), 2)
    
    show_img = np.concatenate(show_img_ls + [pred_img], axis=0)
    resize_size = (int(show_img.shape[1] * resize), int(show_img.shape[0] * resize))
    show_img = cv2.resize(show_img, resize_size)
    return show_img

def make_show_forpaper(data, reco_data, loss, img_size=[856, 480], graph_img_size=[856, 480], img_ls=None, resize=0.5):
    data = data.permute(0, 2, 3, 1).contiguous().tolist()   # n t v c
    data = data[0]
    img_pred = img_ls[-1].copy()
    img_ori = img_pred.copy()
    one_data = data[-1]
    img_pred = draw_graph(one_data[:][:], img_size=graph_img_size, img=img_pred, color=(255, 0, 0))
    reco_data = reco_data.permute(0, 3, 2, 1).contiguous()
    reco_data = reco_data.squeeze(2).tolist()[0]
    pred_img = draw_graph(reco_data[:][:], img_size=graph_img_size, img=img_pred, color=(0, 0, 255))
    pred_img = np.concatenate([pred_img, img_ori], axis=0)
    resize_size = (int(pred_img.shape[1] * resize), int(pred_img.shape[0] * resize))
    pred_img = cv2.resize(pred_img, resize_size)

    return pred_img

    # show_img_ls += [pred_img]
    # return show_img_ls
    
def make_show_high(data, reco_data, loss, img_size=[856, 480], graph_img_size=[856, 480], img_ls=None, resize=0.5):
    # data ncptv
    show_img_ls = []
    # print(data.shape, reco_data.shape)
    data = data.permute(0, 1, 3, 2, 4).contiguous()
    data, reco_data = data.squeeze(4), reco_data.squeeze(4) # nctp
    data = data.permute(0, 2, 3, 1).contiguous().tolist()   # n t p c
    reco_data = reco_data.permute(0, 2, 3, 1).contiguous().tolist()   # n t p c
    data, reco_data = data[0], reco_data[0]
    # print(data)
    # print(reco_data)
    # print(data[:][-1][:])
    # for one_data in data:
    # print('one show', graph_img_size)
    img_pred = img_ls[-1].copy()
    for i in range(len(data)):
        one_data = data[i]
        img_np = img_ls[i]
        for one_person in one_data:
            x, y = [int(one_person[0] * graph_img_size[0]), int(one_person[1] * graph_img_size[1])]
            cv2.circle(img_np, (x, y), 6, (0, 0, 255), -1)
        show_img_ls.append(img_np)
    
    for one_person in reco_data[0]:
        x, y = [int(one_person[0] * graph_img_size[0]), int(one_person[1] * graph_img_size[1])]
        cv2.circle(img_pred, (x, y), 6, (0, 0, 255), -1)
    cv2.line(img_pred, [1, 0], [1, graph_img_size[1]], (0, 255, 0), 2)
    
    show_img = np.concatenate(show_img_ls + [img_pred], axis=0)
    resize_size = (int(show_img.shape[1] * resize), int(show_img.shape[0] * resize))
    show_img = cv2.resize(show_img, resize_size)
    return show_img
