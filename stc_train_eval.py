import os
from pickle import FALSE, NONE
import random
import collections
import numpy as np
from itertools import product

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

from models.gcae.gcae import Encoder
from models.fe.fe_model import init_fenet

from models.dc_gcae.dc_gcae import DC_GCAE, load_ae_dcec
from models.dc_gcae.dc_gcae_training import dc_gcae_train
from models.gcae.gcae_training import Trainer

from utils.data_utils import ae_trans_list
from utils.train_utils import get_fn_suffix, init_clusters
from utils.train_utils import csv_log_dump
from utils.scoring_utils import dpmm_calc_scores, score_dataset, avg_scores_by_trans, score_dataset_mix
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_seg_hight_dataset import PoseHighSegDataset
from utils.pose_ad_argparse import init_stc_parser, init_stc_sub_args
from utils.optim_utils.optim_init import init_optimizer, init_scheduler

# def make_ratios(ratios, num_block=2):
#     if num_block == 0:
#         return
#     for ratio in ratios:
        

def main():
    parser = init_stc_parser()
    args = parser.parse_args()
    log_dict = collections.defaultdict(int)

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.add_scale and args.in_channels <= 3:
        args.in_channels = args.in_channels + 2
    args, ae_args, dcec_args, res_args = init_stc_sub_args(args)
    print(args)
    with open(args.log_path, 'a') as f:
        f.write(str(args)+'\n')

    
    high_low_eval = vars(args).get('high_low_eval', False)
    if high_low_eval:
        high_model_path = '/home/yaboliu/work/research/gepc_hl/work_dir_test1/gtae_pred_cluster/stc/Dec24_1950/checkpoints/Dec24_1950_stc_sagc_checkpoint/done5_Dec24_1957_dcec10_1_checkpoint.pth.tar'
        low_model_path = '/home/yaboliu/work/research/gepc/work_dir_test1/gtae_pred_cluster/stc/Dec23_1745/checkpoints/Dec23_1745_stc_sagc_checkpoint/done13_Dec23_1750_dcec10_1_checkpoint.pth.tar'
        low_loss_model_path = ''
        # ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # ratios = [[i, 1-i] for i in ratio]
        
        ratio = list(range(0, 105, 5))
        ratio = [i / 100 for i in ratio]
        # ratio = [1, 0]
        normalize = True
        use_loss = True
        
        if use_loss:
            input_ratios = []
            # ratio = (list(product(ratio, ratio, ratio, ratio)))
            ratios = (list(product(ratio, ratio)))
            for i in ratios:
                if sum(i) == 1:
                    input_ratios.append(i)
        
        # print(input_ratios)
        # dc_gcae_high = load_ae_dcec(high_model_path, load_level='high')
        # dc_gcae_low = load_ae_dcec(low_model_path, load_level='low')
        
        # dataset, loader = get_dataset_and_loader(ae_args, level='low')
        # dp_scores_low, gt, metadata_low, loss_low = dpmm_calc_scores(dc_gcae_low, dataset['train'], dataset['test'], args=res_args, ret_metadata=True, pred_loss=True)
        # np.save('dp_scores_low', dp_scores_low)
        # np.save('metadata_low', metadata_low)
        # np.save('loss_low', loss_low)
        
        # dataset, loader = get_dataset_and_loader(ae_args, level='high')
        # res_args.batch_size = 1
        # dp_scores_high, gt, metadata_high, loss_high = dpmm_calc_scores(dc_gcae_high, dataset['train'], dataset['test'], args=res_args, ret_metadata=True, pred_loss=True)
        # np.save('dp_scores_high', dp_scores_high)
        # np.save('metadata_high', metadata_high)
        # np.save('loss_high', loss_high)

        dp_scores_high, metadata_high  = np.load('dp_scores_high.npy'), np.load('metadata_high.npy')
        dp_scores_low, metadata_low = np.load('dp_scores_low.npy'), np.load('metadata_low.npy')
        loss_high, loss_low = np.load('loss_high.npy'), np.load('loss_low.npy')
        # loss_low = np.load('/home/yaboliu/work/research/gepc_hl/work_dir_candidate/gtae_pred_cluster/pred/stc/Dec28_1216/checkpoints/0_reco_loss.npy')
        print('dp_scores_high: ', dp_scores_high.shape, 'dp_scores_low: ', dp_scores_low.shape)
        
        # dp_auc, dp_shift, dp_sigma = score_dataset(loss_low, metadata_low)
        # print(dp_auc, dp_shift, dp_sigma)
        # return

        # dp_aucs, dp_shift, dp_sigma = score_dataset_mix(dp_scores_high, metadata_high, None, dp_scores_low, metadata_low, None, 
        #                                                 ratios=ratios, normalize=normalize)
        dp_aucs, shifts_scores_low, sigmas_scores_low, shifts_loss_low, sigmas_loss_low, output_ratios = score_dataset_mix(
            dp_scores_high, metadata_high, loss_high, dp_scores_low, metadata_low, loss_low, ratios=input_ratios, normalize=normalize)

        max_auc = 0
        max_idx = 0
        for i in range(len(dp_aucs)):
            if dp_aucs[i] > max_auc:
                max_auc = dp_aucs[i]
                max_idx = i
            # print('ratio: {}, auc: {:.4f}'.format(output_ratios[i], dp_aucs[i]))
        print('Best: auc {:.4f} shifts_scores_low {}, sigmas_scores_low {}, shifts_scores_low {}, sigmas_scores_low {}, output_ratios {}'.format(
            max_auc, shifts_scores_low[max_idx], sigmas_scores_low[max_idx], shifts_loss_low[max_idx], sigmas_loss_low[max_idx], str(output_ratios[max_idx])))
        with open(args.log_path, 'a') as f:
            f.write('Best: auc {:.4f} shifts_scores_low {}, sigmas_scores_low {}, shifts_scores_low {}, sigmas_scores_low {}, output_ratios {}\n'.format(
            max_auc, shifts_scores_low[max_idx], sigmas_scores_low[max_idx], shifts_loss_low[max_idx], sigmas_loss_low[max_idx], str(output_ratios[max_idx])))
        return 
        
        
        
        
    level = args.level
    dataset, loader = get_dataset_and_loader(ae_args, level)

    ae_fn = vars(args).get('ae_fn', None)
    dcec_fn = vars(args).get('dcec_fn', None)
    resume = vars(args).get('resume', False)

    if dcec_fn:  # Load pretrained models
        pretrained = True
        dc_gcae = load_ae_dcec(dcec_fn)
        args.ae_fn = dcec_fn.split('/')[-1]
        res_args.ae_fn = dcec_fn.split('/')[-1]
    else:
        pretrained = False
        if ae_fn:  # Load pretrained AE and train DCEC
            fe_model = init_fenet(args)
            fe_model.load_checkpoint(ae_fn)
        else:  # Train an AE
            backbone = 'resnet' if args.patch_features else None
            model = init_fenet(args, backbone, level=level )
            # print(model)

            loss = nn.MSELoss()
            ae_optimizer_f = init_optimizer(args.ae_optimizer, lr=args.ae_lr)
            ae_scheduler_f = init_scheduler(args.ae_sched, lr=args.ae_lr, epochs=args.ae_epochs)
            trainer = Trainer(ae_args, model, loss, loader['train'], loader['test'], dataset['test'], optimizer_f=ae_optimizer_f,
                              scheduler_f=ae_scheduler_f, fn_suffix=get_fn_suffix(args))
            ae_fn, log_dict['F_ae_loss'] = trainer.train(checkpoint_filename=ae_fn, args=ae_args)
            args.ae_fn = dcec_args.ae_fn = res_args.ae_fn = ae_fn
            fe_model = trainer.model

            # if args.new_preprocess:
            #     return

        # Train DCEC models
        encoder = Encoder(model=fe_model).to(args.device)
        hidden_dim, initial_clusters = init_clusters(dataset, dcec_args, encoder)
        dc_gcae = DC_GCAE(fe_model, hidden_dim, n_clusters=args.n_clusters, initial_clusters=initial_clusters)
        del fe_model
        _, log_dict['F_delta_labels'], log_dict['F_dcec_loss'] = dc_gcae_train(dc_gcae, dataset['train'], dataset['test'], dcec_args, res_args)
        
    if resume:
        dcec_args.ae_fn = dcec_fn
        _, log_dict['F_delta_labels'], log_dict['F_dcec_loss'] = dc_gcae_train(dc_gcae, dataset['train'], dcec_args)

    # Normality scoring phase
    dc_gcae.eval()
    if pretrained and getattr(args, 'dpmm_fn', False):
        pt_dpmm = args.dpmm_fn
    else:
        pt_dpmm = None

    dp_scores, gt, metadata = dpmm_calc_scores(dc_gcae, dataset['train'], dataset['test'],
                                               args=res_args, ret_metadata=True, pt_dpmm_path=pt_dpmm)

    dp_scores_tavg, _ = avg_scores_by_trans(dp_scores, gt, args.num_transform)
    max_clip = 5 if args.debug else None
    dp_auc, dp_shift, dp_sigma = score_dataset(dp_scores_tavg, metadata, max_clip=max_clip, dataset=args.dataset)

    # Logging and recording results
    print("Done with {} AuC for {} samples and {} trans".format(dp_auc, dp_scores_tavg.shape[0], args.num_transform));
    log_dict['dp_auc'] = 100 * dp_auc
    csv_log_dump(args, log_dict)


def get_dataset_and_loader(args, level='low'):
    patch_size = int(args.patch_size)
    if args.patch_db:
        patch_suffix_str = 'ing{}x{}.lmdb'.format(patch_size, patch_size)
        patch_size = (patch_size, patch_size)
        patch_db_path = {k: os.path.join(v, k+patch_suffix_str) for k, v in args.vid_path.items()}
    else:
        patch_db_path = {k: None for k, v in args.vid_path.items()}

    trans_list = ae_trans_list[:args.num_transform]

    dataset_args = {'transform_list': trans_list, 'debug': args.debug, 'headless': args.headless,
                    'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale, 'seg_len': args.seg_len,
                    'patch_size': patch_size, 'return_indices': True, 'return_metadata': True,
                    'new_preprocess': args.new_preprocess, 'only_normal': args.only_normal, 'add_center': args.add_center, 'filter_bbox': args.filter_bbox,
                    'dataset': args.dataset, 'add_scale': args.add_scale, 'in_channels': args.in_channels, 'filter_border': args.filter_border,
                    'filter_independent': args.filter_independent, 'filter_cover': args.filter_cover, 'dataset_sence': args.dataset_sence, 
                    'high_filter_continuous': args.high_filter_continuous, 'vid_res': args.vid_res, 'use_bbox_high_level': args.use_bbox_high_level,
                    'use_filter': args.use_filter
                    }

    
    if args.level == 'high':
        batch_size = 1
    else:
        batch_size = args.batch_size
        
    loader_args = {'batch_size': batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset, loader = dict(), dict()
    for split in ['train', 'test']:
        dataset_args['seg_stride'] = args.seg_stride if split is 'train' else 1  # No strides for test set
        dataset_args['train_seg_conf_th'] = args.train_seg_conf_th if split is 'train' else 0.0
        if split == 'train':
            if args.train_dataset_sence is None:
                dataset_sence = args.dataset_sence
            else:
                dataset_sence = args.train_dataset_sence
            dataset_args['dataset_sence'] = dataset_sence
        else:
            dataset_args['dataset_sence'] = args.dataset_sence
                
        if args.patch_features:
            dataset[split] = PoseSegDataset(args.pose_path[split], args.vid_path[split], patch_db_path[split],
                                            **dataset_args)
        else:
            if level == 'high':
                dataset[split] = PoseHighSegDataset(args.pose_path[split], **dataset_args)
            elif level == 'low':
                print('dataset....')
                dataset[split] = PoseSegDataset(args.pose_path[split], **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    return dataset, loader

def collate_fn(data):  # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    data.sort(key=lambda x: len(x[0][0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    label_list = []
    min_len = len(data[0][0][0]) # 最短的数据长度 
    for batch in range(0, len(data)): #
        data_list.append(data[batch][0][:, :min_len])
        label_list.append(data[batch][1])
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy


def save_result_npz(args, scores, scores_tavg, metadata, sfmax_maxval, auc, dp_auc=None):
    debug_str = '_debug' if args.debug else ''
    auc_int = int(1000 * auc)
    dp_auc_str = ''
    if dp_auc is not None:
        dp_auc_int = int(1000 * dp_auc)
        dp_auc_str = '_dp{}'.format(dp_auc_int)
    auc_str = '_{}'.format(auc_int)
    res_fn = args.ae_fn.split('.')[0] + '_res{}{}{}.npz'.format(dp_auc_str, auc_str, debug_str)
    res_path = os.path.join(args.ckpt_dir, res_fn)
    np.savez(res_path, scores=scores, sfmax_maxval=sfmax_maxval, args=args, metadata=metadata,
             scores_tavg=scores_tavg, dp_best=dp_auc)


if __name__ == '__main__':
    main()

