import os
import time
import argparse


def init_stc_args():
    parser = init_stc_parser()
    args = parser.parse_args()
    return init_stc_sub_args(args)


def init_stc_sub_args(args):
    if args.patch_features:  # No pose augmentations for patch models
        args.num_transform = 1
    if args.debug:
        args.ae_epochs = 10
        args.dcec_epochs = 25

    # shanghai
    # avenue
    dataset = args.dataset
    if dataset in ['shanghai', 'HR_shanghai']:
        args.data_dir = '/home/yaboliu/data/cvae/shanghaitech/'
        args.vid_path = {'train': [os.path.join(args.data_dir, 'training/frames/'), 2, 3, 4],
                     'test':  [os.path.join(args.data_dir, 'testing/frames/'), 2, 4, 3]}    # scene_len, clip_len, img_len
        # args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose3/json_results/'),
        #               'test':  os.path.join(args.data_dir, 'testing/pose3/json_results/')}
        args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose_hrnet48/json_results/'),
                      'test':  os.path.join(args.data_dir, 'testing/pose_hrnet48/json_results/')}
        args.use_filter = False
        
    elif dataset in ['avenue', 'HR_avenue']:
        args.data_dir = '/home/yaboliu/data/cvae/avenue/'
        args.vid_path = {'train': [os.path.join(args.data_dir, 'training/frames/'), 2, 2, 4],
                        'test':  [os.path.join(args.data_dir, 'testing/frames/'), 2, 2, 4]}    # scene_len, clip_len, img_len
        # args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose/json_results/'),
        #                 'test':  os.path.join(args.data_dir, 'testing/pose/json_results/')}
        args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose/json_results/'),
                        'test':  os.path.join(args.data_dir, 'testing/pose_refine1_5/json_results/')}
    elif dataset == 'ucsdped1':
        args.data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped1/'
        args.vid_path = {'train': [os.path.join(args.data_dir, 'Train/frames/'), 2, 3, 3],
                        'test':  [os.path.join(args.data_dir, 'Test/frames/'), 2, 3, 3]}    # scene_len, clip_len, img_len
        args.pose_path = {'train': os.path.join(args.data_dir, 'Train/pose/json_results/'),
                        'test':  os.path.join(args.data_dir, 'Test/pose/json_results/')}
    elif dataset == 'ucsdped2':
        args.data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/'
        args.vid_path = {'train': [os.path.join(args.data_dir, 'Train/frames/'), 2, 3, 3],
                        'test':  [os.path.join(args.data_dir, 'Test/frames/'), 2, 3, 3]}    # scene_len, clip_len, img_len
        # args.pose_path = {'train': os.path.join(args.data_dir, 'Train/pose/json_results/'),
        #                 'test':  os.path.join(args.data_dir, 'Test/pose/json_results/')}
        args.pose_path = {'train': os.path.join(args.data_dir, 'Train/pose/json_results/'),
                        'test':  os.path.join(args.data_dir, 'Test/pose_refined/json_results/')}
    elif dataset == 'corridor':
        args.data_dir = '/home/yaboliu/data/cvae/corridor'
        args.vid_path = {'train': [os.path.join(args.data_dir, 'training/frames/'), 2, 6, 7],
                        'test':  [os.path.join(args.data_dir, 'testing/frames/'), 2, 6, 7]}    # scene_len, clip_len, img_len
        # args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose/json_results/'),
        #                 'test':  os.path.join(args.data_dir, 'testing/pose/json_results/')}
        args.pose_path = {'train': os.path.join(args.data_dir, 'training/pose_hrnet48/json_results/'),
                        'test':  os.path.join(args.data_dir, 'testing/pose_hrnet48/json_results/')}

    # print(args.pose_path['test'])
    dataset_sence_name = '_'.join(args.dataset_sence)
    args.exp_dir = os.path.join(args.exp_dir, args.dataset, dataset_sence_name, args.level)
    print(args.exp_dir)
    args.ckpt_dir = create_exp_dirs(args.exp_dir)
    args.log_path = os.path.join(args.ckpt_dir, 'train.txt')
    args.optimizer = args.ae_optimizer
    
    ae_args = args_rm_prefix(args, 'ae_')
    dcec_args = args_rm_prefix(args, 'dcec_')
    dcec_args.arch = args.arch
    dcec_args.pred_next_frame = args.pred_next_frame
    dcec_args.level = args.level
    dcec_args.log_path = args.log_path
    
    ae_args.arch = args.arch
    ae_args.log_path = args.log_path
    ae_args.pred_next_frame = args.pred_next_frame
    ae_args.input_frames = args.seg_len
    ae_args.seg_len = args.seg_len
    ae_args.level = args.level
    ae_args.num_workers = args.num_workers
    ae_args.new_preprocess = args.new_preprocess
    ae_args.add_center = args.add_center
    ae_args.high_low_eval = args.high_low_eval
    # ae_args.new_preprocess = args.new_preprocess
    ae_args.ckpt_dir = args.ckpt_dir
    ae_args.vid_path = args.vid_path
    res_args = args_rm_prefix(args, 'res_')
    return args, ae_args, dcec_args, res_args


def init_stc_parser():
    parser = argparse.ArgumentParser("Pose_AD_Experiment")
    # General Args
    # # default_data_dir='/home/yaboliu/work/research/gepc/gepc/data/pose'
    # default_data_dir='/home/yaboliu/data/cvae/shanghaitech/'
    # default_data_dir='/home/yaboliu/data/cvae/avenue/'
    default_data_dir='/home/yaboliu/data/cvae/avenue/'
    default_exp_dir='work_dir_ablation_zhengshi/ablation/'
    
    
    parser.add_argument('--arch', type=str, default='gtae_pred')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--new_preprocess', type=bool, default=False)
    parser.add_argument('--only_normal', type=bool, default=True)
    parser.add_argument('--filter_bbox', type=int, default=-1)    # 2200
    parser.add_argument('--filter_border', type=int, default=-1) # 5
    parser.add_argument('--filter_independent', type=bool, default=False)
    parser.add_argument('--filter_cover', type=bool, default=False)
    parser.add_argument('--use_bbox_high_level', type=bool, default=False)
    
    parser.add_argument('--num_s_transformer', type=int, default=6) # 5
    parser.add_argument('--num_t_transformer', type=int, default=6) # 5
    
    parser.add_argument('--add_center', type=bool, default=False)   #add scale
    parser.add_argument('--add_scale', type=bool, default=False)   #add scale
    parser.add_argument('--ae_new_pipeline', type=bool, default=True)
    parser.add_argument('--headless', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='HR_shanghai')
    parser.add_argument('--dataset_sence', nargs='+', type=str)
    parser.add_argument('--train_dataset_sence', nargs='+', type=str)
    parser.add_argument('--vid_res', nargs='+', type=int, default=[856, 480])
    
    parser.add_argument('--level', type=str, default='high')
    parser.add_argument('--high_filter_continuous', type=bool, default=True)
    
    parser.add_argument('--seg_len', type=int, default=7, metavar='SGLEN',
                        help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--pred_next_frame', type=int, default=4)
    parser.add_argument('--n_clusters', '-k', type=int, default=10,  metavar='K',
                        help='number of clusters (default: 10)')
    
    parser.add_argument('--high_low_eval', type=bool, default=False)
    
    parser.add_argument('--ae_epochs', '-ae_e', type=int, default=40,  metavar='E',
                        help='Number of epochs per cycle. (default: 10)')
    parser.add_argument('--ae_batch_size', '-ae_b', type=int, default=512, metavar='B',
                        help='Batch sizes for autoencoder. (default: 512)') # 1536
    parser.add_argument('--ae_lr', type=float, default=1e-4, metavar='LR',
                        help='Optimizer Learning Rate Parameter (default: 1e-4)')
    parser.add_argument('--dcec_batch_size', '-dcec_b', type=int, default=512, metavar='B',
                        help='Batch size (default: 512)')
    
    
    
    parser.add_argument('--debug', action='store_true',
                        help='Debug experiment script with minimal epochs. (default: False)')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV',
                        help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999,
                        help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1],
                        help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--num_transform', type=int, default=0, metavar='T',
                        help='number of transformations to use for augmentation (default: 5)')
    # parser.add_argument('--headless', action='store_true',
    #                     help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--dropout', type=float, default=0., metavar='DROPOUT',
                        help='Dropout training Parameter (default: 0.3)')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1],
                        help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=0, metavar='PNS', choices=[0, 1],
                        help='Scale keeping proportions [1/0] (default: 0)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR',
                        help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                        help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=8, metavar='W',
                        help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH',
                        help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--act', type=str, default='relu', metavar='ACT_TYPE', 
                        help="Activation used in ST-GCN [relu, mish] (default: 'relu')")
    parser.add_argument('--conv_oper', type=str, default='sagc', metavar='CONV_OPER',
                        help="Convolutional Operator to be used [sagc, gcn,tt] (default: 'sagc')")
    parser.add_argument('--tcp_oper', type=str, default='conv', metavar='CONV_OPER',
                        help="Convolutional Operator to be used [sagc, gcn] (default: 'sagc')")
    
    parser.add_argument('--seg_stride', type=int, default=1, metavar='SGST',
                        help='Stride for training segment sliding window (default: 8)')
    parser.add_argument('--patch_features', action='store_true',
                        help='Use patch features instead of coordinates. (default: False)')
    parser.add_argument('--patch_db', action='store_true',
                        help='Use pre-extracted patch db files. (default: False)')
    parser.add_argument('--patch_size', '-ps', type=int, default=16, metavar='PS',
                        help='Patch size for training (default: 16)')
    # AE Args
    parser.add_argument('--ae_fn', type=str, metavar='AE_FN',
                        help="Path to a trained AE models to start with")
    parser.add_argument('--ae_optimizer', '-ae_o', type=str, default='adam', metavar='AE_OPT',
                        help="Optimizer (default: 'adam')")
    parser.add_argument('--ae_sched', '-ae_s', type=str, default='tri', metavar='AE_SCH',
                        help="Optimization LR scheduler (default: 'tri')")
    
    parser.add_argument('--ae_weight_decay', '-ae_wd', type=float, default=1e-5, metavar='WD',
                        help='Optimizer Weight Decay Parameter (default: 1e-5)')
    parser.add_argument('--ae_lr_decay', '-ae_ld', type=float, default=0.99, metavar='LD',
                        help='Optimizer Learning Rate Decay Parameter (default: 0.99)')
    parser.add_argument('--ae_test_every', type=int, default=20, metavar='T',
                        help='How many epochs between test evaluations (default: 20)')
    
    # K-means init Args
    parser.add_argument('--k_init_downsample', '-ikds', type=int, default=1,  metavar='K_DS',
                        help='Downsample factor for K-means init data (default: 1)')
    parser.add_argument('--k_init_batch', '-ikbs', type=int, default=4,  metavar='K_BS',
                        help='Batch size for K-means init data (default: 4)')
    # DEC Args
    parser.add_argument('--dcec_fn', type=str, metavar='DCEC_FN',
                        help="Path to a trained DCEC models to start with")
    
    parser.add_argument('--gamma', '-g', type=float, default=1,  metavar='G',
                        help='Gamma values for weighting clustering loss (default: 0.6)')
    parser.add_argument('--alpha', '-a', type=float, default=1e-3,  metavar='G',
                        help='Alpha value for weighting L2 regularization (default: 1e-3)')
    parser.add_argument('--dcec_epochs', '-dcec_e', type=int, default=60, metavar='N',
                        help='number of epochs to train per cycle (default: 25)')
    parser.add_argument('--pretrain_epochs', type=int, default=0, metavar='T',
                        help='Number of epochs for which the AE is trained without clustering (gamma=0) \
                        Unused by default. (default: 0)')
    
    parser.add_argument('--dcec_lr', type=float, default=8e-4, metavar='L',
                        help='Learning decay factor per epoch 8e-4')
    parser.add_argument('--dcec_lr_decay', type=float, default=0.98, metavar='DC_LRD',
                        help='Learning rate decay factor per epoch, supports arrays (default: 0.99')
    parser.add_argument('--dcec_optimizer', '-dcec_o', type=str, default='adam', metavar='DEC_OPT',
                        help="Optimizer (default: 'adam')")
    parser.add_argument('--dcec_sched', '-dcec_s', type=str, default='tri', metavar='DEC_SCH',
                        help="Optimization LR scheduler (default: 'tri')")
    parser.add_argument('--dcec_weight_decay', '-dcec_wd', type=float, default=1e-5,  metavar='WD',
                        help='DCEC optimizer Weight Decay Parameter (default: 1e-5)')
    parser.add_argument('--update_interval', type=float, default=2.0, metavar='I',
                        help='Update interval for target distribution P. Float, for fractional update (default: 2.0)')

    # Scoring
    parser.add_argument('--save_results', type=int, default=1, metavar='SR', choices=[0, 1],
                        help='Save results to npz (default: 1)')
    parser.add_argument('--res_batch_size', '-res_b', type=int, default=64,  metavar='B',
                        help='Batch size for scoring. (default: 256)')
    parser.add_argument('--dpmm_fn', type=str, metavar='DPMM_FN',
                        help="Path to a fitted DPMM model")
    return parser


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            ae_key = key[len(prefix):]
            wp_args_dict[ae_key] = value

    return wp_args


def create_exp_dirs(experiment_dir):
    time_str = time.strftime("%b%d_%H%M")

    dirmap = 'stc'

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints/')
    dirs = [checkpoints_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return checkpoints_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)
