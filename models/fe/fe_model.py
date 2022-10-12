from models.gcae.gcae import GCAE
from models.gcae.gtae import GTAE
from models.gcae.gtae_pred import GTAEPred, GTAEHighPred
from models.fe.patchmodel import PatchModel
from models.fe.patch_resnet import pt_resnet


def init_fenet(args, backbone='resnet', rm_linear=True, split_seqs=True, level=None, **kwargs):
    """
    Initialize the whole feature extraction models, including patch feature extractor (if used) and graph AE
    :param args:
    :param backbone:
    :param rm_linear:
    :param split_seqs:
    :param kwargs:
    :return:
    """
    patch_fe = pt_resnet(backbone=backbone, rm_linear=rm_linear)
    # in_channels = getattr(patch_fe, 'outdim', 3)
    in_channels = getattr(args, 'in_channels', 3)
    arch = getattr(args, 'arch', 'gcae')
    headless = getattr(args, 'headless', False)
    graph_args = None
    if level and (arch in ['gtae_pred']):
        print(level * 10)
        if level == 'high':
            gcae = GTAEHighPred(
                    args,
                    in_channels,
                    graph_args=graph_args,
                    dropout=args.dropout,
                    conv_oper=args.conv_oper,
                    act=args.act,
                    input_frames=args.seg_len,
                    headless=headless,
                    split_seqs=split_seqs,
                    **kwargs)
        elif level == 'low':
            gcae = GTAEPred(
                    args,
                    in_channels,
                    graph_args=graph_args,
                    dropout=args.dropout,
                    conv_oper=args.conv_oper,
                    act=args.act,
                    input_frames=args.seg_len,
                    headless=headless,
                    split_seqs=split_seqs,
                    num_s_transformer=args.num_s_transformer,
                    num_t_transformer=args.num_t_transformer,
                    **kwargs)
    else:
        if arch == 'gtae':
            gcae = GTAE(in_channels,
                    graph_args=graph_args,
                    dropout=args.dropout,
                    conv_oper=args.conv_oper,
                    act=args.act,
                    headless=headless,
                    split_seqs=split_seqs,
                    **kwargs)
        elif arch == 'gcae':
            gcae = GCAE(in_channels,
                    graph_args=graph_args,
                    dropout=args.dropout,
                    conv_oper=args.conv_oper,
                    act=args.act,
                    headless=headless,
                    split_seqs=split_seqs,
                    **kwargs)
            
        elif arch == 'gtae_pred':
            gcae = GTAEPred(
                    args,
                    in_channels,
                    graph_args=graph_args,
                    dropout=args.dropout,
                    conv_oper=args.conv_oper,
                    act=args.act,
                    input_frames=args.seg_len,
                    headless=headless,
                    split_seqs=split_seqs,
                    **kwargs)
    
    return PatchModel(patch_fe, gcae, backbone=backbone, arch=arch)


