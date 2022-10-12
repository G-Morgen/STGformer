import os
import torch
import torch.nn as nn

from models.gcae.gcae import GCAE
from models.gcae.gtae import GTAE
from models.gcae.gtae_pred import GTAEPred
from models.gcae.gtae_pred import GTAEHighPred
from models.fe.patch_resnet import pt_resnet


class PatchModel(nn.Module):
    """
    A Wrapper class for hadling per-patch feature extraction
    """
    def __init__(self, patch_fe, gcae, backbone='resnet', arch='gcae'):
        super().__init__()
        self.backbone = backbone
        self.patch_fe = patch_fe
        self.arch = arch
        self.outdim = getattr(self.patch_fe, 'outdim', 3)
        self.gcae = gcae

    def forward(self, x_input):
        if self.arch == 'gtae_pred':
            x = x_input[..., :-1, :]
            input_feature_graph = x_input[..., -1, :].unsqueeze(2)
            # print('pathmodel forward, ', x.shape)
            z, x_size, x_ref = self.graph_encode(x)
            reco_graph = self.decode(z, x_size, x_ref)
            # output = self.model(input_data)
            # graph_loss = self.loss(output[..., :-1], gt_data[..., :-1])
            # position_loss = self.loss(output[..., -1], gt_data[..., -1])
            # reco_loss = graph_loss + position_loss
        else:
            x = x_input
            input_feature_graph = self.extract_patch_features(x)
            z, x_size = self.graph_encode(input_feature_graph)
            reco_graph = self.decode(z, x_size, None)
        # print('pathmodel:', reco_graph.shape, input_feature_graph.shape)
        return reco_graph, input_feature_graph

    def encode(self, x_input):
        if self.arch == 'gtae_pred':
            x = x_input[..., :-1, :]
            feature_graph = x_input[..., -1, :].unsqueeze(2)
            z, x_size, x_ref = self.graph_encode(x)
            # reco_graph = self.decode(z, x_size, x_ref)
            return z, x_size, x_ref, feature_graph
        else:
            feature_graph = self.extract_patch_features(x_input)
            z, x_size, _ = self.gcae.encode(feature_graph)
            return z, x_size, x_input, feature_graph

    def graph_encode(self, input_feature_graph):
        z, x_size, x_ref = self.gcae.encode(input_feature_graph)
        if self.arch == 'gtae_pred':
            return z, x_size, x_ref
        else:
            return z, x_size

    def decode(self, z, x_size, x_ref):
        if self.arch == 'gtae_pred':
            x = self.gcae.decode(z, x_size, x_ref)
        else:
            x = self.gcae.decode(z, x_size)
            x = x[:, :, -1, :]
            x = x.unsqueeze(2)
        return x

    def extract_patch_features(self, x):
        # Take a [N, C, T, V, W, H] tensor,
        # permute and view as [N*T*V, C_in, W, H]
        # Apply models
        # Return feature shape of [N, C_new, T, V]
        if self.backbone is None:  # Using keypoints only, w/o patches
            return x
        else:
            n, c, t, v, w, h = x.size()
            x_perm = x.permute(0, 2, 3, 1, 4, 5).contiguous()
            x_perm = x_perm.view(n * t * v, c, w, h)
            f_perm = self.patch_fe(x_perm)  # y is [N*T*V, C_new]
            f_perm = f_perm.view(n, t, v, -1)
            feature_graph = f_perm.permute(0, 3, 1, 2).contiguous()
            return feature_graph

    def get_patchmodel_dict(self, epoch, args=None, optimizer=None):
        state = {
            'epoch': epoch + 1,
            'outdim': self.outdim,
            'backbone': self.backbone,
            'patch_model': self.patch_fe.state_dict(),
            'gcae': self.gcae.state_dict(),
            'args': args,
        }
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if hasattr(self.gcae, 'num_class'):
            state['n_classes'] = self.gcae.num_class
        if hasattr(self.gcae, 'h_dim'):
            state['h_dim'] = self.gcae.h_dim
        return state

    def save_checkpoint(self, epoch, args=None, optimizer=None, filename=None):
        state = self.get_patchmodel_dict(epoch, args=args, optimizer=optimizer)
        path_join = os.path.join(args.ckpt_dir, filename)
        torch.save(state, path_join)

    def load_checkpoint(self, path):
        try:
            patchmodel_dict = torch.load(path)
            self.load_patchmodel_dict(patchmodel_dict)
            print("Checkpoint loaded successfully from '{}')\n" .format(path))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'.".format(path))

    def load_patchmodel_dict(self, patchmodel_dict, backbone=None, args=None, load_level=None):
        if args is None:
            args = patchmodel_dict['args']
        self.backbone = patchmodel_dict.get('backbone', backbone)
        self.patch_fe = pt_resnet(backbone=self.backbone)

        fe_state_dict = patchmodel_dict.get('patch_model', patchmodel_dict)
        gcae_state_dict = patchmodel_dict.get('gcae', patchmodel_dict)

        if backbone is not None:
            self.patch_fe.load_state_dict(fe_state_dict)

        in_channels = getattr(self.patch_fe, 'outdim', 3)
        headless = args.headless
        arch  = getattr(args, 'arch', 'gcae')
        # high_low_eval  = getattr(args, 'high_low_eval', None)
        
        if load_level:
            if load_level == 'high':
                self.gcae = GTAEHighPred(
                                    args=args,
                                    in_channels=2,
                                    # graph_args=graph_args,
                                    dropout=args.dropout,
                                    conv_oper=args.conv_oper,
                                    act=args.act,
                                    headless=headless,
                                    # split_seqs=split_seqs,
                                    # **kwargs
                                    )
            elif load_level == 'low':
                self.gcae = GTAEPred(
                                args=args,
                                in_channels=2,
                                # graph_args=graph_args,
                                dropout=args.dropout,
                                conv_oper=args.conv_oper,
                                act=args.act,
                                headless=headless,
                                # split_seqs=split_seqs,
                                # **kwargs
                                )
        else:
            if arch == 'gtae':
                self.gcae = GTAE(in_channels,
                            dropout=args.dropout,
                            conv_oper=args.conv_oper,
                            act=args.act,
                            headless=headless)
            elif arch == 'gcae':
                self.gcae = GCAE(in_channels,
                                dropout=args.dropout,
                                conv_oper=args.conv_oper,
                                act=args.act,
                                headless=headless)
            elif arch == 'gtae_pred':
                self.gcae = GTAEPred(
                                    args=args,
                                    in_channels=2,
                                    # graph_args=graph_args,
                                    dropout=args.dropout,
                                    conv_oper=args.conv_oper,
                                    act=args.act,
                                    headless=headless,
                                    # split_seqs=split_seqs,
                                    # **kwargs
                                    )
        self.gcae.load_state_dict(gcae_state_dict)

