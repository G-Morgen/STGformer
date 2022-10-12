import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.graph.graph import Graph
from models.graph.st_graph_conv_block import ConvBlock
from einops import rearrange, repeat
import copy


class GTAEPred(nn.Module):
    def __init__(self, args, in_channels, ff_hidden_size=64, num_self_att_layers=1, nhead=8, 
                 graph_args=None, split_seqs=True, eiw=True, dropout=0.0, input_frames=6, 
                 conv_oper=None, act=None, headless=False, **kwargs):
        super(GTAEPred, self).__init__()
        # 单个batch内的t=12，默认temporal卷积核大小为9
        # input_frames = seg_len - 1
        in_channels_ls = [in_channels] + [32, 64]
        self.feature_dim_size = in_channels_ls[-1]
        self.ff_hidden_size = ff_hidden_size
        # self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        
        self.nhead = nhead
        self.args = args
        # if conv_oper == 'tt':
        #     temporal_net = TemporalTransformer
        # else:
        #     temporal_net = TCN
            
        temporal_net = TemporalTransformer
        
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        # print('A shape: ', self.graph.A.shape)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # print('new_preprocess>>>>>>>', args.new_preprocess)
        if args.add_center:
            self.num_nodes = A.shape[1] + 1
            A = torch.sum(A, dim=0)
            #添加最后重心关键点的邻接矩阵
            A_temp = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, requires_grad=False)
            A_temp[:self.num_nodes - 1, :self.num_nodes - 1] = A
            A_temp[-1][-1] = 1
            A_temp[-1][:] = 1 / self.num_nodes
            A_temp[:][-1] = 1 / self.num_nodes
            
        else:
            self.num_nodes = A.shape[1]
            A = torch.sum(A, dim=0)
            A_temp = A
            
        # self.num_nodes = A.shape[1]
        # A = torch.sum(A, dim=0)
        # A_temp = A
        
        self.register_buffer('A', A_temp)
        
        print('A shape: ', self.A.shape)
        # print(self.A)
        
        self.headless = headless
        self.add_center = args.add_center
        
        arch_dict = {'enc_stride': [1, 1, 3, 1, 2, 2], #[1, 1, 2, 1, 1, 3, 1, 1, 1],
                    #  'dec_stride': [1, 2, 2, 1, 3, 1]
                    }  # [1, 3, 1, 1, 2, 1]
        self.arch_dict = arch_dict
        
        # temporal_kernelsize = 9
        # temporal_padding = ((temporal_kernelsize - 1) // 2, 0)
        
        self.num_GNN_layers_encoder = len(arch_dict['enc_stride'])
        # self.num_GNN_layers_decoder = len(arch_dict['dec_stride'])


        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.conv1x1_encode = nn.ModuleList()
        for i in range(len(in_channels_ls) - 1):
            self.conv1x1_encode.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_ls[i], in_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(in_channels_ls[i + 1]),
                    # nn.ReLU()
                    )
            )
            
        self.conv1x1_decode = nn.ModuleList()
        in_channels_ls.reverse()
        for i in range(len(in_channels_ls) - 2):
            self.conv1x1_decode.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_ls[i], in_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(in_channels_ls[i + 1]),
                    nn.ReLU() 
                    )
            )
        self.decode_final = nn.Conv2d(in_channels_ls[-2], in_channels_ls[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        
        # encoder
        self.ugformer_encoder = torch.nn.ModuleList()
        self.lst_gnn_encoder = torch.nn.ModuleList()
        self.tcn_encoder = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers_encoder):
            self.ugformer_encoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
                self.num_self_att_layers, self.num_nodes, dropout))
            self.lst_gnn_encoder.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))
            self.tcn_encoder.append(temporal_net(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
                self.num_self_att_layers, input_frames, dropout, (self.arch_dict['enc_stride'][_layer], 1)))

        # # decoder
        # self.ugformer_decoder = torch.nn.ModuleList()
        # self.lst_gnn_decoder = torch.nn.ModuleList()
        # self.tcn_decoder = torch.nn.ModuleList()
        # self.upsample_decoder = torch.nn.ModuleList()
        # for _layer in range(self.num_GNN_layers_decoder):
        #     # if ((arch_dict['dec_stride'][_layer] != 1) and (conv_oper != 'tt')):
        #     #     self.upsample_decoder.append(nn.Upsample(scale_factor=(arch_dict['dec_stride'][_layer], 1), mode='bilinear'))
        #     # else:
        #     #     self.upsample_decoder.append(nn.Identity())
        #     self.ugformer_decoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
        #         self.num_self_att_layers, self.num_nodes, dropout))
        #     self.lst_gnn_decoder.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))
        #     self.tcn_decoder.append(temporal_net(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
        #         self.num_self_att_layers, input_frames, dropout, (1, 1)))
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.feature_dim_size, input_frames, self.num_nodes))
        self.pred_token = nn.Parameter(torch.randn(1, self.feature_dim_size, 1, self.num_nodes))


    def forward(self, x, ret_z=False):
        # print('forward')
        # print(x.shape)
        z, x_size, x_ref = self.encode(x)
        
        x_reco = self.decode(z, x_size, x_ref)
        # print(x_reco.shape)
        if ret_z:
            return x_reco, z
        else:
            return x_reco

        
    def encode(self, x):
        # x = x.unsqueeze(4)
        # N, C, T, V = x.size()
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)
        
        
        
        for layer_idx in range(len(self.conv1x1_encode)):
            x = self.conv1x1_encode[layer_idx](x)
        n, c, t, v = x.shape
        # print('>>>>>> encoder')
        
        # print('x: ', x.shape)
        pred_token = repeat(self.pred_token, '() c t v -> n c t v', n = n)
        # print('pred_token: ', pred_token.shape)
        # print(x.shape, pred_token.shape)
        x = torch.cat((x, pred_token), dim=2)
        # print('x: ', x.shape)
        
        x = x + self.pos_embedding[:]
        
        for layer_idx in range(self.num_GNN_layers_encoder):
            # x = x + copy.deepcopy(self.pos_embedding)[:]
            n, c, t, v = x.shape
            res = x
            # print(0, x.shape)
            x = x.permute(0, 2, 1, 3).contiguous().view(n*t, c, v)
            x = x.permute(2, 0, 1).contiguous()    # v, nt, c
            # print(1, x.shape)
            x = self.ugformer_encoder[layer_idx](x)    # v, nt, c
            # print(2, x.shape)
            x = x.permute(1, 0, 2).contiguous().view(n, t, v, c)
            x = x.permute(0, 3, 1, 2).contiguous()    # n, c, t, v
            # print(3, x.shape)
            x = self.lst_gnn_encoder[layer_idx](x, self.A)   # n, c, t, v
            # print(4, x.shape)
            x = self.tcn_encoder[layer_idx](x)   # n, c, t, v
            # print(5, x.shape)
            # res = F.interpolate(res, size=x.shape[-2:], mode='bilinear', align_corners=False)
            if layer_idx != (self.num_GNN_layers_encoder - 1):
                x = x + res
            
        # x = x.contiguous().unsqueeze(1)
        # x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        
        # x_ref = x[:, :, -1, :].contiguous()
        # x_ref = x_ref.unsqueeze(2).contiguous()
        
        # # x_ref = x
        # x = x_ref.view(n, -1).contiguous()
        # x_size = x.size()
       
        
        x_ref = x[:, :, -1, :].contiguous()
        x_ref = x_ref.unsqueeze(2).contiguous()
        
        # x = x_ref.view(n, -1).contiguous()
        x = x.view(n, -1)
        
        x_size = x.size()
        return x, x_size, x_ref


    def decode(self, z, x_size, x_ref=None):
        x = x_ref
        
        # # Decoding layers
        # x = z.view(x_size)
        # # print(0, x.shape)
        # N, C, T, V, M = x_size
        # x = x.permute(0, 4, 1, 2, 3).contiguous()
        # x = x.view(N * M, C, T, V)
        # # print(0, x.shape)
        # # n, c, t, v = x.shape
        
        # for layer_idx in range(self.num_GNN_layers_decoder):
            
        #     # res = x
        #     # print(1, x.shape)
        #     # x = self.upsample_decoder[layer_idx](x)
        #     n, c, t, v = x.shape
        #     # print(2, x.shape)
        #     x = x.permute(0, 2, 1, 3).contiguous().view(n*t, c, v)
        #     x = x.permute(2, 0, 1).contiguous()    # v, nt, c
        #     # print(3, x.shape)
        #     x = self.ugformer_decoder[layer_idx](x)    # v, nt, c
        #     # print(4, x.shape)
        #     x = x.permute(1, 0, 2).view(n, t, v, c).contiguous()
        #     x = x.permute(0, 3, 1, 2).contiguous()    # n, c, t, v
        #     # print(5, x.shape)
        #     x = self.lst_gnn_decoder[layer_idx](x, torch.sum(self.A, dim=0))
        #     # print(6, x.shape)
        #     x = self.tcn_decoder[layer_idx](x)
        #     # print(7, x.shape)
            
        #     # if (layer_idx + 1) != self.num_GNN_layers_decoder:
        #     #     res = F.interpolate(res, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     #     x = x + res
            
        # implicit_encoding = x[:, :, -1, :]
                
        for layer_idx in range(len(self.conv1x1_decode)):
            x = self.conv1x1_decode[layer_idx](x)
        x = self.decode_final(x)
        
        # if self.args.new_preprocess:
        #     x = self.sigmoid(x)
        # else:
        #     x = self.tanh(x)
            
        # x = self.sigmoid(x)   # n, c, t, v
        
        # NM, c, t, v = x.size()
        # pred = x[:, :, -1, :]
        # x = x.view(N, M, c, t, v)
        # x = x.permute(0, 2, 3, 4, 1).contiguous()
        # print(x.shape)
        return x 
    
    

class GTAEHighPred0(nn.Module):
    def __init__(self, args, in_channels, ff_hidden_size=64, num_self_att_layers=1, nhead=8, 
                 graph_args=None, split_seqs=True, eiw=True, dropout=0.0, input_frames=6, 
                 conv_oper=None, act=None, headless=False, **kwargs):
        super(GTAEHighPred, self).__init__()
        # 单个batch内的t=12，默认temporal卷积核大小为9
        # input_frames = seg_len - 1
        in_channels_ls = [in_channels] + [32, 64]
        self.feature_dim_size = in_channels_ls[-1]
        self.ff_hidden_size = ff_hidden_size
        # self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        
        self.nhead = nhead
        self.args = args
        # if conv_oper == 'tt':
        #     temporal_net = TemporalTransformer
        # else:
        #     temporal_net = TCN
            
        temporal_net = TemporalTransformer
        
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        # print('A shape: ', self.graph.A.shape)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # print('new_preprocess>>>>>>>', args.new_preprocess)
        if args.add_center:
            self.num_nodes = A.shape[1] + 1
            A = torch.sum(A, dim=0)
            #添加最后重心关键点的邻接矩阵
            A_temp = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, requires_grad=False)
            A_temp[:self.num_nodes - 1, :self.num_nodes - 1] = A
            A_temp[-1][-1] = 1
            A_temp[-1][:] = 1 / self.num_nodes
            A_temp[:][-1] = 1 / self.num_nodes
            
        else:
            self.num_nodes = A.shape[1]
            A = torch.sum(A, dim=0)
            A_temp = A
            
        self.num_nodes = 1
        # self.num_nodes = A.shape[1]
        # A = torch.sum(A, dim=0)
        # A_temp = A
        
        self.register_buffer('A', A_temp)
        
        print('A shape: ', self.A.shape)
        # print(self.A)
        
        self.headless = headless
        self.add_center = args.add_center
        
        arch_dict = {'enc_stride': [1, 1, 3, 1, 2, 2], #[1, 1, 2, 1, 1, 3, 1, 1, 1],
                    #  'dec_stride': [1, 2, 2, 1, 3, 1]
                    }  # [1, 3, 1, 1, 2, 1]
        self.arch_dict = arch_dict
        
        # temporal_kernelsize = 9
        # temporal_padding = ((temporal_kernelsize - 1) // 2, 0)
        
        self.num_GNN_layers_encoder = len(arch_dict['enc_stride'])
        # self.num_GNN_layers_decoder = len(arch_dict['dec_stride'])


        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.conv1x1_encode = nn.ModuleList()
        for i in range(len(in_channels_ls) - 1):
            self.conv1x1_encode.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_ls[i], in_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(in_channels_ls[i + 1]),
                    # nn.ReLU()
                    )
            )
            
        self.conv1x1_decode = nn.ModuleList()
        in_channels_ls.reverse()
        for i in range(len(in_channels_ls) - 2):
            self.conv1x1_decode.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_ls[i], in_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(in_channels_ls[i + 1]),
                    nn.ReLU() 
                    )
            )
        self.decode_final = nn.Conv2d(in_channels_ls[-2], in_channels_ls[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, input_frames))
        self.max_pool = nn.AdaptiveMaxPool2d((1, input_frames))
        
        
        # encoder
        self.ugformer_encoder = torch.nn.ModuleList()
        # self.lst_gnn_encoder = torch.nn.ModuleList()
        self.tcn_encoder = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers_encoder):
            self.ugformer_encoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
                self.num_self_att_layers, self.num_nodes, dropout))
            # self.lst_gnn_encoder.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))
            # self.tcn_encoder.append(temporal_net(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
            #     self.num_self_att_layers, input_frames, dropout, (self.arch_dict['enc_stride'][_layer], 1)))
            self.tcn_encoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
                self.num_self_att_layers, self.num_nodes, dropout))

        # # decoder
        # self.ugformer_decoder = torch.nn.ModuleList()
        # self.lst_gnn_decoder = torch.nn.ModuleList()
        # self.tcn_decoder = torch.nn.ModuleList()
        # self.upsample_decoder = torch.nn.ModuleList()
        # for _layer in range(self.num_GNN_layers_decoder):
        #     # if ((arch_dict['dec_stride'][_layer] != 1) and (conv_oper != 'tt')):
        #     #     self.upsample_decoder.append(nn.Upsample(scale_factor=(arch_dict['dec_stride'][_layer], 1), mode='bilinear'))
        #     # else:
        #     #     self.upsample_decoder.append(nn.Identity())
        #     self.ugformer_decoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
        #         self.num_self_att_layers, self.num_nodes, dropout))
        #     self.lst_gnn_decoder.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))
        #     self.tcn_decoder.append(temporal_net(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
        #         self.num_self_att_layers, input_frames, dropout, (1, 1)))
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.feature_dim_size, 100, input_frames, self.num_nodes))
        self.pred_token = nn.Parameter(torch.randn(1, self.feature_dim_size, 100, 1, self.num_nodes))


    def forward(self, x, ret_z=False):
        # print('forward')
        # print(x.shape)
        z, x_size, x_ref = self.encode(x)
        
        x_reco = self.decode(z, x_size, x_ref)
        # print(x_reco.shape)
        if ret_z:
            return x_reco, z
        else:
            return x_reco

        
    def encode(self, x):
        # x = x.unsqueeze(4)
        # N, C, T, V = x.size()
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)
        
        # print('pred_token: ', self.pred_token.shape)
        # print('pred_token: ', self.pos_embedding.shape)
        # print('x.shape: ', x.shape)
        
        # 1 c p t 1
        # print(x.shape)
        n, c, p, t, v = x.shape
        x = x.squeeze(-1)
        for layer_idx in range(len(self.conv1x1_encode)):
            x = self.conv1x1_encode[layer_idx](x)
        x = x.unsqueeze(-1) # 1 c p t 1
        
        # print('>>>>>> encoder')
        
        # print('x: ', x.shape)
        # pred_token = repeat(self.pred_token, '() c t v -> n c t v', n = n)
        
        x = torch.cat((x, self.pred_token[:, :, :p, :, :]), dim=3)
        # print('x: ', x.shape)
        
        x = x + self.pos_embedding[:, :, :p, :, :]
        
        for layer_idx in range(self.num_GNN_layers_encoder):
            # x = x + copy.deepcopy(self.pos_embedding)[:]
            n, c, p, t, v = x.shape
            x = x.squeeze(0).squeeze(-1)   # c p t
            res = x
            # print(0, x.shape)
            # x = x.permute(2, 0, 1).contiguous().view(n*t, c, v)
            x = x.permute(1, 2, 0).contiguous()    # p, t, c
            # print(1, x.shape)
            x = self.ugformer_encoder[layer_idx](x)    # p, t, c
            # print(2, x.shape)
            # x = x.permute(1, 0, 2).contiguous().view(n, t, v, c)
            x = x.permute(1, 0, 2).contiguous()    # t, p, c
            # print(3, x.shape)
            # x = self.lst_gnn_encoder[layer_idx](x, self.A)   # n, c, t, v
            # print(4, x.shape)
            x = self.tcn_encoder[layer_idx](x)   # t, p, c
            x = x.permute(2, 1, 0).contiguous()   # c, p, t
            # print(5, x.shape)
            # res = F.interpolate(res, size=x.shape[-2:], mode='bilinear', align_corners=False)
            if layer_idx != (self.num_GNN_layers_encoder - 1):
                x = x + res
            x = x.unsqueeze(0).unsqueeze(-1)    # n, c, p, t, v
            
        # x = x.contiguous().unsqueeze(1)
        # x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        
        # x = x.unsqueeze(0).unsqueeze(-1)    # n, c, p, t, v
        x_ref = x[:, :, :, -1, :].contiguous()
        x_ref = x_ref.unsqueeze(3).contiguous()
        # print(x_ref.shape)
        
        # x_ref = x
        # x = x_ref.view(n, -1).contiguous()
        x = x.squeeze(-1)
        # x = self.avg_pool(x)
        # print('>' * 10)
        # print(x.shape)
        x = self.max_pool(x)
        # print(x.shape)
        x = x.view(n, -1)
        
        x_size = x.size()
        # print(x.shape, x_ref.shape, x_size)
        return x, x_size, x_ref


    def decode(self, z, x_size, x_ref=None):
        x = x_ref   # n, c, p, t, v
        
        # # Decoding layers
        # x = z.view(x_size)
        # # print(0, x.shape)
        # N, C, T, V, M = x_size
        # x = x.permute(0, 4, 1, 2, 3).contiguous()
        # x = x.view(N * M, C, T, V)
        # # print(0, x.shape)
        # # n, c, t, v = x.shape
        
        # for layer_idx in range(self.num_GNN_layers_decoder):
            
        #     # res = x
        #     # print(1, x.shape)
        #     # x = self.upsample_decoder[layer_idx](x)
        #     n, c, t, v = x.shape
        #     # print(2, x.shape)
        #     x = x.permute(0, 2, 1, 3).contiguous().view(n*t, c, v)
        #     x = x.permute(2, 0, 1).contiguous()    # v, nt, c
        #     # print(3, x.shape)
        #     x = self.ugformer_decoder[layer_idx](x)    # v, nt, c
        #     # print(4, x.shape)
        #     x = x.permute(1, 0, 2).view(n, t, v, c).contiguous()
        #     x = x.permute(0, 3, 1, 2).contiguous()    # n, c, t, v
        #     # print(5, x.shape)
        #     x = self.lst_gnn_decoder[layer_idx](x, torch.sum(self.A, dim=0))
        #     # print(6, x.shape)
        #     x = self.tcn_decoder[layer_idx](x)
        #     # print(7, x.shape)
            
        #     # if (layer_idx + 1) != self.num_GNN_layers_decoder:
        #     #     res = F.interpolate(res, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     #     x = x + res
            
        # implicit_encoding = x[:, :, -1, :]
                
        x = x.squeeze(-1)   # n, c, p, t
        for layer_idx in range(len(self.conv1x1_decode)):
            x = self.conv1x1_decode[layer_idx](x)
        x = self.decode_final(x)
        
        # if self.args.new_preprocess:
        #     x = self.sigmoid(x)
        # else:
        #     x = self.tanh(x)
            
        x = self.sigmoid(x)
        
        # NM, c, t, v = x.size()
        # pred = x[:, :, -1, :]
        # x = x.view(N, M, c, t, v)
        # x = x.permute(0, 2, 3, 4, 1).contiguous()
        # print(x.shape)
        x = x.unsqueeze(-1)
        return x 
    
class GTAEHighPred1(nn.Module):
    def __init__(self, args, in_channels, ff_hidden_size=64, num_self_att_layers=1, nhead=8, 
                 graph_args=None, split_seqs=True, eiw=True, dropout=0.0, input_frames=6, 
                 conv_oper=None, act=None, headless=False, **kwargs):
        super(GTAEHighPred, self).__init__()
        # 单个batch内的t=12，默认temporal卷积核大小为9
        # input_frames = seg_len - 1
        enc_channels_ls = [input_frames - 1] + [32, 64, 128, 256, 512]
        dec_channels_ls = [512, 256, 128, 64, 32] + [1]
        
        self.conv1x1_encode = nn.ModuleList()
        for i in range(len(enc_channels_ls) - 1):
            self.conv1x1_encode.append(
                nn.Sequential(
                    nn.Conv2d(enc_channels_ls[i], enc_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    nn.ReLU(),
                    # nn.Conv2d(enc_channels_ls[i + 1], enc_channels_ls[i + 1], 3, padding=1),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    # nn.ReLU(),
                    )
            )
            
        self.conv1x1_decode = nn.ModuleList()
        for i in range(len(dec_channels_ls) - 2):
            self.conv1x1_decode.append(
                nn.Sequential(
                    nn.Conv2d(dec_channels_ls[i], dec_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(dec_channels_ls[i + 1]),
                    nn.ReLU(),
                    # nn.Conv2d(dec_channels_ls[i + 1], dec_channels_ls[i + 1], 3, padding=1),
                    # nn.BatchNorm2d(dec_channels_ls[i + 1]),
                    # nn.ReLU(),
                    )
            )
        self.decode_final = nn.Conv2d(dec_channels_ls[-2], dec_channels_ls[-1], 1)
        self.fc_final = nn.Linear(dec_channels_ls[-2], 2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.max_pool = nn.AdaptiveMaxPool2d((1, in_channels))
        
    def forward(self, x, ret_z=False):
        # print('forward')
        # print(x.shape)
        z, x_size, x_ref = self.encode(x)
        
        x_reco = self.decode(z, x_size, x_ref)
        # print(x_reco.shape)
        if ret_z:
            return x_reco, z
        else:
            return x_reco

        
    def encode(self, x):
        n, c, p, t, v = x.shape
        x = x.squeeze(-1)
        x = x.permute(0, 3, 2, 1).contiguous()  # n t p c
        for layer_idx in range(len(self.conv1x1_encode)):
            # res = x
            x = self.conv1x1_encode[layer_idx](x)  # n t p c
            # x = x + res
            
        x_ref = x
        x = self.max_pool(x)
        x = x.view(n, -1)
        
        x_size = x.size()
        return x, x_size, x_ref

    def decode(self, z, x_size, x_ref=None):
        x = x_ref  # n c t p
        # print('>>>>>')
        # print(x.shape)
        for layer_idx in range(len(self.conv1x1_decode)):
            # res = x
            x = self.conv1x1_decode[layer_idx](x)
            # print(x.shape)
            # x = x + res
        # print(x.shape)
        # x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape)
        # n, t, p, c = x.shape
        # x = x.view(n*t*p, c)
        x = self.decode_final(x)
        # print(x.shape)
        # x = self.fc_final(x)
        # print(x.shape)
        # x = x.view(n, t, p, 2)
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.sigmoid(x)  # n t p c
        
        # NM, c, t, v = x.size()
        # pred = x[:, :, -1, :]
        # x = x.view(N, M, c, t, v)
        x = x.permute(0, 3, 2, 1).contiguous()  # n c p t
        x = x.unsqueeze(-1)
        # print(x.shape)
        return x 
    
 
class GTAEHighPred(nn.Module):
    def __init__(self, args, in_channels, ff_hidden_size=64, num_self_att_layers=1, nhead=8, 
                 graph_args=None, split_seqs=True, eiw=True, dropout=0.0, input_frames=6, 
                 conv_oper=None, act=None, headless=False, **kwargs):
        super(GTAEHighPred, self).__init__()
        # 单个batch内的t=12，默认temporal卷积核大小为9
        # input_frames = seg_len - 1
        enc_channels_ls = [input_frames - 1] + [32, 64, 128, 256, 512]
        dec_channels_ls = [512, 256, 128, 64, 32] + [1]
        
        self.conv1x1_encode = nn.ModuleList()
        for i in range(len(enc_channels_ls) - 1):
            self.conv1x1_encode.append(
                nn.Sequential(
                    nn.Conv2d(enc_channels_ls[i], enc_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    nn.ReLU(),
                    # nn.Conv2d(enc_channels_ls[i + 1], enc_channels_ls[i + 1], 3, padding=1),
                    # nn.BatchNorm2d(enc_channels_ls[i + 1]),
                    # nn.ReLU(),
                    )
            )
            
        self.conv1x1_decode = nn.ModuleList()
        for i in range(len(dec_channels_ls) - 2):
            self.conv1x1_decode.append(
                nn.Sequential(
                    nn.Conv2d(dec_channels_ls[i], dec_channels_ls[i + 1], 1),
                    # nn.BatchNorm2d(dec_channels_ls[i + 1]),
                    nn.ReLU(),
                    # nn.Conv2d(dec_channels_ls[i + 1], dec_channels_ls[i + 1], 3, padding=1),
                    # nn.BatchNorm2d(dec_channels_ls[i + 1]),
                    # nn.ReLU(),
                    )
            )
        self.decode_final = nn.Conv2d(dec_channels_ls[-2], dec_channels_ls[-1], 1)
        self.fc_final = nn.Linear(dec_channels_ls[-2], 2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.max_pool = nn.AdaptiveMaxPool2d((1, in_channels))
        
    def forward(self, x, ret_z=False):
        # print('forward')
        # print(x.shape)
        z, x_size, x_ref = self.encode(x)
        
        x_reco = self.decode(z, x_size, x_ref)
        # print(x_reco.shape)
        if ret_z:
            return x_reco, z
        else:
            return x_reco

        
    def encode(self, x):
        n, c, p, t, v = x.shape
        x = x.squeeze(-1)
        x = x.permute(0, 3, 2, 1).contiguous()  # n t p c
        for layer_idx in range(len(self.conv1x1_encode)):
            # res = x
            x = self.conv1x1_encode[layer_idx](x)  # n t p c
            # x = x + res
            
        x_ref = x
        x = self.max_pool(x)
        x = x.view(n, -1)
        
        x_size = x.size()
        return x, x_size, x_ref

    def decode(self, z, x_size, x_ref=None):
        x = x_ref  # n c t p
        # print('>>>>>')
        # print(x.shape)
        for layer_idx in range(len(self.conv1x1_decode)):
            # res = x
            x = self.conv1x1_decode[layer_idx](x)
            # print(x.shape)
            # x = x + res
        # print(x.shape)
        # x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape)
        # n, t, p, c = x.shape
        # x = x.view(n*t*p, c)
        x = self.decode_final(x)
        # print(x.shape)
        # x = self.fc_final(x)
        # print(x.shape)
        # x = x.view(n, t, p, 2)
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.sigmoid(x)  # n t p c
        
        # NM, c, t, v = x.size()
        # pred = x[:, :, -1, :]
        # x = x.view(N, M, c, t, v)
        x = x.permute(0, 3, 2, 1).contiguous()  # n c p t
        x = x.unsqueeze(-1)
        # print(x.shape)
        return x 

class TCN(nn.Module):
    # def __init__(self, feature_dim_size, feature_dim_size, temporal_kernelsize, dropout):
    def __init__(self, feature_dim_size, nhead, ff_hidden_size, num_self_att_layers, \
                input_frames, dropout, stride):
        super(TCN, self).__init__()
        temporal_kernelsize = 9
        temporal_padding = ((temporal_kernelsize - 1) // 2, 0)
        self.net = nn.Sequential(
                    nn.Conv2d(feature_dim_size, feature_dim_size,
                                (temporal_kernelsize, 1), stride, temporal_padding),
                    nn.BatchNorm2d(feature_dim_size),
                    nn.ReLU(),
                    nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
    
class CommonTransformer(nn.Module):
    def __init__(self, feature_dim_size, nhead, ff_hidden_size, num_self_att_layers, num_nodes, dropout):
        super(CommonTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim_size, nhead=nhead, \
                                                dim_feedforward=ff_hidden_size, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_self_att_layers)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, feature_dim_size))
    def forward(self, x):   # v n c
        x = x.permute(1, 0, 2).contiguous() # n v c
        # print(x.shape, self.pos_embedding.shape)
        # x = x + self.pos_embedding[:] # n v c
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # v n c
        return self.transformer(x)
    
    
class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim_size, nhead, ff_hidden_size, num_self_att_layers, num_nodes, dropout, stride):
        super(TemporalTransformer, self).__init__()
        # # self.average = nn.AvgPool2d(feature_dim_size, feature_dim_size, (1, num_nodes))
        # encoder_layers = TransformerEncoderLayer(d_model=feature_dim_size, nhead=nhead, \
        #                                         dim_feedforward=ff_hidden_size, dropout=dropout)
        # self.transformer = TransformerEncoder(encoder_layers, num_self_att_layers)
        self.transformer = CommonTransformer(feature_dim_size, nhead, ff_hidden_size, num_self_att_layers, num_nodes, dropout)
    def forward(self, x):
        # input (n, c, t, v)
        # x = self.average(x) # n, c, t, 1
        n, c, t, v = x.shape
        x = x.permute(2, 0, 3, 1).contiguous().view(t, n*v, c)  # t nv c
        # x = x.squeeze(-1)   # t n c
        x = self.transformer(x)   # t nv c
        x = x.view(t, n, v, c)
        x = x.permute(1, 3, 0, 2).contiguous()
        return x

""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # input (n, c, t, v) adj (v, v)
        # support = torch.mm(input, self.weight)  # [b, num_nodes, out_features]
        
        # input = torch.einsum('ntvc,vo->ntvo', (input, self.weight))
        # input = input.permute(0, 3, 1, 2)
        # support = torch.bmm(input, self.weight) #(n*t, v, out_features)
        # support = support.permute(0, 2, 1)
        # output = torch.spmm(adj, support)   # [num_node, out_features]
        output = torch.einsum('nctv,vw->nctw', (input, adj))    # nctv
        if self.bias is not None:
            output = output + self.bias
        n, c, t, v = output.shape
        output = output.permute(0, 2, 1, 3).contiguous().view(n*t, c, v)
        output = self.bn(output)
        output = self.act(output)
        output = output.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        output = output + input
        return output
    
    def forward0(self, input, adj):
        return input
