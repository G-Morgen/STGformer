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



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.BatchNorm1d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class GTAEPred(nn.Module):
    def __init__(self, args, in_channels, ff_hidden_size=64, num_self_att_layers=1, nhead=16, 
                 graph_args=None, split_seqs=True, eiw=True, dim_head=64, dropout=0.1, input_frames=6, 
                 conv_oper=None, act=None, headless=False, **kwargs):
        super(GTAEPred, self).__init__()
        # 单个batch内的t=12，默认temporal卷积核大小为9
        # input_frames = seg_len - 1
        # in_channels_ls = [in_channels] + [32, 64]
        dim = 1024
        mlp_dim = 2048
        self.num_GNN_layers_encoder = 6
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

        self.register_buffer('A', A_temp)
        print('A shape: ', self.A.shape)
        # print(self.A)
        
        self.headless = headless
        self.add_center = args.add_center
        
        # arch_dict = {'enc_stride': [1, 1, 3, 1, 2, 2], #[1, 1, 2, 1, 1, 3, 1, 1, 1],
        #             #  'dec_stride': [1, 2, 2, 1, 3, 1]
        #             }  # [1, 3, 1, 1, 2, 1]
        # self.arch_dict = arch_dict
        
        # temporal_kernelsize = 9
        # temporal_padding = ((temporal_kernelsize - 1) // 2, 0)
        
        # self.num_GNN_layers_encoder = len(arch_dict['enc_stride'])
        # self.num_GNN_layers_decoder = len(arch_dict['dec_stride'])


        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        # self.conv1x1_encode = nn.ModuleList()
        # for i in range(len(in_channels_ls) - 1):
        #     self.conv1x1_encode.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels_ls[i], in_channels_ls[i + 1], 1),
        #             # nn.BatchNorm2d(in_channels_ls[i + 1]),
        #             # nn.ReLU()
        #             )
        #     )
            

        

            
        self.conv1x1_decode = nn.ModuleList()
        in_channels_ls = [dim, 512, 64, 2]
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
        # self.tanh = nn.Tanh()
        
        
        # encoder
        self.ugformer_encoder = torch.nn.ModuleList()
        self.lst_gnn_encoder = torch.nn.ModuleList()
        self.tcn_encoder = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers_encoder):
            self.ugformer_encoder.append(Transformer(dim, 1, self.nhead, dim_head, mlp_dim, dropout=dropout))
            self.lst_gnn_encoder.append(GraphConvolution(dim, dim, act=torch.relu))
            self.tcn_encoder.append(Transformer(dim, 1, self.nhead, dim_head, mlp_dim, dropout=dropout))
            
            
            # self.ugformer_encoder.append(CommonTransformer(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
            #     self.num_self_att_layers, self.num_nodes, dropout))
            # self.tcn_encoder.append(temporal_net(self.feature_dim_size, self.nhead, self.ff_hidden_size, \
            #     self.num_self_att_layers, input_frames, dropout, (self.arch_dict['enc_stride'][_layer], 1)))

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
            
        self.pos_embedding = nn.Parameter(torch.randn(1, input_frames, self.num_nodes, dim))
        self.pred_token = nn.Parameter(torch.randn(1, 1, self.num_nodes, dim))
        
        self.to_embedding = nn.Linear(in_channels, dim)
        # self.dropout = nn.Dropout2d()
        self.to_latent = nn.Identity()
        self.encoder_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, dim))
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, in_channels),
            nn.ReLU()
            )


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
        
        
        
        # for layer_idx in range(len(self.conv1x1_encode)):
        #     x = self.conv1x1_encode[layer_idx](x)
        # n, c, t, v = x.shape
        # print('>>>>>> encoder')
        
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.to_embedding(x)    
        n, t, v, c = x.shape
        # print('x: ', x.shape)
        pred_token = repeat(self.pred_token, '() t v c-> n t v c', n = n)
        # print('pred_token: ', pred_token.shape)
        # print(x.shape, pred_token.shape)
        x = torch.cat((x, pred_token), dim=1)
        x = x + self.pos_embedding[:]
        # x = self.dropout(x)
        # print('x: ', x.shape)
        
        # x = x + copy.deepcopy(self.pos_embedding)[:]
        
        for layer_idx in range(self.num_GNN_layers_encoder):
            
            n, t, v, c = x.shape
            # res = x
            # print(0, x.shape)
            x = x.view(n*t, v, c)
            # x = x.permute(2, 0, 1).contiguous()    # v, nt, c
            # print(1, x.shape)
            x = self.ugformer_encoder[layer_idx](x)    # nt, v, c
            # print(2, x.shape)
            x = x.view(n, t, v, c)
            x = x.permute(0, 3, 1, 2).contiguous()    # n, c, t, v
            # print(3, x.shape)
            x = self.lst_gnn_encoder[layer_idx](x, self.A)   # n, c, t, v
            # print(4, x.shape)
            x = x.permute(0, 3, 2, 1).contiguous().view(n*v, t, c)
            x = self.tcn_encoder[layer_idx](x)   # nv, t, c
            x = x.view(n, v, t, c)
            x = x.permute(0, 2, 1, 3).contiguous()    # n, t, v, c
            # print(5, x.shape)
            # res = F.interpolate(res, size=x.shape[-2:], mode='bilinear', align_corners=False)
            # x = x + res    # n, t, v, c
            
        # x = x.contiguous().unsqueeze(1)
        # x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        
        # x_ref = x[:, -1, :, :]
        # x_ref = x_ref.unsqueeze(1)
        # x = self.encoder_head(x)    # n, t, v, c
        x_ref = x
        x = x[:, -1, :, :]
        x = x.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
        
        
        # x_ref = x
        x_size = x.size()
        # print('x_size: ', x_size)
        # x = x.view(n, -1)
        # print(x.shape, x_ref.shape, x_size)
        return x, x_size, x_ref


    def decode(self, z, x_size, x_ref=None):
        x = x_ref
        # print(x)
        # x = self.mlp_head(x)    # n, t, v, c
        # x = x[:, -1, :, :]
        # x = x.unsqueeze(1).permute(0, 3, 1, 2).contiguous()    # n, 1, v, c
        
        
        # print(out)
        # print('>>>>>')
        x = x.permute(0, 3, 1, 2).contiguous()
        for layer_idx in range(len(self.conv1x1_decode)):
            x = self.conv1x1_decode[layer_idx](x)
        x = self.decode_final(x)
        
        x = x[:, :, -1, :].unsqueeze(2)
        
        # if self.args.new_preprocess:
        #     x = self.sigmoid(x)
        # else:
        #     x = self.tanh(x)
            
        # x = self.sigmoid(x)
        
        # NM, c, t, v = x.size()
        # pred = x[:, :, -1, :]
        # x = x.view(N, M, c, t, v)
        # x = x.permute(0, 2, 3, 4, 1).contiguous()
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
