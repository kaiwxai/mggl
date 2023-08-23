from __future__ import absolute_import
from __future__ import division

import torch.nn.init as init
import torch.nn.functional as F 
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd
import dgl
from sklearn import preprocessing
import math
from dgl.nn import GATConv


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, dilation=1, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation_size = dilation ** i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation_size, dilation=dilation_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        y = x.transpose(1, 2)
        for layer in self.layers:
            y = layer(y)
        return y.transpose(1, 2)

class DemandEncoder(nn.Module):
    def __init__(self, seq_len, num_features, num_hidden, s_layers=1, l_layers=1, kernel_size_s=3, kernel_size_l=7, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.s_layers = s_layers
        self.l_layers = l_layers
        self.kernel_size_s = kernel_size_s
        self.kernel_size_l = kernel_size_l
        self.dropout = dropout
        self.tcn_s = TemporalConvNet(num_features, num_hidden, kernel_size_s, dropout=dropout, num_layers=s_layers)
        self.tcn_l = TemporalConvNet(num_hidden, num_hidden, kernel_size_l, dropout=dropout, num_layers=l_layers)
        self.fc = nn.Linear(num_hidden * 2, num_hidden)

    def forward(self, x):
        demand_permuted = x.permute(0, 2, 1)
        out_s = self.tcn_s(demand_permuted)
        out_l = self.tcn_l(out_s)
        out = torch.cat([out_s[:, -1:, :], out_l[:, -1:, :]], dim=2)
        out = self.fc(out)
        return out.squeeze()


class DemandDecoder(nn.Module):
    def __init__(self, seq_len, num_features, num_hidden, s_layers, l_layers, kernel_size_s, kernel_size_l, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.s_layers = s_layers
        self.l_layers = l_layers
        self.kernel_size_s = kernel_size_s
        self.kernel_size_l = kernel_size_l
        self.dropout = dropout
        self.tcn_s = TemporalConvNet(num_features, num_hidden, kernel_size_s, dropout=dropout, num_layers=s_layers)
        self.tcn_l = TemporalConvNet(num_hidden, num_hidden, kernel_size_l, dropout=dropout, num_layers=l_layers)
        self.out_layer = nn.Linear(num_hidden, 1)

    def forward(self, z):
        x_hat = self.tcn_s(z.unsqueeze(-1)).squeeze(-1)
        for i in range(self.l_layers):
            x_hat = self.tcn_l(x_hat).squeeze(-1)
        # print(x_hat.shape)
        # .transpose(1, 2)
        x_hat = self.out_layer(x_hat[:, :self.seq_len, :]).permute(0, 2, 1)
        return x_hat

    
class MultiTypeAttributesCoEncoder(nn.Module):
    def __init__(self, dynamic_attributes_dim, static_attributes_dim, num_dynamic_attributes, num_industries):
        super(MultiTypeAttributesCoEncoder, self).__init__()
        self.dynamic_attributes_dim = dynamic_attributes_dim
        self.static_attributes_dim = static_attributes_dim
        self.num_dynamic_attributes = num_dynamic_attributes
        self.num_industries = num_industries
        # Linear layer for obtaining representative values from t_1 to t_n
        # self.dynamic_representation = nn.Linear(dynamic_attributes_dim, num_dynamic_attributes)
        # Transformer-based model for deriving probability values of the dynamic attributes
        self.embed_dim = 64
        self.dynamic_representation = nn.Linear(self.dynamic_attributes_dim, self.embed_dim)
        self.static_representation = nn.Linear(self.static_attributes_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, 1)
        self.num_heads = 8
        while (self.embed_dim % self.num_heads) != 0:
            self.embed_dim += 1
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=self.num_heads, num_encoder_layers=6)
        
        # Linear layer for concatenating the encoded dynamic and static attributes
        self.final_representation = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self, dynamic_attributes, static_attributes):
        # One-hot encoding of static attributes
        # static_attributes_one_hot = self.static_one_hot(static_attributes)
        # Obtain representative values from t_1 to t_n
        dynamic_representations = self.dynamic_representation(dynamic_attributes)
        static_representations = self.static_representation(static_attributes)
        # Construct a representative filtering network for user dynamic attributes using a transformer-based model
        dynamic_representations = dynamic_representations.permute(1, 0, 2)
        sequence_length = dynamic_representations.size(0)
        # target = dynamic_representations
        # dynamic_representations = dynamic_representations[:-1]
#         print(dynamic_representations.shape)
#         print(dynamic_attributes.shape)
        frequency_values = self.value(self.transformer(dynamic_representations, dynamic_attributes.permute(1, 0, 2))).squeeze()
        frequency_values = frequency_values.permute(1, 0)
        # Derive the set of probability values for the dynamic attribute
        probability_values = F.softmax(frequency_values, dim=0)
        # Select the attribute with the highest probability as the representative value of the dynamic attribute
        selected_dynamic_attributes = torch.argmax(probability_values, dim=1)
        dynamic_representation = dynamic_representations[selected_dynamic_attributes, torch.arange(dynamic_attributes.shape[0]), :]
        # dynamic_representation = dynamic_representation.view(-1, self.embed_dim * self.num_dynamic_attributes)
        concatenated_attributes = torch.cat((static_representations, dynamic_representation), dim=1)
        final_representation = self.final_representation(concatenated_attributes)

        return final_representation

    
class seq2gauss_model(nn.Module):
    def __init__(self, time_feat_size, industry_size, device,
                 conv_channel=[64,64], conv_size=[1,3], embed_size=64, dropout=0.2, k=9):
        super(seq2gauss_model, self).__init__()
        self.time_feat_size = time_feat_size
        self.industry_size = industry_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.device = device
        self.k = k
        self.conv_channel = conv_channel
        self.conv_size = conv_size
        self.embed_time_mean = nn.Linear(self.time_feat_size, self.embed_size)
        self.embed_time_var = nn.Linear(self.time_feat_size, self.embed_size)
        self.embed_industry_mean = nn.Linear(self.industry_size, self.embed_size)
        self.embed_industry_var = nn.Linear(self.industry_size, self.embed_size)

        self.conv = nn.Sequential(
            nn.Conv2d(self.embed_size, conv_channel[0], conv_size[0]),
            nn.ELU()
        )

        for idx, (channel, kernel_size) in enumerate(zip(conv_channel[1:], conv_size[1:])):
            self.conv.add_module("conv", nn.Conv2d(conv_channel[idx], channel, kernel_size))
            self.conv.add_module("elu", nn.ELU())

        self.fc = nn.Sequential(
            nn.Linear(conv_channel[-1] * (k - sum(conv_size) + len(conv_size)) ** 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 64)
        )


    def forward(self, time_feat, industry_np):
        size = int(time_feat.shape[0])
        # industry_np = self.convert_one_hot(industry_np, self.industry_size)
        
        time_mean = self.embed_time_mean(time_feat).unsqueeze(1)		# (batch_size, 1, embed_size)
        time_std = (F.elu(self.embed_time_var(time_feat)) + 1).unsqueeze(1)	# (batch_size, 1, embed_size)
        industry_mean = self.embed_industry_mean(industry_np).unsqueeze(1)
        industry_std = (F.elu(self.embed_industry_var(industry_np)) + 1).unsqueeze(1)

        # reparameterize tricks: eps*std + mu
        samples_time = torch.randn((size, self.k, self.embed_size)).to(self.device) * time_std + time_mean 	# (batch_size, k, embed_size)
        samples_industry = torch.randn((size, self.k, self.embed_size)).to(self.device) * industry_std + industry_mean

        samples_time = samples_time.unsqueeze(2)			# (batch_size, k, 1, embed_size)
        samples_time = samples_time.repeat(1, 1, self.k, 1)		# (batch_size, k, k, embed_size)

        samples_industry = samples_industry.unsqueeze(2)
        samples_industry = samples_industry.repeat(1, 1, self.k, 1)		# (batch_size, k, k, embed_size)
        map = samples_time *  samples_industry.transpose(1, 2)		# (batch_size, k, k, embed_size)

        x = self.conv(map.permute(0, 3, 1, 2))
        x = x.reshape(size, -1)
        x = self.fc(x)
        
        x = torch.sigmoid(x)
        return x

    
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size()[0], -1)

class MVGAT(nn.Module):
    def __init__(self, num_graphs=3, num_gat_layer=2, in_dim=14, hidden_dim=64, emb_dim=32, num_heads=2, residual=True):
        super(MVGAT, self).__init__()
        self.num_graphs = num_graphs
        self.num_gat_layer = num_gat_layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.residual = residual

        self.multi_gats = nn.ModuleList()
        for j in range(self.num_gat_layer):
            gats = nn.ModuleList()
            for i in range(self.num_graphs):
                if j == 0:
                    gats.append(GATConv(self.in_dim,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                elif j == self.num_gat_layer - 1:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.emb_dim // self.num_heads,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                else:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
            self.multi_gats.append(gats)

    def forward(self, graphs, feat):
        views = []
        for i in range(self.num_graphs):
            for j in range(self.num_gat_layer):
                if j == 0:
                    z = self.multi_gats[j][i](graphs[i], feat)
                else:
                    z = self.multi_gats[j][i](graphs[i], z)
                if j != self.num_gat_layer - 1:
                    z = F.relu(z)
                z = z.flatten(1)
            views.append(z)
        return views


class FusionModule(nn.Module):
    def __init__(self, num_graphs, emb_dim, alpha):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.self_q = nn.ModuleList()
        self.self_k = nn.ModuleList()
        
        for i in range(self.num_graphs):
            self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, views):
        # run fusion by self attention
        cat_views = torch.stack(views, dim=0)
        self_attentions = []
        for i in range(self.num_graphs):
            Q = self.self_q[i](cat_views)
            K = self.self_k[i](cat_views)
            # (3, num_nodes, 64)
            attn = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim=-1)
            # (3, num_nodes, num_nodes)
            output = torch.matmul(attn, cat_views)
            self_attentions.append(output)
        self_attentions = sum(self_attentions) / self.num_graphs
        # (3, num_nodes, 64 * 2)
        for i in range(self.num_graphs):
            views[i] = self.alpha * self_attentions[i] + (1 - self.alpha) * views[i]
        # further run multi-view fusion
        mv_outputs = []
        for i in range(self.num_graphs):
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])
        fused_outputs = sum(mv_outputs)
        # next_in = [(view + fused_outputs) / 2 for view in views]
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]

    
class Agg_Graph(nn.Module):
    def __init__(self, num_nodes, num_areas, emb_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_nodes = num_areas
#         self.Theta1 = nn.Parameter(torch.FloatTensor(num_nodes, emb_dim))
        self.Theta2 = nn.Parameter(torch.FloatTensor(emb_dim, num_nodes))
        self.Theta3 = nn.Parameter(torch.FloatTensor(num_nodes, num_areas))
        
        self.reset_parameters()
    
    def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.Theta1.shape[1])
#         self.Theta1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Theta2.shape[1])
        self.Theta2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Theta3.shape[1])
        self.Theta3.data.uniform_(-stdv, stdv)
        

    def forward(self, embedding):
        embedding = torch.mm(embedding, self.Theta2)
        adj = torch.mm(self.Theta3.T, torch.mm(embedding, self.Theta3))
        adj = F.relu(adj)
        return embedding, adj.cpu()

    