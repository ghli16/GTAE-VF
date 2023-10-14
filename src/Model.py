import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from graphT import *
from torch_geometric.data import Data,Batch
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
class main_model(nn.Module):
    def __init__(self, args):
        super(main_model, self).__init__()
        self.encoder = DIFFormer(args.num_node_features, args.hidden_channels, args.out_channels,
              num_heads=args.num_heads, num_layers=args.num_layers, kernel=args.kernel,
              use_bn=args.use_bn, use_residual=args.use_residual, use_graph=args.use_graph,
              use_weight=args.use_weight, alpha=args.alpha,
              dropout=args.dropout).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decoder = DIFFormer(args.num_node_features1, args.hidden_channels1, args.out_channels1,
              num_heads=args.num_heads1, num_layers=args.num_layers1, kernel=args.kernel1,
              use_bn=args.use_bn1, use_residual=args.use_residual1, use_graph=args.use_graph1,
              use_weight=args.use_weight1, alpha=args.alpha1,
              dropout=args.dropout).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.MLP1 = nn.Linear(args.mlpin_dim, args.mlp1_dim)
        self.MLP2 = nn.Linear(args.mlp1_dim, args.mlp2_dim)
        self.MLP3 = nn.Linear(args.mlp2_dim, args.mlp3_dim)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.pool = pyg_nn.global_mean_pool


    def forward(self, data):
       batch = data.batch
       data_decoder = data

       fea = self.encoder(data_decoder)
       data_decoder.x = fea
       decoder_output = self.decoder(data_decoder)

       feature_s = self.pool(fea, batch)
       feature = self.MLP1(feature_s)
       #feature = self.MLP2(feature)
       #feature = self.MLP3(feature)

       #feature = global_self_attention_pool(feature, batch)
       active_func = nn.Sigmoid()
       output = active_func(feature)
       return output, decoder_output, feature_s



