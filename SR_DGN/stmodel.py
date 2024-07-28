import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv

class STMODEL(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STMODEL, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.linear1 = nn.Linear(out_dim, num_hidden, bias=True)
        self.linear2 = nn.Linear(num_hidden, in_dim, bias=True)


    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h3 = F.elu(self.linear1(h2))
        h4 = self.linear2(h3)

        return h2, h4  # F.log_softmax(x, dim=-1)