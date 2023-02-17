import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import GCN
from utils.config import parse_args

class QNet(nn.Module):
    def __init__(self, args, state_dim):
        super(QNet, self).__init__()
        self.hidden_dim = args.rl_hid
        self.layer1 = nn.Linear(state_dim, self.hidden_dim)
        self.layer2 = GCN(self.hidden_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, 1)
        
        self.reset_parameters()

    def forward(self, x, mps):  
        h = F.relu(self.layer1(x))
        num_meta_path = len(mps)
        embeds = []
        for i in range(num_meta_path):
            x = self.layer2(h, mps[i])
            embeds.append(x.unsqueeze(-1))
        embeds = torch.cat(embeds, dim=-1)
        embeds = torch.mean(embeds, dim=-1)
        q_value = self.out_layer(embeds)
       
        return q_value

    def get_embed(self, x, mps):
        h = F.relu(self.layer1(x))
        num_meta_path = len(mps)
        embeds = []
        for i in range(num_meta_path):
            x = self.layer2(h, mps[i])
            embeds.append(x.unsqueeze(-1))
        embeds = torch.cat(embeds, dim=-1)
        embeds = torch.mean(embeds, dim=-1)
        return embeds

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.out_layer.reset_parameters()

if __name__ == "__main__":
    from dataset import GraphLoader
    args = parse_args()
    G = GraphLoader("acm")
    statedim = 4
    qnet = QNet(args, statedim).cuda()
    state = torch.randn(G.X.shape[0], statedim).cuda()
    q_value = qnet.get_embed(state, G.adj_list)
    print(q_value.shape)

        



