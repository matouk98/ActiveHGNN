import numpy as np
import torch
import time
import os
import argparse
from torch.distributions import Categorical
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    
    parser.add_argument("--rllr",type=float,default=0.001)
    parser.add_argument("--rl_bs",type=int,default=16)
    parser.add_argument("--rl_epoch",type=int,default=300)
    parser.add_argument("--rl_update",type=int,default=20)
    parser.add_argument("--rl_train_epoch",type=int,default=100)
    parser.add_argument("--rl_train_iters",type=int,default=10)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--rl_hid",type=int,default=8)

    parser.add_argument("--mql_epoch",type=int,default=20)
    parser.add_argument("--mql_update",type=int,default=5)
    
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--dataset",type=str,default="acm")
    parser.add_argument("--tgt_dataset",type=str,default="acm")
    parser.add_argument("--state",type=str,default="s4")
    
    parser.add_argument("--query_init",type=int,default=10)
    parser.add_argument("--query_bs",type=int,default=15)
    parser.add_argument("--query_cs",type=int,default=5)

    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_centrality",type=int,default=1)
    parser.add_argument("--use_dist",type=int,default=1)

    args = parser.parse_args()

    return args
