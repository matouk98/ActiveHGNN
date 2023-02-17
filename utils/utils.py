import os, time, multiprocessing
import gc
import random
import numpy as np
import scipy
import scipy.spatial
import torch
import sys
sys.path.append('../')
from dataset import GraphLoader

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        if self.stdout is None:
            print("error!")
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def Euclidean_Distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_sim(x, y):
    return (torch.matmul(x, y.t()) + 1) / 2
    
def uniform_actions(pool, bs=200):
    actions = np.random.choice(pool, bs, replace=False).tolist()
    return actions

def greedy_actions(pool, value, bs=200):
    value = value.view(-1)
    in_pool_value = value[pool]
    in_pool_choose = torch.argsort(in_pool_value, descending=True)[:bs].long()
    actions = pool[in_pool_choose]
    
    actions = actions.numpy().tolist()
    return actions

def print_rl_params(args):
    print("Source Dataset:{}\tTarget Dataset:{}".format(args.dataset, args.tgt_dataset))
    print("Uncertainty:{}\tCentrality:{}\tEmb_Distance:{}".format(args.use_entropy, args.use_centrality, args.use_dist))
    print("Query Init:{}\tQuery BS:{}".format(args.query_init, args.query_bs))
    print("RL Learning Rate:{:.4f}".format(args.rllr))
    print("RL Batch Size:{}".format(args.rl_bs))
    print("RL Update Freq:{}".format(args.rl_update))
    print("RL Train Iters:{}".format(args.rl_train_iters))
    print("RL Cls Model Train Epoch:{}".format(args.rl_train_epoch))
    print("Gamma:{:.2f}".format(args.gamma))
    print("Hidden Size:{}".format(args.rl_hid))

def print_mql_params(args):
    print("Source Dataset:{}\tTarget Dataset:{}".format(args.dataset, args.tgt_dataset))
    print("Query Init:{}\tQuery BS:{}".format(args.query_init, args.query_bs))
    print("RL Learning Rate:{:.4f}".format(args.rllr))
    print("RL Batch Size:{}".format(args.rl_bs))
    print("MQL Epochs:{}".format(args.mql_epoch))
    print("MQL Update Freq:{}".format(args.mql_update))
    print("RL Train Iters:{}".format(args.rl_train_iters))
    print("RL Cls Model Train Epoch:{}".format(args.rl_train_epoch))
    print("Gamma:{:.2f}".format(args.gamma))
    print("Hidden Size:{}".format(args.rl_hid))

def print_res(acc, macro, micro):
    acc_str, macro_str, micro_str = 'Acc:{:.2f}'.format(acc[0]*100.0), 'Macro:{:.2f}'.format(macro[0]*100.0), 'Micro:{:.2f}'.format(micro[0]*100.0) 
    for i in range(1, acc.shape[0]):
        acc_str += '\t{:.2f}'.format(acc[i]*100.0)
        macro_str += '\t{:.2f}'.format(macro[i]*100.0)
        # micro_str += '\t{:.2f}'.format(micro[i]*100.0)
    print(acc_str)
    print(macro_str)
    # print(micro_str)

def get_graph(dataset):
    return GraphLoader(dataset)

if __name__ == '__main__':
    pass
    
    