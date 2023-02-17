import os
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import sys

from utils.config import parse_args
from utils.evaluate import eval_metrics
from utils.utils import get_graph
from copy import deepcopy
from model import MGCN


class Player(nn.Module):

    def __init__(self, G, args):

        super(Player, self).__init__()
        self.G = G
        self.args = args
        
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.fulllabel = self.G.Y
        self.net = MGCN(self.args.nhid, self.G.stat['nclass'], self.G.feat_dim_list, self.args.feat_drop, self.args.attn_drop, self.G.num_meta_path)
        self.net = self.net.cuda()
        self.reset() #initialize
        
    def makeValTestMask(self):
        valmask = torch.zeros(self.G.stat['nnode']).to(torch.float)
        testmask = torch.zeros(self.G.stat['nnode']).to(torch.float)
       
        testid = self.G.idx_test
        testmask[testid] = 1.
        self.testlabel = self.G.Y[testid]
        valid = self.G.idx_val
        valmask[valid] = 1.
        self.vallabel = self.G.Y[valid]
        if hasattr(self.G, 'idx_train'):
            trainid = self.G.idx_train
        else:
            trainid = []
            for i in range(self.G.stat['nnode']):
                if testmask[i] + valmask[i] == 0:
                    trainid.append(i)

        self.valid = torch.tensor(valid).long()
        self.testid = torch.tensor(testid).long()
        self.trainid = torch.tensor(trainid).long()
        # print(self.valid.shape, self.testid.shape, self.trainid.shape)
        self.valmask = valmask.cuda()
        self.testmask = testmask.cuda()

    def query(self, nodes):
        if torch.is_tensor(nodes):
            nodes = nodes.cpu().numpy().tolist()
        for node in nodes:
            if self.trainmask[node] + self.valmask[node] + self.testmask[node] > 0:
                print("error!")
        
        self.trainmask[nodes] = 1.

    def remove(self, nodes):
        if torch.is_tensor(nodes):
            nodes = nodes.cpu().numpy().tolist()
        for node in nodes:
            if self.trainmask[node] == 0:
                print("error!")
            
        self.trainmask[nodes] = 0
    
    def get_pool(self):
        remain_pos = []
        for i in range(self.trainid.shape[0]):
            nodeid = self.trainid[i]
            if self.testmask[nodeid] + self.valmask[nodeid] + self.trainmask[nodeid] == 0:
                remain_pos.append(nodeid)

        remain_pos = torch.tensor(remain_pos).long()
        return remain_pos

    def train_once(self):
        nlabeled = torch.sum(self.trainmask)
        self.net.train()
        self.opt.zero_grad()

        output = self.net(self.G.X, self.G.adj_list)
        losses = self.loss_func(output, self.fulllabel)
        
        loss = torch.sum(losses * self.trainmask) / nlabeled
        loss.backward()
        self.opt.step()
        
        return output

    def get_output(self):
        with torch.no_grad():
            self.net.eval()
            self.outputs = self.net(self.G.X, self.G.adj_list).detach()

    def get_embed(self):
        with torch.no_grad():
            self.net.eval()
            self.embeds = self.net.get_embed(self.G.X, self.G.adj_list).detach()
        
    def validation(self, test=False):
        if test:
            mask = self.testmask
            labels = self.testlabel
            index = self.testid
        else:
            mask = self.valmask
            labels = self.vallabel
            index = self.valid
        
        with torch.no_grad():
            self.net.eval()
            output = self.net(self.G.X, self.G.adj_list)
            pred = torch.argmax(output, dim=1)
            pred = pred[index]
            acc, f1_macro, f1_micro = eval_metrics(pred.detach().cpu(), labels.detach().cpu())

        return acc, f1_macro, f1_micro

    def trainmodel(self, epochs=100):
        best_val, best_acc, best_f1_macro, best_f1_micro = 0.0, 0.0, 0.0, 0.0
        val_list = []
        for i in range(epochs):
            self.train_once()
            val_acc, val_f1_macro, val_f1_micro = self.validation()
            if val_acc > best_val:
                best_val = val_acc
                test_acc, f1_macro, f1_micro = self.validation(test=True)
                best_acc = test_acc
                best_f1_macro = f1_macro
                best_f1_micro = f1_micro
                best_model = deepcopy(self.net)

        self.net.load_state_dict(best_model.state_dict())
        return best_acc, best_f1_macro, best_f1_micro

    def model_reset(self):
        self.net.reset_parameters()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=5e-4)
       
    def reset(self, resplit=True):
        if resplit:
            self.makeValTestMask()
        self.trainmask = torch.zeros(self.G.stat['nnode']).to(torch.float).cuda()
        self.model_reset()
        
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    G = get_graph("acm")
    p = Player(G, args)
    
    pool = p.get_pool()
    n = pool.shape[0]
    print(n)
    sample_prob = torch.ones(n) / n
    random_sample = torch.multinomial(sample_prob, num_samples=40)
    actions = pool[random_sample]

    p.query(actions)
    best_acc, best_f1_macro, best_f1_micro = p.trainmodel(100)
    print(best_acc)

    # tf = time.time()
    # acc = 0.0
    # for i in range(30):
    #     p.reset()
    #     pool = p.get_pool()
    #     n = pool.shape[0]
    #     sample_prob = torch.ones(n) / n
    #     random_sample = torch.multinomial(sample_prob, num_samples=10)
    #     actions = pool[random_sample]

    #     p.query(actions)
    #     best_acc, best_f1_macro, best_f1_micro = p.trainmodel(100)

    #     pool = p.get_pool()
    #     n = pool.shape[0]
    #     sample_prob = torch.ones(n) / n
    #     random_sample = torch.multinomial(sample_prob, num_samples=10)
    #     actions = pool[random_sample]

    #     p.query(actions)
    #     # p.model_reset()
    #     best_acc, best_f1_macro, best_f1_micro = p.trainmodel(100)
    #     acc += best_acc

    #     print(torch.sum(p.trainmask).item())
    
    # acc /= 30
    # print("Time:{:.2f}\tAcc:{:.2f}".format(time.time()-tf, acc*100))
    
    
    
    