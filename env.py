import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from player import Player
from utils.config import parse_args
from utils.utils import uniform_actions
from sklearn.cluster import KMeans

def percd(input):
    ranks = torch.argsort(input)
    ranks = (torch.argsort(ranks) + 1.0) / input.shape[0]
    return ranks

def perc(input):
    ranks = np.argsort(input, kind='stable')
    ranks = (np.argsort(ranks, kind='stable') + 1) / input.shape[0]
    return ranks

def calc_centrality(G):
    centralities = nx.pagerank(G) #centralities.append(nx.harmonic_centrality(G))
    L = len(centralities)
    cenarray = np.zeros(L).astype(float)
    
    cenarray[list(centralities.keys())] = list(centralities.values())
    maxi, mini = np.max(cenarray), np.min(cenarray)
    normcen = (cenarray - mini) / (maxi - mini)
    
    return torch.from_numpy(perc(normcen)).float()

def calc_entropy(output):
    entropy = torch.sum(- output * torch.log(output + 1e-7), dim=1)
    entropy /= math.log(output.shape[1])
    return entropy

def get_centrality(graph_list):
    centrality = []
    for g in graph_list:
        normcen = calc_centrality(g)
        centrality.append(normcen.unsqueeze(1))
    centrality = torch.cat(centrality, dim=-1)
    centrality = torch.mean(centrality, dim=-1)
    return centrality

def localdiversity(probs, adj, deg):
    indices = adj.coalesce().indices()
    
    N = adj.size()[0]
    classnum = probs.size()[-1]
    maxentro = np.log(float(classnum))
    edgeprobs = probs[:,indices.transpose(0,1),:]
    
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)

    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))
    
    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)
    mean_kl_ht = sum_kl_ht/(deg+1e-10)
    mean_kl_th = sum_kl_th/(deg+1e-10)
    # normalize
    mean_kl_ht = mean_kl_ht / mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th = mean_kl_th / mean_kl_th.max(dim=1, keepdim=True).values
    return mean_kl_ht,mean_kl_th

def Euclidean_Distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self, player, tgt_player, args):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.player = player
        self.tgt_player = tgt_player
        self.args = args
        
        self.cenperc = torch.from_numpy(perc(player.G.cenarray)).float()
        self.tgt_cenperc = torch.from_numpy(perc(tgt_player.G.cenarray)).float()
        
        self.statedim = self.get_state().size(-1)
        print("State Dim:{}".format(self.statedim))

    def step(self, actions, n_epochs=100, dataset="source", query=True):
        if dataset == "source":
            p = self.player
        else:
            p = self.tgt_player
        if query:
            p.query(actions)
        
        best_acc, best_f1_macro, best_f1_micro = p.trainmodel(epochs=n_epochs)
        return best_acc, best_f1_macro, best_f1_micro

    def reset(self, dataset='source'):
        if dataset == 'source':
            self.player.reset()
        else:
            self.tgt_player.reset()

    def init_train(self, dataset='source'):
        if dataset == 'source':
            p = self.player
        else:
            p = self.tgt_player
        init_set = uniform_actions(p.get_pool(), bs=self.args.query_init)
        p.query(init_set)
        acc, f1_macro, f1_micro = p.trainmodel()
        return acc, f1_macro, f1_micro

    def get_state(self, dataset='source'):
        features = []
        if dataset == "source":
            p = self.player
        else:
            p = self.tgt_player
        
        if self.args.use_entropy:
            p.get_output()
            output = F.softmax(p.outputs, dim=1)
            entropy = calc_entropy(output)
            features.append(entropy.float().cuda())

        if self.args.use_centrality:
            if dataset == 'target':
                features.append(self.tgt_cenperc.float().cuda())
            else:
                features.append(self.cenperc.float().cuda())
        
        if self.args.use_dist:
            p.get_embed()
            embeds = p.embeds.detach()
            has_select = torch.where(p.trainmask==1)[0]
            if not has_select.shape[0]:
                has_select = [1]
            y_embeds = embeds[has_select, :].detach()
            
            coreset_dist = Euclidean_Distance(embeds, y_embeds)
            coreset_dist = torch.min(coreset_dist, dim=1)[0]
            coreset_dist = percd(coreset_dist)
            features.append(coreset_dist)
       
        selected = p.trainmask + p.valmask + p.testmask
        features.append(1 - selected.float().cuda())
        state = torch.stack(features, dim=-1)
        
        return state

if __name__ == "__main__":
    from dataset import GraphLoader
    tf = time.time()
    args = parse_args()
    G = GraphLoader(args.dataset)
    
    p = Player(G, args)
    print("Time:{:.2f}\tPlayer OK".format(time.time()-tf))
    env = Env(p, p, args)
    print("Time:{:.2f}\tEnv OK".format(time.time()-tf))
    env.reset()
    env.init_train()
    print("Time:{:.2f}\tInit OK".format(time.time()-tf))
    '''
    for i in range(args.query_cs):
        actions = env.uniform_actions(args.query_bs)
        reward = env.step(actions)
        print(reward)
    '''

    
