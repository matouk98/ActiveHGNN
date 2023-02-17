import os
import torch
import networkx as nx
import numpy as np
from utils import load_data
from scipy import sparse

def pagerank(g):
    pg = nx.pagerank(g)
    num = len(pg)
    cenarray = np.zeros(num).astype(float)
    cenarray[list(pg.keys())] = list(pg.values())
    return cenarray

class GraphLoader(object):
    def __init__(self, name, root="./data", args=None):
        self.name = name
        self.args = args
        self.data_path = os.path.join(root, name)

        self.load_graph()
        self.process()

    def load_graph(self):
        nei_index_list, feat_list, adj_list, graph_list, label, type_num = load_data(self.name)
        self.type_num = type_num
        self.adj_list = adj_list
        self.graph_list = graph_list
        self.nei_index_list = nei_index_list
        self.Y = torch.argmax(label, dim=1)
        self.X = feat_list[0]
        self.num_meta_path = len(adj_list)
        self.feat_dim_list = [f.shape[1] for f in feat_list]

        idx = np.arange(self.X.shape[0])
        np.random.seed(218)
        np.random.shuffle(idx)
        self.idx_test = idx[:1000]
        self.idx_val = idx[1000:1500]

        self.stat = {}
        self.stat['name'] = self.name
        self.stat['nnode'] = self.X.shape[0]
        self.stat['nfeat'] = self.X.shape[1]
        self.stat['nclass'] = label.shape[1]
        self.stat['nmetapath'] = self.num_meta_path
        print(self.stat)

    def process(self):
        self.Y = self.Y.cuda()
        self.X = self.X.cuda()
        for i in range(self.num_meta_path):
            self.adj_list[i] = self.adj_list[i].cuda()

        g = nx.Graph()
        for i in range(self.type_num[0]):
            g.add_node(i)

        tot = self.type_num[0]
        for t, neighbor_list in enumerate(self.nei_index_list):
            for i, neighbor in enumerate(neighbor_list):
                for n in neighbor:
                    g.add_edge(i, tot+n)    
            tot += self.type_num[t+1]
        
        # print(g.number_of_nodes(), sum(self.type_num))
        pagerank = nx.pagerank(g)
        num = len(pagerank)
        cenarray = np.zeros(num).astype(float)
        cenarray[list(pagerank.keys())] = list(pagerank.values())
        self.cenarray = cenarray[:self.type_num[0]]
        # print(self.cenarray.shape, self.type_num)

if __name__ == "__main__":
    G = GraphLoader("acm")
    



    