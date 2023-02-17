import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
import sys
import networkx as nx
import pickle
from scipy import sparse

sys.path.append('../')

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def load_acm(path='./data/acm/'):
    # The order of node types: 0 p 1 a 2 s
    type_num = [4019, 7167, 60]
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pap_graph = nx.from_scipy_sparse_matrix(pap)
    psp_graph = nx.from_scipy_sparse_matrix(psp)
 
    # [4019]
    label = th.LongTensor(label)
    nei_a = [i.tolist() for i in nei_a]
    nei_s = [i.tolist() for i in nei_s]
    
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    # [4019, 4019]
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], [pap_graph, psp_graph], label, type_num

def load_dblp(path='./data/dblp/'):
    # The order of node types: 0 a 1 p 2 c 3 t
    type_num = [4057, 14328, 7723, 20]  # the number of every node type
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    apa_graph = nx.from_scipy_sparse_matrix(apa)
    apcpa_graph = nx.from_scipy_sparse_matrix(apcpa)
    aptpa_graph = nx.from_scipy_sparse_matrix(aptpa)
    
    label = th.LongTensor(label)
    nei_p = [i.tolist() for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
   
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], [apa_graph, apcpa_graph, aptpa_graph], label, type_num

def load_aminer(path='./data/aminer/'):
    # The order of node types: 0 p 1 a 2 r
    type_num = [6564, 13329, 35890]  # the number of every node type
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pap_graph = nx.from_scipy_sparse_matrix(pap)
    prp_graph = nx.from_scipy_sparse_matrix(prp)
    
    label = th.LongTensor(label)
    nei_a = [i.tolist() for i in nei_a]
    nei_r = [i.tolist() for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], [pap_graph, prp_graph], label, type_num

def load_data(dataset_name):
    if dataset_name == 'acm':
        return load_acm()
    elif dataset_name == 'dblp':
        return load_dblp()
    elif dataset_name == 'aminer':
        return load_aminer()
   
if __name__ == "__main__":
    load_data('acm')


