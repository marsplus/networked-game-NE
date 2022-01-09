import numpy as np
import networkx as nx
import scipy.sparse as sp

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def gen_graph(n, graphType, seed=123):
    if graphType == 'BA':
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif graphType == 'SW':
        G = nx.watts_strogatz_graph(n, 10, 0.2, seed=seed)
    elif graphType == 'RG':
        G = nx.random_geometric_graph(n, 0.2, seed=seed)
    return G

## initialize action probability
def gen_action_logprob(n, actionDim, seed=123):
    np.random.seed(seed)
    action_prob = np.random.rand(n, actionDim)
    action_prob /= action_prob.sum(axis=1)[:, np.newaxis]
    return torch.from_numpy(np.log(action_prob))   

## output the adjacency matrix and each player's strategy
def gen_data(n, graphType, actionDim, seed=123):
    G = gen_graph(n, graphType, seed=seed)
    adj = sp.coo_matrix(nx.adjacency_matrix(G))
    action_prob = gen_action_logprob(n, actionDim)
    playerToNeigh = {i: torch.LongTensor(list(G.neighbors(i))) for i in range(len(G))}
    return sparse_mx_to_torch_sparse_tensor(adj), action_prob, playerToNeigh


## sample from a gumbel distribution
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


## gumble softmax
def gumbel_softmax(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    g = F.softmax(y / temperature, dim=-1)
    shape = g.size()
    g_hard = torch.zeros_like(g).view(shape[-1])
    g_hard[torch.argmax(g)] = 1
    return (g_hard - g).detach() + g


## sample players' actions
def sample_action(n, strategy, temperature=0.8):
    all_actions = torch.zeros(n, strategy.size()[-1])
    for i in range(n):
        logprobs = strategy[i, :]
        all_actions[i, :] = gumbel_softmax(logprobs, temperature)
    return all_actions


## the regret for Best-Shot games
def best_shot_regret(n, all_actions, neighborMap, c=0.3):    
    ## compute regret
    regret = 0.0
    for i in range(n):
        self_action = all_actions[i, 1]
        ## the number of neighbors selecting 1
        neighbor_action_cnt = all_actions[neighborMap[i], 1].sum()
        ## self_action > 0, neighbor_action_cnt > 0, regret = c
        ## self_action = 0, neighbor_action_cnt = 0, regret = 1 - c
        regret += max(0, 1 - self_action - neighbor_action_cnt) * (1 - c) - \
                  self_action * neighbor_action_cnt * min(0, 1 - self_action - neighbor_action_cnt) * c 
    return regret


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)