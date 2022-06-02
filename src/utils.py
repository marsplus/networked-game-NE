import numpy as np
import networkx as nx
import scipy.sparse as sp

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.cluster import KMeans
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities


## extract community information into a matrix M
## nonzero entries in the i-th row of M encode 
## these players in community i
def extract_community(G):
    min_comm = np.floor(len(G) * 0.005)
    comms = [list(c) for c in greedy_modularity_communities(G)]
    if any([len(c) <= min_comm for c in comms]):
        ### the 2nd term is to combine all agents in the communities with size less than min_comm 
        ### into a single community
        comms_final = [c for c in comms if len(c) > min_comm] + [[i for c in [L for L in comms if len(L) < min_comm] for i in c]]
    else:
        comms_final = comms
    num_comms = len(comms_final)
    n = len(G)
    M = np.zeros((num_comms, n))
    for i in range(num_comms):
        M[i, list(comms_final[i])] = 1
    return M



### extract community information based on spectral clustering
### which takes both the network structure and the agents' utilities 
### into account
def extract_community_spectralClustering(G, K, *param):
    """
        K: the number of groups
    """
    n = len(G)
    all_param = np.hstack([par.reshape(-1, 1) for par in param])

    ### get the weighted adjacency matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in G.neighbors(i):
            if j > i:
                A[i, j] = computeCosSim(all_param[i, :], all_param[j, :])
    A = (A + A.T) / 2


    ### get the Laplacian matrix
    L = np.diag(A.sum(axis=1)) - A

    ### SVD decomposition of the Laplacian
    U, _, _ = np.linalg.svd(L, hermitian=True)
    ### Discarding the last column of U, then 
    ### we keep the last K columns of the remaining
    U       = U[:, :-1][:, -1-K:-1]

    ### Kmeans on U
    km = KMeans(n_clusters=K).fit(U)
    M = np.zeros((K, n))
    for i in range(n):
        group_aff = km.labels_[i]
        M[group_aff, i] = 1

    return M


def gen_graph(n, graph, seed=123, ID=4, withinProb=0.2):
    """
        Generate individual-level graph
    """
    if graph == 'BA':
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif graph == 'ER':
        G = nx.gnm_random_graph(n, n*(n-1)*0.02/2)
    elif graph == 'SW':
        G = nx.watts_strogatz_graph(n, 10, 0.2, seed=seed)
    elif graph == 'RG':
        G = nx.random_geometric_graph(n, 0.4, seed=seed)
    elif graph == 'Email':
        G = nx.read_edgelist('../data/email-Eu-core-cc.txt', delimiter=' ', nodetype=int)
        mapping = {n: i for i, n in enumerate(list(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)
    elif graph == 'BTER':
        G = nx.read_edgelist(f'../data/BTER/BTER_{ID:02}.txt', delimiter=' ', nodetype=int)
    elif graph == 'SBM':
        ## simulate a stochastic-block model
        sizes    = [150, 100, 100, 80, 80, 120, 120, 50, 50, 30, 30]
        # off_prob = 0.001
        off_prob = 0.01
        in_prob  = lambda: np.random.uniform(withinProb-0.05, withinProb+0.05)
        # in_prob  = lambda: np.random.uniform(0.99, 1)
        nComm    = len(sizes)
        probs    = [[in_prob() if c == r else off_prob for c in range(nComm)] for r in range(nComm)]
        G        = nx.stochastic_block_model(sizes, probs, seed=seed)
    elif graph == 'Complete':
        G = nx.complete_graph(n)
    elif graph == 'Facebook':
        G = nx.read_edgelist('../data/facebook_combined.txt', nodetype=int)
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    return G



def gen_group_graph(nG, theta, M, adj, eps=1e-5):
    """
        Generate a group-level graph that somehow reflects the individual-level
        connectivity.
    """
    # assert(0 <= theta <= 1)

    # adjG   = M @ adj @ M.T          ### adjG[i, j]: the totoal number of connections from group i to group j
    # adjG   = np.diag(1 / (np.diag(adjG) + eps)) @ adjG
    # adjG  -= np.diag(np.diag(adjG)) ### zero out the diagonal entries

    # for k in range(nG):
    #     ### th: the threshold to determine if there is an edge between two groups
    #     th         = np.quantile(adjG[k, :], theta)
    #     idx_one    = np.nonzero(adjG[k, :] >= th)[0] 
    #     idx_zero   = np.nonzero(adjG[k, :] <  th)[0] 
    #     adjG[k, :][idx_one]  = 1
    #     adjG[k, :][idx_zero] = 0
    # return nx.from_numpy_matrix(adjG)

    adjG = M @ adj @ M.T
    adjG = np.diag(1 / (np.diag(adjG) + eps)) @ adjG @ np.diag(1 / (np.diag(adjG) + eps))
    adjG -= np.diag(np.diag(adjG))
    adjG[adjG > adjG.mean()] = 1
    adjG[adjG <= adjG.mean()] = 0
    return nx.from_numpy_matrix(adjG)

    



def gen_beta(G, var, M=None, mode=None, comple=True):
    n = len(G)
    # if mode == 'gaussian':
    #     beta = np.random.multivariate_normal(mean=np.zeros(n), cov=var*np.identity(n))
    if mode in ['homophily', 'fully-homophily', 'gaussian']:
        num_comm = M.shape[0]
        beta = np.zeros(n)
        for k in range(num_comm):        
            ## generate induced subgraph for each comm.
            idx = np.nonzero(M[k, :])[0]
            n_sub = len(idx)
            subgraph = G.subgraph(idx)
            adj_sub = np.array(nx.adjacency_matrix(subgraph).todense())
            lap_sub = np.diag(adj_sub.sum(axis=0)) - adj_sub
            lap_sub_inv = np.linalg.pinv(lap_sub, hermitian=True)
            if mode == 'gaussian':                    
                beta_sub = np.random.multivariate_normal(mean=np.zeros(n_sub), cov=np.diag(np.diag(lap_sub_inv)))
            elif mode == 'homophily':
                beta_sub = np.random.multivariate_normal(mean=np.zeros(n_sub), cov=lap_sub_inv)
            else:
                s = np.mean(np.diag(lap_sub_inv))
                beta_sub  = np.random.normal(scale=s) * np.ones(n_sub)
            beta[idx] = beta_sub
    elif mode == 'identical':
        ### strategic complementarity 
        factor_ = 1 if comple else -1
        beta = factor_ * np.random.uniform() * np.ones(n)
    elif mode in ['global-homophily', 'global-gaussian']:
        adj      = np.array(nx.adjacency_matrix(G).todense())
        lap      = np.diag(adj.sum(axis=1)) - adj
        lap_inv  = np.linalg.pinv(lap, hermitian=True)
        if mode == 'global-gaussian':
            beta = np.random.multivariate_normal(mean=np.zeros(n), cov=np.diag(np.diag(lap_inv)))
        else:
            beta = np.random.multivariate_normal(mean=np.zeros(n), cov=lap_inv)
    elif mode == 'uniform':
        beta = np.random.rand(n)
    return beta



## generate the b vector in linear-quadratic games
def gen_b(G, var, M=None, mode=None):
    n = len(G)
    ## no homophily
    # if mode == 'gaussian':
    #     b_vec = np.random.multivariate_normal(np.zeros(n), var*np.identity(n))
    if mode in ['homophily', 'fully-homophily', 'gaussian']:
        num_comm = M.shape[0]
        b_vec = np.zeros(n)
        for k in range(num_comm):
            ## generate induced subgraph for each comm.
            idx = np.nonzero(M[k, :])[0]
            n_sub = len(idx)
            subgraph = G.subgraph(idx)
            adj_sub = np.array(nx.adjacency_matrix(subgraph).todense())
            lap_sub = np.diag(adj_sub.sum(axis=0)) - adj_sub
            lap_sub_inv = np.linalg.pinv(lap_sub, hermitian=True)
            if mode == 'gaussian':                    
                b_sub = np.random.multivariate_normal(mean=np.zeros(n_sub), cov=np.diag(np.diag(lap_sub_inv)))
            elif mode == 'homophily':
                b_sub = np.random.multivariate_normal(mean=np.zeros(n_sub), cov=lap_sub_inv) 
            else:
                b_sub  = np.random.uniform() * np.ones(n_sub)
            b_vec[idx] = b_sub
    elif mode == 'identical':
        b_vec = np.random.uniform() * np.ones(n)
    elif mode == 'global-homophily':
        adj     = np.array(nx.adjacency_matrix(G).todense())
        lap     = np.diag(adj.sum(axis=1)) - adj
        lap_inv = np.linalg.pinv(lap, hermitian=True)
        b_vec   = np.random.multivariate_normal(mean=np.zeros(n), cov=lap_inv)
    elif mode == 'uniform':
        b_vec = np.random.rand(n)
    return b_vec


### compute cosine similarity
def computeCosSim(vec_a, vec_b):
    return np.dot(vec_a.squeeze(), vec_b.squeeze()) * 1.0 / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


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


## euclidean projection onto L2 ball
def euclidean_proj_l2(origin, epsilon):
    return origin / max(epsilon, np.linalg.norm(origin, 2))


