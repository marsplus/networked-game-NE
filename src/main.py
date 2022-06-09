
import time
import argparse
import numpy as np
import networkx as nx

import torch
import torch.optim as optim
import dill as pickle
from Games import LQGame
from utils import gen_graph, gen_b, extract_community, gen_beta

from models import GCN
import torch.nn as nn


## Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', default=0)
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--num_random', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--tol', type=float, default=0.1)
parser.add_argument('--graph', type=str, default='BTER')
parser.add_argument('--group_graph', type=str, default='BA')
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--nfeat', type=int, default=1)
parser.add_argument('--nclass', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alter_iter', type=int, default=10)
parser.add_argument('--gnn_iter', type=int, default=100)
parser.add_argument('--elem', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


### load data
################################################
G = gen_graph(args.n, args.graph)
args.n = len(G)
M = extract_community(G)
adj = np.array(nx.adjacency_matrix(G).todense())
num_comm = M.shape[0]
b_vec = gen_b(G, M, mode='homophily')

# beta = np.random.rand(args.n) * 0.01
beta_var = 0.1
beta = gen_beta(G, beta_var, M, homophily=True)
################################################

### generate community level information
G_comm = gen_graph(num_comm, args.group_graph)
adj_comm = np.array(nx.adjacency_matrix(G_comm).todense())
N_mat = np.diag(1.0 / M.sum(axis=1))
b_comm = N_mat @ M @ b_vec
beta_comm = N_mat @ M @ beta
LQG_comm = LQGame(num_comm, b_comm, adj_comm, beta_comm)
y_ne = LQG_comm.approx_NE().numpy()
################################################

### compute information needed for projection
proj_target = (np.identity(num_comm) - N_mat @ np.diag(M @ beta) @ adj_comm) @ y_ne
proj_mat = N_mat @ M @ (np.identity(args.n) - np.diag(beta) @ adj)
################################################

LQG = LQGame(args.n, b_vec, adj, beta, proj_mat=proj_mat, proj_target=proj_target, proj_tol=args.tol)
# x_ne = LQG.approx_NE()
x_init = torch.rand(args.n)


def compare_heuristics():
    """
    Compare different heuristics to find the NE
    """

    # x_final,  L_regret_min  = LQG.regretMIN(x_init, maxIter=args.epochs)

    # ## save for analysis
    # with open(f'../result/VK/history_{args.graph}_beta_gaussian.p', 'wb') as fid:
    #     pickle.dump(L_regret_min, fid)


    ####################################################################################
    opti = 'Adam'
    if not args.baseline:
        x_final, L = LQG.grad_BR(maxIter=args.epochs, x_init=x_init, lr=args.lr, \
                                        elementwise=args.elem, optimizer=opti, mode=args.mode, baseline=False)

        ## save for analysis
        with open(f'../result/{args.mode}/{args.graph}_tol_{args.tol:.2f}_beta_gaussian_elem_{args.elem}_LR_{args.lr:.3f}_gGraph_{args.group_graph}_baseline_{args.baseline}_optimizer_{opti}.p', 'wb') as fid:
            pickle.dump(L, fid)
        
        # x_ = LQG.merit_SDP()
        # print()
    else:
        x_final, L = LQG.grad_BR(maxIter=args.epochs, x_init=x_init, lr=args.lr, \
                                        elementwise=args.elem, optimizer=opti, mode=args.mode, baseline=True)
        ## save baseline
        with open(f'../result/baseline/{args.graph}_LR_{args.lr:.3f}_gGraph_{args.group_graph}_elem_{args.elem}_baseline_{args.baseline}_optimizer_{opti}.p', 'wb') as fid:
            pickle.dump(L, fid)



def GNN_para_BR(x_init):
    """
    Parameterize the best-response function with a GNN
    """
    model = GCN(nin=args.n,
                nhid=args.hidden,
                nout=args.n,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)   
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    model.train()

    ## optimize GNN every [alter_iter] iterations of the game's 
    ## best response updating
    alter_iter = args.alter_iter
    GNN_train_epochs = args.gnn_iter
    Iter = int(args.epochs / alter_iter)
    for _ in range(Iter):
        # curr_regret = LQG.regret(x_init).sum()
        # print(f"regret: {curr_regret.item():.4f}")
        
        for _ in range(GNN_train_epochs):
            optimizer.zero_grad()
            x_new = model(x_init.reshape(-1, 1), adj_tensor)
            loss = LQG.regret(x_new.squeeze()).sum()
            loss.backward()
            optimizer.step()
            # print(f"GNN loss: {loss.item():.4f}")
        x_new = model(x_init.reshape(-1, 1), adj_tensor).detach().squeeze()
        curr_regret = LQG.regret(x_new).sum()
        print(f"regret after GNN: {curr_regret.item():.4f}")
        
        # customized_regret = lambda x: LQG.utility(x_new.detach().squeeze()) - LQG.utility(x)
        def customized_regret(x):
            x_opt = x_new.clone()
            r = torch.zeros(args.n)
            for i in range(args.n):
                x_ = x.clone()
                x_[i] = x_opt[i]
                r[i]  = max(LQG.utility_(x_, i) - LQG.utility_(x, i), 0)
            return r

        x_init, _ = LQG.grad_BR(maxIter=alter_iter, elementwise=False, x_init=x_init, \
                                optimizer='Adam', projection=False, customized_regret=customized_regret)   

        # dist = torch.dist(x_ne, x_init)
        dist = LQG.check_quality(x_init)
        print(f"Distance to be a NE: {dist.item():.4f}")
    print()

compare_heuristics()
#GNN_para_BR(x_init)
