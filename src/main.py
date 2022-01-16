
import time
import argparse
import numpy as np
import networkx as nx

import torch
import torch.optim as optim
import dill as pickle
from Games import LQGame
from utils import gen_graph, gen_b, extract_community

from models import GCN
import torch.nn as nn


## Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--num_random', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--tol', type=float, default=0.5)
parser.add_argument('--graph', type=str, default='BA')
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--nfeat', type=int, default=1)
parser.add_argument('--nclass', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alter_iter', type=int, default=50)
parser.add_argument('--gnn_iter', type=int, default=200)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


## load data
G = gen_graph(args.n, args.graph)
C = extract_community(G)
adj = nx.adjacency_matrix(G).todense()
b_vec = gen_b(args.n, adj, mode='ind')

# beta = np.random.rand(args.n) * 0.1
beta = np.random.multivariate_normal(mean=np.zeros(args.n), cov=0.03*np.identity(args.n))
#beta = 0.05

LQG = LQGame(args.n, b_vec, adj, beta, C, proj_tol=args.tol)
x_ne = LQG.approx_NE()
x_init = torch.rand(args.n)


def compare_heuristics():
    """
    Compare different heuristics to find the NE
    """
    # x_r,  L_r  = LQG.regretMIN(x_init, maxIter=args.epochs)

    _, L_br_withProj = LQG.grad_BR(maxIter=args.epochs, x_init=x_init, \
                                        elementwise=True, optimizer='Adam', projection=True)
    _, L_br_noProj = LQG.grad_BR(maxIter=args.epochs, x_init=x_init, \
                                        elementwise=True, optimizer='Adam', projection=False)

    # x_rg_adam, L_rg_adam = LQG.grad_BR(epochs=args.epochs, x_init=x_init, elementwise=False, optimizer='Adam')

    ## save for analysis
    with open(f'../result/proj/history_{args.graph}_tol_{args.tol:.2f}_beta_randgaussian.p', 'wb') as fid:
        pickle.dump([L_br_withProj, L_br_noProj], fid)



def GNN_para_BR(x_init):
    """
    Parameterize the best-response function with a GNN
    """
    model = GCN(nfeat=args.nfeat,
                nhid=args.hidden,
                nclass=args.nclass,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)   
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    model.train()

    ## optimize GNN every [alter_iter] iterations of the game's 
    ## best response updating
    L = []
    alter_iter = args.alter_iter
    GNN_train_epochs = args.gnn_iter
    for _ in range(args.epochs):
        x_init, _ = LQG.grad_BR(maxIter=alter_iter, x_init=x_init, \
                                            elementwise=True, optimizer='Adam', projection=False)        

        x_new = model(x_init.reshape(-1, 1), adj_tensor)
        fixed_point_dist = torch.dist(x_new.detach().squeeze(), x_init)
        print(f"Distance to be a fixed point: {fixed_point_dist:.4f}")
        if fixed_point_dist > 1e-3:
            for _ in range(GNN_train_epochs):
                optimizer.zero_grad()
                x_new = model(x_init.reshape(-1, 1), adj_tensor)
                loss = LQG.regret(x_new.squeeze()).max()
                loss.backward()
                optimizer.step()
                # print(f"GNN loss: {loss.item():.4f}")
        
        x_init = x_new.squeeze().detach()
        dist = torch.dist(x_init, x_ne)
        print(f"Distance to be a NE: {dist:.4f}")
        L.append(dist.item())
    print()

GNN_para_BR(x_init)
