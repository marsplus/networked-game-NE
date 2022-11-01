import os
import argparse
import numpy as np
import networkx as nx
import dill as pickle

from Games import LQGame
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=1)
    parser.add_argument('--nComm',         type=int,   default=500)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--mode',          type=str,   default='simultaneous')
    parser.add_argument('--aggre',         type=str,   default='mean')
    args  = parser.parse_args()
    np.random.seed(args.seed)
    UB = float('inf')
    LB = 0.0
    Aggre_ = np.mean if args.aggre == 'mean' else np.median

    n = 4
    G = nx.complete_graph(n)
    beta_vec  = -0.5 * np.ones(n)
    b_vec     = 1 * np.ones(n)
    indivGame = LQGame(n, G, b_vec, beta_vec, ub=UB, lb=LB)

    x_start = np.random.rand(n)
    x_ne, x_ne_L = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, mode=args.mode, x_init=x_start)

    for i in x_ne_L:
        print(i)
    print()

    # # ### some statistics
    # A = np.array(nx.adjacency_matrix(G).todense())
    # H = np.diag(beta_vec) @ A - np.eye(n)
    # M = (H + H.T) / 2
    # # K = np.eye(n) + args.lr * H
    # eigVal, eigVec = np.linalg.eig(H)
    # x_tmp = b_vec[np.logical_and(x_ne > 0, x_ne < 1)] + (np.diag(beta_vec) @ A @ x_ne)[np.logical_and(x_ne > 0, x_ne < 1)]




