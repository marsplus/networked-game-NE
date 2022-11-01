import os
import time
import argparse
import numpy as np
import networkx as nx
import dill as pickle

from GamesLarge import LQGame
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--n',             type=int,   default=910)
    parser.add_argument('--graph',         type=str,   default='BTER')
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--beta_mode',     type=str,   default='fully-homophily')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--beta_var',      type=float, default=0.1)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--random_comm',   type=int,   default=0)
    parser.add_argument('--nComm',         type=int,   default=500)
    parser.add_argument('--aggre',         type=str,   default='mean')
    parser.add_argument('--fPath',         type=str,   default='../result/tmp.txt')
    args  = parser.parse_args()
    np.random.seed(args.seed)
    LB = 0.0
    UB = 1.0
    Aggre_ = np.mean if args.aggre == 'mean' else np.median

    G = gen_graph(n=args.n, graph=args.graph, seed=args.seed)
    n = len(G)
    # print("Finished read input graph.")


    ### get communities
    comms = extract_community(G, args.graph, nComm=args.nComm)
    comms = random_community(G, comms) if args.random_comm else comms
    nG    = len(comms)


    ### group-level graph
    GG = gen_group_graph(G, comms)


    ### define individual-level games
    b_vec     = gen_b(G, var=args.b_var, comms=comms, mode=args.b_mode)
    beta_vec  = gen_beta(G, var=args.beta_var, comms=comms, mode=args.beta_mode)
    indivGame = LQGame(n, b_vec, beta_vec, G, lb=LB, ub=UB)

    ### define group-level games
    b_vec_g    = np.array([Aggre_(b_vec[comms[k]]) for k in range(len(comms))])
    beta_vec_g = np.array([Aggre_(beta_vec[comms[k]]) for k in range(len(comms))])
    groupGame  = LQGame(nG, b_vec_g, beta_vec_g, GG, lb=LB, ub=UB)


    ### starting point
    x_start = np.random.rand(n)

    # ### compare with baselines (Vickrey & Koller)
    # t_VK = time.time()
    # x_VK, L_VK = indivGame.regretMIN(maxIter=args.maxIter, x_init=x_start)
    # t_VK = time.time() - t_VK

    ### BR without group-level info
    t_BR = time.time()
    x_BR, L_BR = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, x_init=x_start, full_update=True)
    t_BR = time.time() - t_BR

    ### BR with group info
    ### compute x_hat and y_hat
    t_BRG = time.time()
    y_BR, _ = groupGame.grad_BR(maxIter=args.maxIter, lr=args.lr)
    x_hat = np.zeros(x_BR.shape)
    for k, com in comms.items():
        x_hat[com] = y_BR[k]
    x_BRG, L_BRG = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, x_init=x_hat, full_update=True)
    t_BRG = time.time() - t_BRG


    ### output results
    try:
        for line in zip(*[L_VK, L_BR, L_BRG]):
            print(','.join(map(str, line)))
    except:
        for line in zip(*[L_BR, L_BR, L_BRG]):
            print(','.join(map(str, line)))
    print('---')

    # ### output running time
    # print(f"{t_VK},{t_BR},{t_BRG}")

    







