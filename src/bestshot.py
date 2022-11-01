import os
import time
import argparse
import numpy as np
import networkx as nx
import dill as pickle

from Games import Bestshot
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=500)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--n',             type=int,   default=910)
    parser.add_argument('--graph',         type=str,   default='Email')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--seed',          type=int,   default=15)
    parser.add_argument('--elem',          type=int,   default=1)
    parser.add_argument('--aggre',         type=str,   default='mean')
    parser.add_argument('--mode',          type=str,   default='sequential')
    parser.add_argument('--fPath',         type=str,   default='../result/tmp.txt')
    args  = parser.parse_args()
    np.random.seed(args.seed)
    Aggre_ = np.mean if args.aggre == 'mean' else np.median

    G = gen_graph(n=args.n, graph=args.graph, seed=args.seed)
    n = len(G)
    #print("Finished read input graph.")


    ### get communities
    comms = extract_community(G, args.graph)
    nG    = len(comms)
    #print("Finished extracting communities.")


    ### group-level graph
    GG = gen_group_graph(G, comms)
    #print("Finished generating group-level graphs")


    # print("Start constructing games.")
    ### define individual-level games
    b_vec     = gen_b(G, var=args.b_var, comms=comms, mode=args.b_mode)
    indivGame = Bestshot(n,  G, b_vec)


    # ### tmporary use to check if the game Hessian is negative definite
    # A = np.array(nx.adjacency_matrix(G).todense())
    # S = np.diag(beta_vec) @ A - np.eye(n)
    # M = S + S.T

    ### define group-level games
    b_vec_g    = np.array([Aggre_(b_vec[comms[k]]) for k in range(len(comms))])
    groupGame  = Bestshot(nG,  GG, b_vec_g)


    ### compute NE
    t_ne = time.time()
    x_ne, x_ne_L = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, mode=args.mode)
    t_ne = time.time() - t_ne

    t_group = time.time()
    y_ne, y_ne_L = groupGame.grad_BR(maxIter=args.maxIter, lr=args.lr, mode=args.mode)
    ### compute x_hat and y_hat
    x_hat = np.zeros(x_ne.shape)
    y_hat = np.zeros(y_ne.shape)
    for k, com in comms.items():
        x_hat[com] = y_ne[k]
        y_hat[k]   = Aggre_(x_ne[com])
    x_hat_reg = indivGame.regret(x_hat)
    y_hat_reg = groupGame.regret(y_hat)

    ### BR with x_hat as starting point
    x_group, x_hat_L = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, x_init=x_hat, mode=args.mode)
    t_group = time.time() - t_group


    x_ne_reg = indivGame.regret(x_ne)
    y_ne_reg = groupGame.regret(y_ne)

    
    ### generate a random individual-level profile
    x_random   = np.random.rand(n)
    x_random_reg = indivGame.regret(x_random) 


    ### compute some distances
    y_dist = np.linalg.norm(y_hat-y_ne)
    x_dist = np.linalg.norm(x_hat-x_ne)
    x_dist_random = np.linalg.norm(x_random-x_ne)

    # print(f"x_ne_q: {x_ne_reg:.6f}  |  y_ne_q: {y_ne_reg:.6f}  |  x_hat_q: {x_hat_reg:.6f}  |  x_random_q: {x_random_reg:.6f}")



    # with open(f'../result/Youtube/b_{args.b_mode}_{args.graph}_{args.beta_mode}_numComm_{args.nComm}.p', 'wb') as fid:
    #     pickle.dump([x_ne_L, x_hat_L], fid)
        
    
    MIS_group = np.where(x_group == 1)[0]
    MIS_ne = np.where(x_ne == 1)[0]

    violation_group = len(G.subgraph(MIS_group).edges())
    violation_ne    = len(G.subgraph(MIS_ne).edges())
    # print(f"MIS group: {len(MIS_group)} (violation={violation_group})    time: {t_group:.4f}")
    # print(f"MIS ne: {len(MIS_ne)} (violation={violation_ne})    time: {t_ne:.4f}")

    # ### compute social welfare
    SW_ne = indivGame.sw(x_ne) / args.n
    SW_group = indivGame.sw(x_group) / args.n
    # print(f"{SW_ne:.6f},{SW_group:.6f}")

    for i in range(len(x_hat_L)):
        # print(f"{i*10}    random init: {x_ne_L[i]:.4f}    x_hat init: {x_hat_L[i]:.4f}")
        print(f"{x_ne_L[i]},{x_ne_L[i]},{x_hat_L[i]}")
    print('---' + f"{MIS_group},{violation_group},{MIS_ne},{violation_ne},{SW_ne},{SW_group}" '---')








