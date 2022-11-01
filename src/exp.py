import os
import argparse
import numpy as np
import networkx as nx
import dill as pickle
# import matplotlib.pyplot as plt

from Games import LQGame, BHGame
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--n',             type=int,   default=50)
    parser.add_argument('--graph',         type=str,   default='BTER')
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--beta_mode',     type=str,   default='gaussian')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--beta_var',      type=float, default=0.1)
    parser.add_argument('--control_var',   type=float, default=0.001)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--elem',          type=int,   default=1)
    parser.add_argument('--random_comm',   type=int,   default=0)
    parser.add_argument('--nComm',         type=int,   default=500)
    parser.add_argument('--aggre',         type=str,   default='mean')
    parser.add_argument('--mode',          type=str,   default='sequential')
    parser.add_argument('--output',        type=int,   default=1)
    parser.add_argument('--traj',          type=int,   default=0)
    parser.add_argument('--game',          type=str,   default='BHG')
    args  = parser.parse_args()

    candidate_games = {

        'LQG': LQGame,
        'BHG': BHGame
    }
    game_instance = candidate_games[args.game]

    np.random.seed(args.seed)
    UB = 1.0
    LB = 0.0
    Aggre_ = np.mean if args.aggre == 'mean' else np.median

    G = gen_graph(n=args.n, graph=args.graph, seed=args.seed)
    n = len(G)

    ### get communities
    comms = extract_community(G, args.graph, nComm=args.nComm)
    comms = random_community(G, comms) if args.random_comm else comms
    nG    = len(comms)

    ### group-level graph
    GG = gen_group_graph(G, comms)


    # print("Start constructing games.")
    ### define individual-level games
    b_vec     = gen_b(G, var=args.b_var, comms=comms, mode=args.b_mode)
    beta_vec  = gen_beta(G, var=args.beta_var, comms=comms, mode=args.beta_mode, control_var=args.control_var)
    indivGame = game_instance(n, G, b_vec, beta_vec, ub=UB, lb=LB)

    ### define group-level games
    b_vec_g    = np.array([Aggre_(b_vec[comms[k]]) for k in range(len(comms))])
    beta_vec_g = np.array([Aggre_(beta_vec[comms[k]]) for k in range(len(comms))])
    groupGame  = game_instance(nG, GG, b_vec_g, beta_vec_g, ub=UB, lb=LB)


    ### starting point
    x_start = np.random.rand(n)


    ### compute NE
    x_ne, x_ne_L = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, mode=args.mode, x_init=x_start)
    y_ne, y_ne_L = groupGame.grad_BR(maxIter=args.maxIter, lr=args.lr, mode=args.mode)
    
    x_ne_reg = indivGame.regret(x_ne)
    y_ne_reg = groupGame.regret(y_ne)
    

    ### compute x_hat and y_hat
    x_hat = np.zeros(x_ne.shape)
    y_hat = np.zeros(y_ne.shape)
    for k, com in comms.items():
        x_hat[com] = y_ne[k]
        y_hat[k]   = Aggre_(x_ne[com])
    x_hat_reg = indivGame.regret(x_hat)
    y_hat_reg = groupGame.regret(y_hat)

    
    ### generate a random individual-level profile
    x_random   = np.random.rand(n)
    x_random_reg = indivGame.regret(x_random) 


    ### compute some distances
    y_dist = np.linalg.norm(y_hat-y_ne)
    x_dist_vanila = np.linalg.norm(x_start-x_ne)
    x_dist_random = np.linalg.norm(x_random-x_ne)

    # print(f"x_ne_q: {x_ne_reg:.6f}  |  y_ne_q: {y_ne_reg:.6f}  |  x_hat_q: {x_hat_reg:.6f}  |  x_random_q: {x_random_reg:.6f}")

    ### BR with x_hat as starting point
    x_group_ne, x_hat_L = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr, x_init=x_hat, mode=args.mode)
    x_dist_group = np.linalg.norm(x_hat-x_group_ne)

    # ### some statistics
    A = np.array(nx.adjacency_matrix(G).todense())
    B = np.diag(beta_vec)
    # P = B @ A - np.eye(n)
    # T = np.vstack((np.hstack((P @ A - np.eye(n), b_vec[:, None])), np.hstack((np.zeros(n)[None, :], np.array([[0]])))))
    # H = np.diag(beta_vec) @ A - np.eye(n)
    # x_tmp = b_vec[np.logical_and(x_ne > 0, x_ne < 1)] + (np.diag(beta_vec) @ A @ x_ne)[np.logical_and(x_ne > 0, x_ne < 1)]
    P = np.diag(2 / A.sum(axis=0)) @ A - 2 * np.eye(n)
    eigVal, eigVec = np.linalg.eig(P)
    # F = eigVec @ np.eye(n)
    # c_random = np.linalg.solve(F, x_start)
    # c_hat = np.linalg.solve(F, x_hat)

    if args.traj:
        for i in range(len(x_hat_L)):
            print(f"{x_ne_L[i]},{x_ne_L[i]},{x_hat_L[i]}")
        print('---')

    if args.output:
        print("{} {} {} {} {} {} {} {} {}".format(x_ne_reg, y_ne_reg, x_hat_reg, y_hat_reg, \
                                          x_random_reg, x_dist_vanila, x_dist_group, x_dist_random, y_dist))
































