import os
import torch
import argparse
import numpy as np
import networkx as nx
import dill as pickle

from Games import LQGame
from utils import gen_group_graph, gen_graph, gen_b, extract_community, gen_beta, computeCosSim


def ComputeBetweenGroupSim(M, N, adj, game):
    """
        Compute between group similarity inspired by spectral clustering.
        
        M:   group membership matrix 
        N:   normalization matrix
        adj: unweighted and undirected; we make it a weighted adjacency matrix
             by letting the weight of (i,j) be the cosine similarity between their parameters
    """
    ### generate the matrix H
    H = M.T @ np.sqrt(N)

    ### convert adj to a weighted adjacency matrix
    A = np.zeros(adj.shape)
    for i in range(game.n):
        for j in game.adjList[i]:
            if j > i:
                param_i = np.array([game.b[i].numpy(), game.beta[i].numpy()])
                param_j = np.array([game.b[j].numpy(), game.beta[j].numpy()])
                A[i, j] = computeCosSim(param_i, param_j)
    A = (A + A.T) / 2

    ### compute the Laplacian matrix
    L = np.diag(A.sum(axis=1)) - A

    ### compute the between group similarity
    sim = np.trace(H.T @ L @ H)

    return sim


### convert a group-level NE (i.e., y*) to an individual-level action profile (i.e., \hat{x})
def yStarToXHat(y_ne, M):
    """
        M: encoding group membership
    """
    return (y_ne[np.newaxis, :].numpy() @ M).sum(axis=0)


### how to aggregate the individual-level game
def aggregate_individualLevel(M, N, x, mode='mean'):
    """
        M: group membership matrix
        N: for normalization purpose
        x: the parameter to aggregate
    """
    if mode == 'mean':
        ret = N @ M @ x
    elif mode == 'median':
        nG  = M.shape[0]
        ret = np.zeros(nG)
        for k in range(nG):
            idx    = np.nonzero(M[k, :])[0]
            ret[k] = np.quantile(x[idx], q=0.5)
    return ret





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=200)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--n',             type=int,   default=910)
    parser.add_argument('--graph',         type=str,   default='Email')
    parser.add_argument('--elem',          type=int,   default=1)
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--beta_mode',     type=str,   default='gaussian')
    parser.add_argument('--comple',        type=int,   default=1, help='strategic complementarity?')
    parser.add_argument('--proj',          type=int,   default=0, help='whether to do projection')
    parser.add_argument('--traj',          type=int,   default=1, help='whether to collect trajectories')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--beta_var',      type=float, default=0.1)
    parser.add_argument('--beta_var_g',    type=float, default=0.1)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--withinProb',    type=float, default=0.2)
    parser.add_argument('--theta',         type=float, default=0.85, help='a threshold to zero out group-level edges.')
    parser.add_argument('--fPath',         type=str,   default='../result/tmp.txt')
    args  = parser.parse_args()
    DTYPE = torch.float32
    np.random.seed(args.seed)
    UB = 1.0


    ### individual-level game
    G        = gen_graph(args.n, args.graph, ID=args.seed, withinProb=args.withinProb)
    n        = len(G)    
    ### the aggregation specifics
    M, comms = extract_community(G)
    N        = np.diag(1 / M.sum(axis=1))       
    b_var    = args.b_var 
    b_vec    = gen_b(   G, b_var,    M, mode=args.b_mode)                                    ### marginal benefits vector
    beta_var = args.beta_var
    pe_vec   = gen_beta(G, beta_var, M, mode=args.beta_mode, comple=args.comple)             ### peer effects vector
    adj      = np.array(nx.adjacency_matrix(G).todense())                                       ### individual-level adjacency matrix
    ind_game = LQGame(n, b_vec, adj, pe_vec, M=M, N=N, ub=UB)



    ### the group specifics
    nG          = len(N)
    GG          = gen_group_graph(nG, args.theta, M, adj)
    adjG        = np.array(nx.adjacency_matrix(GG).todense())                    ### the adj. matrix of the groups
    bG_vec      = aggregate_individualLevel(M, N, b_vec, mode='mean')
    # beta_var_g  = args.beta_var_g
    # peG_vec     = gen_beta(GG, beta_var_g)                                     ### peer effects vector
    peG_vec     = aggregate_individualLevel(M, N, pe_vec, mode='mean')
    group_game  = LQGame(nG, bG_vec, adjG, peG_vec, ub=UB)



    ### compare y_hat and y_ne
    if args.traj:
        x_ne, L_ne = ind_game.grad_BR(maxIter=args.maxIter, traj=args.traj, proj=args.proj, lr=args.lr)
    else:
        x_ne       = ind_game.grad_BR(maxIter=args.maxIter, traj=args.traj, proj=args.proj)
    y_hat          = N @ M @ x_ne.numpy()
    y_ne           = group_game.grad_BR(traj=False, proj=False) 

    y_hat_quality = group_game.check_quality(torch.tensor(y_hat, dtype=DTYPE))
    y_ne_quality  = group_game.check_quality(y_ne)
    x_ne_quality  = ind_game.check_quality(x_ne)
    dist_g        = np.linalg.norm(y_hat - y_ne.numpy())

    ### check the quality of x_hat
    x_hat         = yStarToXHat(y_ne, M)
    ### randomly generated an action profile and check its regret
    x_random      = np.random.rand(n)
    x_random_quality = ind_game.check_quality(torch.tensor(x_random, dtype=DTYPE))
    y_random      = np.random.rand(nG)
    y_random_quality = group_game.check_quality(torch.tensor(y_random, dtype=DTYPE))
    x_hat_quality = ind_game.check_quality(torch.tensor(x_hat, dtype=DTYPE))
    dist_i        = np.linalg.norm(x_hat - x_ne.numpy())

    # print("y_hat_q: {:.6f} | y_ne_q: {:.6f} | x_ne_q: {:.6f} | dist_g: {:.6f}".format(y_hat_quality, y_ne_quality, x_ne_quality, dist_g))
    #print()
    print("{} {} {} {} {} {} {} {} {} {}".format(args.beta_var, args.withinProb, y_hat_quality, y_ne_quality, x_ne_quality, 
                                              x_hat_quality, x_random_quality, dist_g, dist_i, args.theta))



    # ### compute individual game's within group similarity
    # withinSim = ComputeBetweenGroupSim(M, N, adj, ind_game)
    # print("{:.2f} {:.6f}".format(args.beta_var, withinSim))

    ### the trajectory with x_hat as an initial point
    # y_ne     = group_game.grad_BR(traj=False, proj=False) 
    # x_hat    = yStarToXHat(y_ne, M)
    # _, L_hat = ind_game.grad_BR(x_init=x_hat, maxIter=args.maxIter, traj=True, lr=args.lr)
    # for l_ne, l_hat in zip(L_ne, L_hat):
    #     print("{} {}".format(l_ne, l_hat))


    # ### append to a text file
    # if os.path.exists(args.fPath):
    #     os.remove(args.fPath)
    # with open(args.fPath, 'a') as fid:
    #     ### write x_ne
    #     fid.write(' '.join(list(map(str, x_ne.numpy()))) + '\n')
    #     ### write y_ne
    #     fid.write(' '.join(list(map(str, y_ne.numpy()))) + '\n')
    #     ### write group info
    #     for com in comms:
    #         fid.write(' '.join(list(map(str, com))) + '\n')







