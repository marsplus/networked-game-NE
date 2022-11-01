import argparse
import numpy as np
import cvxpy as cvx
import networkx as nx
import dill as pickle
from utils import gen_graph, gen_b, extract_community, gen_beta

np.random.seed(123)

class LQG(object):
    def __init__(self, n, b_vec, pe_vec, adj):
        self.n      = n
        self.b_vec  = b_vec
        self.pe_vec = pe_vec
        self.adj    = adj
        self.P      = np.diag(self.pe_vec)

    def F_map(self, ap):
        F = self.b_vec - ap + self.P @ self.adj @ ap
        return F

    def check_unique(self):
        sr = np.abs(np.linalg.eig(np.diag(self.pe_vec * np.ones(self.n)) @ self.adj)[0]).max()
        print("check NE uniqueness: ", sr)
        return True if sr < 1.0 else False

    ### check the quality of an action profile
    def check_quality(self, ap):
        F_map    = self.b_vec - ap + self.P @ self.adj @ ap
        dot_prod = F_map.T @ ap 
        return np.abs(dot_prod)

    def get_NE(self):
        if not self.check_unique():
            raise ValueError("NE is not unique")

        A = np.identity(self.n) - np.diag(self.pe_vec * np.ones(self.n)) @ self.adj
        x_ne = np.linalg.solve(A, self.b_vec)
        return x_ne



class NetworkDesign(object):
    def __init__(self, ind_game, group_game):
        self.ind_game   = ind_game
        self.group_game = group_game
        self.n  = ind_game.n
        self.nG = group_game.n 


    def findNE(self, budget_ind=1.0, budget_group=1.0, eps=1e-7):
        '''
            Search for x_star and y_star with game parameters fixed.
        '''

        ### the variables associated with the individual-level game
        x_star   = cvx.Variable(self.n)
        lambda_1 = cvx.Variable(self.n)
        b_1      = cvx.Variable(self.n)

        ### the variables associated with the group-level game
        y_star   = cvx.Variable(self.nG)
        lambda_2 = cvx.Variable(self.nG)
        b_2      = cvx.Variable(self.nG)


        constraints = []
        ### NE constraints for y_star
        P_2    = np.diag(self.group_game.pe_vec)
        adj_2  = self.group_game.adj
        I_2    = np.identity(self.nG)
        constraints.extend(
                [
                    (P_2 @ adj_2 - I_2) @ y_star + b_2 + lambda_2 == 0,
                    y_star   >= 0,
                    lambda_2 >= 0
                ]
            )

        ### NE constraints for x_star
        P_1    = np.diag(self.ind_game.pe_vec)
        adj_1  = self.ind_game.adj
        I_1    = np.identity(self.n)
        # b_1    = self.ind_game.b_vec
        constraints.extend(
                [
                    (P_1 @ adj_1 - I_1) @ x_star + b_1 + lambda_1 == 0,
                    x_star   >= 0,
                    lambda_1 >= 0
                ]
            )

        ### budget constraints
        b_1_old = self.ind_game.b_vec
        b_2_old = self.group_game.b_vec
        constraints.extend([cvx.norm(b_1_old-b_1) <= budget_ind,
                            cvx.norm(b_2_old-b_2) <= budget_group])

        ### the objective function
        ### we minimize an upper on -x.T * F(x)
        y_aggr  = N @ M @ x_star
        Q_2     = ((I_2 - P_2 @ adj_2).T + (I_2 - P_2 @ adj_2)) / 2
        Q_2     = Q_2 + np.abs(np.linalg.eig(Q_2)[0].min()) * I_2 +  eps * I_2             ### add a small term to make Q_2 PSD
        Q_1     = ((I_1 - P_1 @ adj_1).T + (I_1 - P_1 @ adj_1)) / 2
        Q_1     = Q_1 + np.abs(np.linalg.eig(Q_1)[0].min()) * I_1 +  eps * I_1             ### add a small term to make Q_2 PSD

        obj     = cvx.quad_form(x_star, Q_1) + \
                  cvx.quad_form(y_star, Q_2) + \
                  cvx.norm(y_star - y_aggr, 2)

        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        prob.solve(solver='MOSEK',verbose=False)
        return (x_star.value, y_star.value, b_1.value, b_2.value)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget_ind',   type=float, default=1.0)
    parser.add_argument('--budget_group', type=float, default=1.0)
    parser.add_argument('--n',            type=int,   default=100)
    parser.add_argument('--graph',        type=str,   default='Email')
    parser.add_argument('--output',       type=int,   default=0)
    args = parser.parse_args()


    ### constructin the individual-level game
    G = gen_graph(args.n, args.graph)
    n = len(G)
    b_vec  = np.random.randn(n)                             ### marginal benefits vector
    pe_vec = np.random.randn(n)                             ### peer effects vector
    adj    = np.array(nx.adjacency_matrix(G).todense())     ### individual-level adjacency matrix
    ind_game = LQG(n, b_vec, pe_vec, adj)              


    ### the aggregation specifics
    M = extract_community(G)                                 ### the group memberships of individual-level players
    N = np.diag(1 / M.sum(axis=1))                           ### for normalization                              


    ### the group specifics
    nG          = len(N)
    GG          = gen_graph(nG, 'Complete')
    adjG        = np.array(nx.adjacency_matrix(GG).todense())              ### the adj. matrix of the groups
    bG_vec      = np.random.randn(nG)
    peG_vec     = N @ M @ pe_vec                                            
    group_game  = LQG(nG, bG_vec, peG_vec, adjG)


    ### optimizing the group-level marginal benefits
    NetDesig = NetworkDesign(ind_game, group_game)
    x_star, y_star, b_1_new, b_2_new = NetDesig.findNE(budget_ind=args.budget_ind,
                                                       budget_group=args.budget_group)
    group_game.b_vec  = b_2_new
    ind_game.b_vec    = b_1_new
    ind_approxRatio   = ind_game.check_quality(x_star)
    group_approxRatio = group_game.check_quality(y_star) 
    dist              = np.linalg.norm(N@M@x_star - y_star)

    ### output
    # print("|| y* - y_hat ||_2: ", dist)
    # print("x_NE quality      : ", ind_approxRatio)
    # print("y_NE quality      : ", group_approxRatio)
    # print("bG_orig           : ", bG_vec)
    # print("bG_new            : ", bG_new)
    print("{} {} {} {} {}".format(args.budget_ind, args.budget_group, dist, ind_approxRatio, group_approxRatio))
    

    if args.output:
        with open('../result/example_output.p', 'wb') as fid:
            pickle.dump([b_vec, bG_vec, b_1_new, b_2_new, x_star, y_star], fid)








