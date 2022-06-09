import torch
import numpy as np
from abc import ABC, abstractmethod

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import cvxpy as cvx

DTYPE = torch.float32

## abstract class for games
class Game(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def utility(self):
        pass

    @abstractmethod
    def utility_(self):
        pass

    @abstractmethod
    def check_quality(self):
        pass

    @abstractmethod
    def approx_NE(self):
        pass


class LQGame(Game):
    def __init__(self, n, b_vec, adj, beta, M=None, N=None, lb=0.0, ub=1.0):
        # sr = np.abs(np.linalg.eig(np.diag(beta * np.ones(n)) @ adj)[0]).max()
        # ## whether the game has a unique NE
        # self.unique_ne = True if sr < 1.0 else False

        self.lb = lb
        self.ub = ub
            
        self.n    = n
        self.b    = torch.tensor(b_vec, dtype=DTYPE)
        self.adj  = torch.tensor(adj, dtype=DTYPE)
        ## make sure the diagonal items are zeros
        self.adj -= torch.diag(torch.diag(self.adj))
        self.beta = torch.tensor(beta, dtype=DTYPE)

        ## compute Laplacian matrix
        self.laplacian = torch.diag(self.adj.sum(axis=0)) - self.adj

        ## group membership information
        if (M is not None) and (N is not None):
            self.M = torch.tensor(M, dtype=DTYPE)
            self.N = torch.tensor(N, dtype=DTYPE)

        # if self.unique_ne: 
        #     self.x_ne  = self.approx_NE()

        
        self.adjList = {}
        for i in range(self.n):
            adjlist = torch.nonzero(self.adj[i, :]).squeeze().tolist() 
            if type(adjlist) is list:
                self.adjList[i] = adjlist
            else:
                self.adjList[i] = [adjlist]


    ## check the computation of gradients
    def check_gradient(self, x, grad):
        grad_true = self.b - x + torch.diag(self.beta * torch.ones(self.n)) @ ( self.adj @ x )
        dist = torch.dist(torch.abs(grad_true), torch.abs(grad))
        print(f"Gradient difference: {dist:.8f}")


    ## output a vector of utilities
    def utility(self, x):
        u = self.b * x - 0.5 * x * x + torch.diag(self.beta * torch.ones(self.n)) @ x * (self.adj @ x)
        return u

    
    ## output a single player's utility
    def utility_(self, x, i):
        beta = self.beta * torch.ones(self.n)
        return self.b[i] * x[i] - 0.5 * x[i] ** 2 + beta[i] * x[i] * torch.dot(self.adj[i, :], x)


    ## potential function
    def potential(self, x):
        Q = torch.eye(self.n) - torch.diag(self.beta * torch.ones(self.n)) @ self.adj
        poten = torch.dot(self.b, x) - 0.5 * torch.dot(x, Q @ x)
        return poten


    ## approx. NE
    def approx_NE(self):
        beta = self.beta * torch.ones(self.n)
        A    = torch.eye(self.n) - torch.diag(beta) @ self.adj
        x_ne = torch.linalg.solve(A, self.b.reshape(-1, 1))
        # x_ne, _ = torch.solve(self.b.reshape(-1, 1), A)
        x_ne.clamp_(self.lb, self.ub)
        return x_ne.squeeze()


    ## compute gradient map
    def compute_grad(self, xt):
        beta = self.beta * torch.ones(self.n)
        grad = self.b - xt + torch.diag(beta) @ self.adj @ xt
        return grad


    def check_quality(self, x):
        # if self.unique_ne:
        #     return torch.dist(self.x_ne, x)
        # else:
        return self.regret(x).max()


    ## output a vector of regrets
    def regret(self, x):
        r = torch.zeros(self.n)
        for i in range(self.n):
            r[i] = self.regret_(x, i)
        return r   


    ## output a single player's regret
    def regret_(self, x, i):
        x_opt = x.clone()
        x_opt[i] = self.best_response_(x, i)
        rt = self.utility_(x_opt, i) - self.utility_(x, i)    
        assert(rt >= 0.0 or torch.abs(rt) <= 1e-5)
        return rt


    ## output a vector of best responses
    def best_response(self, x):
        x_ = self.b + torch.diag(self.beta * torch.ones(self.n)) @ self.adj @ x
        x_.clamp_(self.lb, self.ub)
        return x_


    ## output a single player's best response
    def best_response_(self, x, i):
        beta = self.beta * torch.ones(self.n)
        x_ = self.b[i] + beta[i] * torch.dot(self.adj[i, :], x)
        x_.clamp_(self.lb, self.ub)
        return x_



    ## Vickrey&Koller, AAAI02
    ###########################################################################
    def compute_G_(self, x, i, nIter=3):
        x_ = x.clone()
        neigh_effect = torch.dot(self.adj[i, :], x_)
        beta = self.beta * torch.ones(self.n)
        r_old = self.regret(x_, lb=float('-inf'), ub=float('inf'))

        ## temporarily used for differentiation
        utility_ = lambda x_i, i, neffect: self.b[i] * x_i - 0.5 * x_i ** 2 + beta[i] * x_i * neffect

        ## compute i's best response
        i_opt = self.b[i] + beta[i] * neigh_effect

        y = Variable(torch.rand_like(x[i]), requires_grad=True)
        optimizer = optim.LBFGS([y], lr=0.01, history_size=30, max_iter=15, line_search_fn="strong_wolfe")

        ## the objective function to minimize
        def f(y):
            x_[i] = y
            r_ = r_old.clone()
            ## i's regret
            r_[i] = utility_(i_opt, i, neigh_effect) - utility_(y, i, neigh_effect) 

            ## i's neighbors regret
            for j in self.adjList[i]:
                n_effect = torch.dot(self.adj[j, :], x_)
                j_opt = self.b[j] + beta[j] * n_effect
                ## j's regret
                r_[j] = utility_(j_opt, j, n_effect) - utility_(x_[j], j, n_effect)
            
            return -(r_old.sum() - r_.sum())
            
        for _ in range(nIter):
            optimizer.zero_grad()
            obj = f(y)
            obj.backward(retain_graph=True)
            # print("obj: ", obj.item())
            optimizer.step(lambda: f(y))

        return y.item(), -f(y).item()


    def regretMIN(self, x, maxIter=1500, lb=0, ub=1):
        ## initialize the G vector
        G  = torch.zeros(self.n)
        x_ = torch.zeros(self.n)
        for i in range(self.n):
            x_[i], G[i] = self.compute_G_(x, i)
            
        x_opt = x.clone()
        Iter = 0
        L = [self.check_quality(x_opt)]
        while True:
            if Iter >= maxIter:
                break

            idx = torch.argmax(G).item()
            if G[idx] <= 0:
                break
            else:
                x_opt[idx] = x_[idx]
                x_opt.clamp_(lb, ub)

                if (Iter+1) % 50 == 0 or Iter == 0:
                    dist = self.check_quality(x_opt.clamp(lb, ub))
                    L.append(dist.item())
                    print(f"Iter: {Iter+1:04d} | Dist: {dist.item():.4f}")
                Iter += 1
                
                ## update
                x_[idx], G[idx] = self.compute_G_(x_opt, idx)
                for j in self.adjList[idx]:
                    x_[j], G[j] = self.compute_G_(x_opt, j)
        
        return x_opt, L

    ###########################################################################
    def aggregate_group(self, x):
        """
            aggregate the individual-level action profile by averaging
        """
        return self.N @ self.M @ x

    def distribute_group(self, y):
        """
            distribute the group-level action profile back to the individual-level game
        """
        return (y[None, :] @ self.M).sum(axis=0)


    def grad_BR(self, maxIter=100, lr=0.01, x_init=None, elementwise=True, optimizer='Adam', traj=False, proj=False):

        if x_init is not None:
            x_init = torch.tensor(x_init, dtype=DTYPE)
            x = nn.Parameter(x_init.clone(), requires_grad=True)
        else:
            x = nn.Parameter(torch.rand(self.n), requires_grad=True)
    
        L = []
        dist = self.check_quality(x.detach())
        L.append(dist.item())
        
        if optimizer == 'Adam':
            optimizer = optim.Adam([x], lr=lr)
        elif optimizer == 'SGD':
            optimizer = optim.SGD([x], lr=lr)
        else:
            raise ValueError("Unknown optimizer in grad_BR")
        # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.999)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

        for Iter in range(maxIter):
            optimizer.zero_grad()
            if elementwise:
                loss = -self.utility(x) 
                grad_ = torch.zeros(self.n)
                for i in range(self.n):
                    x.grad = None
                    loss[i].backward(retain_graph=True)
                    grad_[i] = x.grad[i]
                x.grad = grad_
                # self.check_gradient(x.data, x.grad.data)
            else:
                r = self.regret(x) 
                loss = r.sum()
                loss.backward()

            optimizer.step()
            # scheduler.step()

            ## make sure the action profile is in [0, 1]
            x.data = x.data.clamp(0, 1)

            ## projection with the help of the group-level game
            if proj:
                x.data = self.distribute_group(self.aggregate_group(x.data))

            ## if we need the optimization trajectory
            if traj:
                dist = self.check_quality(x.detach())
                L.append(dist.item())

            if (Iter+1) % 10 == 0 or Iter == 0:
                r = self.check_quality(x.detach())
                print(f"Iter: {Iter+1:04d} | Regret: {r.item():.4f}")

        if traj:
            return (x.detach(), L)
        else:
            return x.detach()

    
    ###########################################################################
    

    # ### the idea of SDP based on merit function
    # ###########################################################################
    # def merit_SDP(self):
    #     K = torch.diag(self.beta * torch.ones(self.n)).numpy() @ self.adj.numpy() - \
    #                         torch.eye(self.n).numpy()
    #     Q_mat = cvx.bmat([[0.5 * (K + K.T), 0.5 * self.b.numpy()[:, None]], [0.5 * self.b.numpy()[:, None].T, np.matrix(0)]])
    #     X_opt = cvx.Variable((self.n+1, self.n+1), symmetric=True)
    #     x_opt = cvx.Variable(self.n, nonneg=True)

    #     cons_1 = K @ x_opt + self.b.numpy()
    #     x_hat = cvx.hstack([x_opt, 1])
    #     cons_2 = cvx.bmat([[X_opt, x_hat[:, None]], [x_hat[:, None].T, np.matrix(1)]])

    #     consts = [cons_1 >= 0, cons_2 >> 0, x_opt <= 5, X_opt <= 25]
    #     obj = cvx.trace(Q_mat @ X_opt)
    #     prob = cvx.Problem(cvx.Minimize(obj), consts)
    #     prob.solve(solver=cvx.SCS, verbose=True)
    #     return x_opt.value
        

