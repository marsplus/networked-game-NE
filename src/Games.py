import torch
import numpy as np
from abc import ABC, abstractmethod

import torch.optim as optim
import torch.nn as nn

import cvxpy as cvx
from utils import euclidean_proj_l2

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
    def __init__(self, n, b_vec, adj, beta, C, proj_tol=0.5):
        sr = np.abs(np.linalg.eig(np.diag(beta * np.ones(n)) @ adj)[0]).max()
        assert(sr < 1.0)
        self.n   = n
        self.b   = torch.tensor(b_vec, dtype=DTYPE)
        self.adj = torch.tensor(adj, dtype=DTYPE)
        self.beta= torch.tensor(beta, dtype=DTYPE)
        self.x_ne= self.approx_NE()

        self.C = C
        self.y_at_ne = self.C @ self.x_ne.numpy()
        self.proj_tol = proj_tol

        self.adjList = {
            i: torch.nonzero(self.adj[i, :]).squeeze().tolist() for i in range(self.n)
        }

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
        A = torch.eye(self.n) - torch.diag(beta) @ self.adj
        x_ne, _ = torch.solve(self.b.reshape(-1, 1), A)
        return x_ne.squeeze()


    def check_quality(self, x):
        dist = torch.dist(self.x_ne, x)
        return dist


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
        return self.utility_(x_opt, i) - self.utility_(x, i)    


    ## output a vector of best responses
    def best_response(self, x):
        x_ = self.b + torch.diag(self.beta * torch.ones(self.n)) @ self.adj @ x
        return x_


    ## output a single player's best response
    def best_response_(self, x, i):
        beta = self.beta * torch.ones(self.n)
        x_ = self.b[i] + beta[i] * torch.dot(self.adj[i, :], x)
        return x_



    ## Vickrey&Koller, AAAI02
    ###########################################################################
    def compute_G_(self, x, i, nIter=3):
        from torch.autograd import Variable
        x_ = x.clone()
        neigh_effect = torch.dot(self.adj[i, :], x_)
        beta = self.beta * torch.ones(self.n)
        r_orig = self.regret(x_)
        ## temporarily used for differentiation
        utility_ = lambda x, i, neffect: self.b[i] * x - 0.5 * x ** 2 + beta[i] * x * neffect

        ## compute i's best response
        x_opt = self.b[i] + beta[i] * neigh_effect

        y = Variable(torch.rand_like(x[i]), requires_grad=True)
        optimizer = optim.LBFGS([y], lr=0.01, history_size=30, max_iter=15, line_search_fn="strong_wolfe")

        ## the objective function to minimize
        def f(y):
            r_ = r_orig.clone()
            ## i's regret
            r_[i] = utility_(x_opt, i, neigh_effect) - utility_(y, i, neigh_effect) 

            ## i's neighbors regret
            x_tmp = x.clone()
            x_tmp[i] = y
            for j in self.adjList[i]:
                n_effect = torch.dot(self.adj[j, :], x_tmp)
                y_opt = self.b[j] + beta[j] * n_effect
                ## j's regret
                r_[j] = utility_(y_opt, j, n_effect) - utility_(x_tmp[j], j, n_effect)
            
            return -(r_orig.sum() - r_.sum())
            
        for _ in range(nIter):
            optimizer.zero_grad()
            obj = f(y)
            obj.backward()
            # print("obj: ", obj.item())
            optimizer.step(lambda: f(y))

        # y.detach().clamp_(min=0, max=float('inf'))
        return y.item(), -f(y).item()


    def regretMIN(self, x, maxIter=1500):
        ## initialize the G vector
        G  = torch.zeros(self.n)
        x_ = torch.zeros(self.n)
        for i in range(self.n):
            x_[i], G[i] = self.compute_G_(x, i)
            
        x_opt = x.clone()
        Iter = 0
        L = []
        while True:
            if Iter >= maxIter:
                break

            idx = torch.argmax(G).item()
            if G[idx] <= 0:
                break
            else:
                Iter += 1
                x_opt[idx] = x_[idx]
                dist = torch.dist(x_opt, self.x_ne)
                L.append(dist.item())
                print(f"Iter: {Iter:04d} | Dist: {dist.item():.4f}")
                
                ## update
                x_[idx], G[idx] = self.compute_G_(x_opt, idx)
                for j in self.adjList[idx]:
                    x_[j], G[j] = self.compute_G_(x_opt, j)
        return x_opt, L

    
    ## structural projection
    ## given x^t at step t, this is to solve:
    ## min_{z} || x^t - z ||_2
    ##   s.t.   ||Cz - y||_2 \le epsilon
    def structure_proj_(self, xt):
        z = cvx.Variable(self.n)
        obj = cvx.norm(z - xt.detach().numpy(), 2)
        const = [cvx.norm(self.C @ z - self.y_at_ne, 2) <= self.proj_tol]
        prob = cvx.Problem(cvx.Minimize(obj), const)
        prob.solve()
        return torch.tensor(z.value, dtype=DTYPE)


    ###########################################################################

    def grad_BR(self, maxIter=200, lr=0.01, x_init=None, elementwise=True, optimizer='Adam', projection=False):
        if x_init != None:
            x = nn.Parameter(x_init.clone(), requires_grad=True)
        else:
            x = nn.Parameter(torch.rand(self.n), requires_grad=True)
        
        if optimizer == 'Adam':
            optimizer = optim.Adam([x], lr=lr)
        elif optimizer == 'SGD':
            optimizer = optim.SGD([x], lr=lr)
        else:
            raise ValueError("Unknown optimizer in grad_BR")
        # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.99)

        L = []
        for epoch in range(maxIter):
            optimizer.zero_grad()
            if elementwise:
                loss = -self.utility(x)
                grad_ = torch.zeros(self.n)
                for i in range(self.n):
                    x.grad = None
                    loss[i].backward(retain_graph=True)
                    grad_[i] = x.grad[i]
                x.grad = grad_
            else:
                r = self.regret(x)
                loss = r.sum()
                loss.backward()

            optimizer.step()
            ## projection step
            if projection:
                x.data = self.structure_proj_(x)

            # torch.clamp(x, min=0, max=float('inf'))
            dist = self.check_quality(x.detach())
            L.append(dist.item())
            # print(f"Epoch: {epoch:04d} | Dist: {dist.item():.4f}")
        return x.detach(), L

    


