import os
import time
import numpy as np
import networkx as nx
import dill as pickle

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

torch.set_default_dtype(torch.float32)


class LQGame(object):
    def __init__(self, n, G, b_vec=None, beta=None, lb=0.0, ub=1.0):
        self.lb = lb
        self.ub = ub
            
        self.n    = n
        self.b    = b_vec
        self.beta = beta

        # self.aggregator = lambda x: x

        self.neigh_idx = {
            i: list(G.neighbors(i)) for i in range(self.n)}  


    def neigh_total(self, x, i):
        idx = self.neigh_idx[i]
        return sum(x[idx])


    ## compute individual gradient 
    def grad_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        g_          = self.b[i] - x[i] + self.beta[i] * neigh_total
        return g_


    ## compute total gradient (the objective function is the total regret)
    def total_grad_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        neigh_total_weighted = sum([self.beta[k] * x[k] for k in self.neigh_idx[i]])
        g_          = self.b[i] - x[i] + self.beta[i] * neigh_total + neigh_total_weighted
        return g_


    def utility_(self, xi, ni, i):
        return self.b[i] * xi - 0.5 * np.power(xi, 2) + self.beta[i] * xi * ni

        
    ## output a single player's best response
    def best_response_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        x_opt       = np.clip(self.b[i] + self.beta[i] * neigh_total, self.lb, self.ub)
        return x_opt


    ## output the regret of the current profile
    def regret(self, x):
        reg = float('-inf')
        for i in range(self.n):
            reg = max(self.regret_(x, i), reg)
        return reg


    ## output a single player's regret
    def regret_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        x_best = self.best_response_(x, i)
        rt = self.utility_(x_best, neigh_total, i) - self.utility_(x[i], neigh_total, i)
        assert(rt >= 0.0 or np.abs(rt) <= 1e-5)
        return rt


    ###########################################################################

    def grad_BR(self, maxIter=100, lr=0.01, x_init=None, full_update=True, elementwise=True, mode='sequential'):
        x = x_init.copy() if x_init is not None else np.random.rand(self.n)
        # x = x_init if x_init is not None else np.zeros(self.n)

        L = []
        for Iter in range(maxIter):
            if (Iter + 1) % 10 == 0 or Iter == 0:
                reg  = self.regret(x)
                L.append(reg)
                # print(f"Iter: {Iter+1} | reg: {reg}")    
            
            if full_update:
                idx = range(self.n)
            else:
                idx = np.random.choice(range(self.n), size=int(self.n * 0.1), replace=False)

            x_tmp = x.copy()
            for i in idx:
                ### whether to update elementwise
                if elementwise:
                    grad_ = self.grad_(x, i)
                else:
                    grad_ = self.total_grad_(x, i)
                
                if mode != 'sequential':
                    x_tmp[i] = x[i] + lr * grad_
                    x_tmp[i] = np.clip(x_tmp[i], self.lb, self.ub)
                else:
                    x[i] = x[i] + lr * grad_
                    x[i] = np.clip(x[i], self.lb, self.ub)
            if mode != 'sequential':
                x = x_tmp
        
        return (x, L)


    ###########################################################################
    ### Vickrey & Koller, AAAI02 ###
    def regretMIN(self, x_init=None, maxIter=1500):
        if x_init is None:
            x_init = np.random.rand(self.n)

        b_tensor = torch.tensor(self.b)
        beta_tensor = torch.tensor(self.beta)

        ## initialize the G vector
        G = torch.zeros(self.n)
        x_tmp = torch.zeros(self.n)
        for i in range(self.n):
            x_tmp[i], G[i] = self.compute_G(x_init, i, b_tensor, beta_tensor)
            

        x_opt = torch.tensor(x_init)
        Iter = 0
        L = [self.regret(x_init)]
        while True:
            if Iter >= maxIter:
                break

            idx = torch.argmax(G).item()
            if G[idx] <= 0:
                break
            else:
                x_opt[idx] = x_tmp[idx]

                if (Iter+1) % 10 == 0 or Iter == 0:
                    reg = self.regret(x_opt.numpy())
                    L.append(reg)
                    # print(f"Iter: {Iter+1:04d} | Reg: {reg:.4f}")
                Iter += 1
                
                ## update
                for j in self.neigh_idx[idx] + [idx]:
                    x_tmp[j], G[j] = self.compute_G(x_opt, j, b_tensor, beta_tensor)
        
        return x_opt.numpy(), L


    def compute_G(self, x, i, b_tensor, beta_tensor, nIter=3):
        x_tmp = torch.tensor(x)
        S_old = torch.tensor(sum([self.regret_(x, k) for k in self.neigh_idx[i] + [i]]))

        ## temporarily used for differentiation
        utility_ = lambda x_i, i, ni: \
            b_tensor[i] * x_i - 0.5 * x_i ** 2 + beta_tensor[i] * x_i * ni
        
        y = Variable(torch.rand_like(x_tmp[i]), requires_grad=True)        
        optimizer = optim.LBFGS([y], lr=0.01, history_size=100, max_iter=30, line_search_fn="strong_wolfe")

        ## the objective function to minimize
        def f(y):
            x_tmp[i] = y
            S_new = 0.0

            for j in self.neigh_idx[i] + [i]:
                nj = self.neigh_total(x_tmp, j)
                xj_best = torch.clamp(b_tensor[j] + beta_tensor[j] * nj, min=self.lb, max=self.ub)
                S_new += utility_(xj_best, j, nj) - utility_(x_tmp[j], j, nj)
            return -(S_old - S_new)

        for _ in range(nIter):
            optimizer.zero_grad()
            obj = f(y)
            obj.backward(retain_graph=True)
            optimizer.step(lambda: f(y))
        
        with torch.no_grad():
            y.clamp_(min=self.lb, max=self.ub)
            return y.detach(), -f(y).detach()


###########################################################################


class Bestshot(object):
    def __init__(self, n, G, b_vec=None, beta=None, lb=0.0, ub=1.0):
        self.n = n
        self.c = b_vec 
        self.lb = 0.0
        self.ub = 1.0
        self.neigh_idx = {
            i: list(G.neighbors(i)) for i in range(self.n)
            } 


    def neigh_total(self, x, i):
        return np.prod(1 - x[self.neigh_idx[i]])


    ## compute individual gradient 
    def grad_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        return neigh_total - self.c[i]


    def utility_(self, xi, ni, i):
        return 1 + (xi - 1) * ni - self.c[i] * xi

        
    ## output a single player's best response
    def best_response_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        if neigh_total - self.c[i] >= 0:
            return 1.0
        else:
            return 0.0

    ## the social welfare of an action profile
    def sw(self, x):
        s = 0.
        for i in range(self.n):
            ni = self.neigh_total(x, i)
            s += self.utility_(x[i], ni, i)
        return s


    ## output the regret of the current profile
    def regret(self, x):
        reg = float('-inf')
        for i in range(self.n):
            reg = max(self.regret_(x, i), reg)
        return reg


    ## output a single player's regret
    def regret_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        x_best = self.best_response_(x, i)
        rt = self.utility_(x_best, neigh_total, i) - self.utility_(x[i], neigh_total, i)
        # assert(rt >= 0.0 or np.abs(rt) <= 1e-5)
        return rt


    def grad_BR(self, maxIter=100, lr=0.01, x_init=None, full_update=True, elementwise=True, mode='sequential'):
        x = x_init.copy() if x_init is not None else np.random.rand(self.n)

        L = []
        for Iter in range(maxIter):
            if (Iter + 1) % 10 == 0 or Iter == 0:
                x_tmp = x.copy()
                x_tmp[x_tmp >= 0.5] = 1
                x_tmp[x_tmp < 0.5]  = 0
                reg  = self.regret(x_tmp)
                L.append(reg)
                # print(f"Iter: {Iter+1} | reg: {reg}")    
            
            if full_update:
                idx = range(self.n)
            else:
                idx = np.random.choice(range(self.n), size=int(self.n * 0.1), replace=False)

            x_tmp = x.copy()
            for i in idx:
                ### whether to update elementwise
                if elementwise:
                    grad_ = self.grad_(x, i)
                else:
                    grad_ = self.total_grad_(x, i)
                
                if mode != 'sequential':
                    x_tmp[i] = x[i] + lr * grad_
                    x_tmp[i] = np.clip(x_tmp[i], self.lb, self.ub)
                else:
                    x[i] = x[i] + lr * grad_
                    x[i] = np.clip(x[i], self.lb, self.ub)
            if mode != 'sequential':
                x = x_tmp

        x[x >= 0.5] = 1
        x[x < 0.5]  = 0
        return (x, L)



class BHGame(object):
    def __init__(self, n, G, b_vec=None, beta=None, lb=0.0, ub=1.0):
        self.lb = lb
        self.ub = ub            
        self.n  = n
        self.eps = 1e-5

        self.neigh_idx = {
            i: list(G.neighbors(i)) for i in range(self.n)}  


    def neigh_total(self, x, i):
        idx = self.neigh_idx[i]
        return sum(x[idx])


    ## compute individual gradient 
    def grad_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        num_neighbor = len(self.neigh_idx[i])
        g = -2 * (x[i] - neigh_total / max(num_neighbor, self.eps))
        return g


    def utility_(self, xi, ni, i):
        num_neighbor = len(self.neigh_idx[i])
        return -1 * np.power(xi - ni / max(num_neighbor, self.eps), 2)

        
    ## output a single player's best response
    def best_response_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        num_neighbor = len(self.neigh_idx[i])
        x_opt       = np.clip(neigh_total / max(num_neighbor, self.eps), self.lb, self.ub)
        return x_opt


    ## output the regret of the current profile
    def regret(self, x):
        reg = float('-inf')
        for i in range(self.n):
            reg = max(self.regret_(x, i), reg)
        return reg


    ## output a single player's regret
    def regret_(self, x, i):
        neigh_total = self.neigh_total(x, i)
        x_best = self.best_response_(x, i)
        rt = self.utility_(x_best, neigh_total, i) - self.utility_(x[i], neigh_total, i)
        assert(rt >= 0.0 or np.abs(rt) <= 1e-5)
        return rt


    ###########################################################################

    def grad_BR(self, maxIter=100, lr=0.01, x_init=None, full_update=True, elementwise=True, mode='sequential'):
        x = x_init.copy() if x_init is not None else np.random.rand(self.n)
        # x = x_init if x_init is not None else np.zeros(self.n)

        L = []
        for Iter in range(maxIter):
            if (Iter + 1) % 10 == 0 or Iter == 0:
                reg  = self.regret(x)
                L.append(reg)
                # print(f"Iter: {Iter+1} | reg: {reg}")    
            
            if full_update:
                idx = range(self.n)
            else:
                idx = np.random.choice(range(self.n), size=int(self.n * 0.1), replace=False)

            x_tmp = x.copy()
            for i in idx:
                ### whether to update elementwise
                if elementwise:
                    grad_ = self.grad_(x, i)
                else:
                    grad_ = self.total_grad_(x, i)
                
                if mode != 'sequential':
                    x_tmp[i] = x[i] + lr * grad_
                    x_tmp[i] = np.clip(x_tmp[i], self.lb, self.ub)
                else:
                    x[i] = x[i] + lr * grad_
                    x[i] = np.clip(x[i], self.lb, self.ub)
            if mode != 'sequential':
                x = x_tmp
        
        return (x, L)



