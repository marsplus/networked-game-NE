"""
    Linear Quadratic Games
"""
import os
import time
import numpy as np
import networkx as nx

import torch
import torch.optim as optim
from typing import Tuple

DEVICE = 'cpu'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


## an abstract class for games
class NetworkGame:
    n: int
    G: nx.Graph
    lb: float = 0.0
    ub: float = 1.0

    def _neigh_total(self):
        pass

    def _utility(self, *args) -> float:
        pass

    def _best_response(self, x: np.array, i: int) -> float:
        pass

    def _profile_regret(self, x: np.array) -> float:
        pass

    def _player_regret(self, x: np.array, i: int) -> float:
        pass



class LQGame(NetworkGame):
    def __init__(self, 
                 n: int, 
                 G: nx.Graph, 
                 b_vec: np.array = None,
                 beta_vec: np.array = None,
                 lb=0.0, 
                 ub=1.0):
        """
            Initialize a Linear Quadratic Game
            Parameters:
                n (int): the number of nodes/players
                G (nx.Graph): a networkx Graph objece representing the underlying network
                b_vec (default None): the marginal benefits 
                beta_vec (default None): the peer effects vector 
                lb/ub (float): the lower/upper bounds on the players' actions
        """
        self.lb = lb
        self.ub = ub
        self.n = n
        self.b_vec = b_vec
        self.beta_vec = beta_vec

        ## index to the neighbors
        self.neigh_idx = {
            i: list(G.neighbors(i)) for i in range(self.n)}  

    def _neigh_total(self, 
                     x: np.array, 
                     i: int) -> int:
        """
            Get the number of neighbors for player i
            Parameters:
                x (np.array): the current action profile for all players
                i (int): the index of the player
            
            Returns:
                the number of neighbors of i
        """
        return x[self.neigh_idx[i]].sum()

    def _grad(self, 
              x: np.array, 
              i: int) -> np.array:
        """
            Compute the gradient of player i's utility w.r.t. her own action
            NOTE: this computation is specific to the LQ games.
        """
        neigh_inv = self._neigh_total(x, i)
        grad = self.b_vec[i] - x[i] + self.beta_vec[i] * neigh_inv 
        return grad

    def _grad_total_regret(self, 
                           x: np.array, 
                           i: int) -> float:
        """
            Update a player's action to minimize the regret of the current action profile
        """
        neigh_inv = self._neigh_total(x, i)
        weighted_sum = sum([self.beta_vec[k] * x[k] for k in self.neigh_idx[i]])
        grad = self.b_vec[i] - x[i] + self.beta_vec[i] * neigh_inv + weighted_sum
        return grad

    def _utility(self, xi: float, ni: float, i: int) -> float:
        """
            The utility for a LQG player
        """
        return self.b_vec[i] * xi - 0.5 * xi**2 + self.beta_vec[i] * xi * ni
    
    def _best_response(self, x: np.array, i: int) -> float:
        """
            Get a player's best response
        """
        neigh_inv = self._neigh_total(x, i)
        x_best = np.clip(self.b_vec[i] + self.beta_vec[i] * neigh_inv, self.lb, self.ub)
        return x_best

    def _profile_regret(self, x: np.array) -> float:
        """
            Get the regret for the current action profile x, i.e., 
            how much better it could be if another action profile was in place.
        """
        rgt = 1e-9
        for i in range(self.n):
            rgt = max(self._player_regret(x, i), rgt)
        return rgt

    def _player_regret(self, x: np.array, i: int) -> float:
        """
            Get a player's regret based on the current action profile x, i.e.,
            how much better the player would be if she deviated from the current action x_i
        """
        neigh_inv = self._neigh_total(x, i)
        x_best = self._best_response(x, i)
        rgt = self._utility(x_best, neigh_inv, i) - self._utility(x[i], neigh_inv, i)
        assert abs(rgt) >= 0.0, "LQG: regret should be >= 0.\n"
        return rgt


    ###########################################################################
    def grad_BR(self, 
                maxIter: int = 100, 
                lr: float = 0.01, 
                x_init: np.array = None, 
                all_players_update: bool = True, 
                cand_ratio: float = 0.1,
                elementwise_update: bool = True, 
                update_mode: str = 'sequential',
                verbose: bool = False) -> Tuple[np.array, list]:
        """
            Perform gradient-based updates to current action profile.
            The updates can be either: 1) sequential, where a player's updates is immediatedly available to the subsequent players or 
                                       2) simultaneous, where the players' updates are available all at once.
            Parameters:
                maxIter (int): max number of iterations
                lr (float): the step size of the updates
                x_init (np.array): initial action profile
                all_players_update (bool): whether all players get the chance to udpate their actions, or just a subset of players can do
                cand_ratio (float): if just a subset of players can update, use this as a ratio to randomly sample the subset.
                elementwise_update (bool): elementwise_update = True: a player updates her own action to maximize her utility
                                           elementwise_update = False: a player updates her own action to minimize the regret of the current action profile
                update_mode (str): either `sequential` or `simultaneous`.
                verbose (bool): whether to print results.
                output_regret (bool): whether to return the regret through the updates.
            Returns:
                x (np.array): the updated action profile
                r (list): the list of regrets
        """
        x = x_init.copy() if x_init is not None else np.random.uniform(self.lb, self.ub, self.n)
        r = []
        for it in range(maxIter):
            rgt = self._profile_regret(x)
            r.append(rgt)
            if verbose:
                print(f"Iter: {it+1} | reg: {rgt}")    
            cand_idx = range(self.n) if all_players_update else \
                            np.random.choice(range(self.n), size=int(self.n * cand_ratio), replace=False)
            x_tmp = x.copy()
            for i in cand_idx:
                grad = self._grad(x, i) if elementwise_update else self._grad_total_regret(x, i)            
                if update_mode != 'sequential':
                    ## the updates will be effective all at once
                    x_tmp[i] = x[i] + lr * grad
                    x_tmp[i] = np.clip(x_tmp[i], self.lb, self.ub)
                else:
                    x[i] = x[i] + lr * grad
                    x[i] = np.clip(x[i], self.lb, self.ub)
            if update_mode != 'sequential':
                x = x_tmp
        return (x, r)


    def min_regret_VK(self, 
                    x_init: np.array = None, 
                    maxIter: int = 1500,
                    verbose=True) -> Tuple[np.array, list]:
            """
                Implement the regret minimization algorithm in Vickrey & Koller, AAAI02.
            """
            if x_init is None:
                x_init = np.random.uniform(self.lb, self.ub, self.n)
            b_tensor = torch.tensor(self.b_vec, device=DEVICE)
            beta_tensor = torch.tensor(self.beta_vec, device=DEVICE)
            ## initialize the G vector defined in the paper
            G = torch.zeros(self.n, device=DEVICE)
            x_tmp = torch.zeros(self.n, device=DEVICE)
            for i in range(self.n):
                x_tmp[i], G[i] = self._compute_G(x_init, i, b_tensor, beta_tensor)

            x_opt = torch.tensor(x_init, device=DEVICE)
            it = 0
            r = [self._profile_regret(x_init)]
            while True:
                if it >= maxIter:
                    break
                idx = torch.argmax(G).item()
                if G[idx] <= 0:
                    break
                else:
                    x_opt[idx] = x_tmp[idx]
                    rgt = self._profile_regret(x_opt.numpy())
                    r.append(rgt)
                    if verbose:
                        print(f"Iter: {it+1:04d} | Reg: {rgt:.4f}")
                    it += 1
                    ## update
                    cand_idx = self.neigh_idx[idx] + [idx]
                    for j in cand_idx:
                        x_tmp[j], G[j] = self._compute_G(x_opt, j, b_tensor, beta_tensor)
            return (x_opt.numpy(), r)


    def _compute_G(self, 
                    x: np.array, 
                    i: int, 
                    b_tensor: torch.Tensor, 
                    beta_tensor: torch.Tensor, 
                    nIter: int = 3,
                    lr: float = 0.01,
                    history_size: int = 100,
                    max_iter: int = 30):
        x_tmp = torch.tensor(x, device=DEVICE)
        cand_idx = self.neigh_idx[i] + [i]
        S_old = torch.tensor(sum([self._player_regret(x, k) for k in cand_idx]), device=DEVICE)
        ## temporarily used for differentiation purposes
        utility_ = lambda x_i, i, ni: \
            b_tensor[i] * x_i - 0.5 * x_i ** 2 + beta_tensor[i] * x_i * ni
        
        y = torch.rand_like(x_tmp[i], requires_grad=True)        
        optimizer = optim.LBFGS([y], lr=lr, history_size=history_size, max_iter=max_iter, line_search_fn="strong_wolfe")
        ## the objective function to minimize
        def f(y):
            x_tmp[i] = y
            S_new = 0.0
            cand_idx = self.neigh_idx[i] + [i]
            for j in cand_idx:
                neighbor_inv = self._neigh_total(x_tmp, j)
                ## the best response of player j
                xj_best = torch.clamp(b_tensor[j] + beta_tensor[j] * neighbor_inv, min=self.lb, max=self.ub)
                ## how better xj_best can bring
                S_new += utility_(xj_best, j, neighbor_inv) - utility_(x_tmp[j], j, neighbor_inv)
            return -(S_old - S_new)

        for _ in range(nIter):
            optimizer.zero_grad()
            obj = f(y)
            obj.backward(retain_graph=True)
            ## pass a closure f(y) into the optimizer for using LBFGS
            optimizer.step(lambda: f(y))
        
        with torch.no_grad():
            y.clamp_(min=self.lb, max=self.ub)
            return y.detach(), -f(y).detach()

    

###########################################################################


if __name__ == "__main__":
    n = 100
    G = nx.watts_strogatz_graph(n, 2, 0.1, seed=seed)
    b_vec = np.random.rand(n)
    beta_vec = np.random.rand(n)
    game = LQGame(n, 
                  G, 
                  b_vec=b_vec,
                  beta_vec=beta_vec)
    
    ## sequential gradient best responses
    game.grad_BR(maxIter=500,
                 verbose=True)
    
    ## VK algorithm
    game.min_regret_VK()