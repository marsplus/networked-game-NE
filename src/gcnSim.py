import os
import dgl
import time
import random
import argparse
import numpy as np
import networkx as nx
import dill as pickle
from itertools import chain

from GCN import *
from Games import LQGame
from utils import gen_graph, extract_community, gen_group_graph, gen_b, gen_beta, random_community


def construct_Q_mat(n, b, beta, A, torch_dtype, torch_device):
    ### the dimension is n+1 because we augment the matrix Q such 
    ### that the linear term disappears.
    
    Q_mat = torch.zeros(n+1, n+1)
    beta_vec = torch.tensor(beta, dtype=torch_dtype, device=torch_device)
    b_vec = torch.tensor(b, dtype=torch_dtype, device=torch_device)
    A_mat = torch.tensor(A, dtype=torch_dtype, device=torch_device)

    I_mat = torch.eye(n, dtype=torch_dtype, device=torch_device)
    P_mat = torch.diag(beta_vec)

    Q_mat[0:n, 0:n] = P_mat @ A_mat - I_mat
    Q_mat[0:n, -1] = 0.5 * b_vec
    Q_mat[-1, 0:n] = 0.5 * b_vec

    Q_mat = Q_mat.type(torch_dtype)
    Q_mat = Q_mat.to(torch_device)

    ### since the VI problem is a maximization prolbem
    return -Q_mat


def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.
    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer



def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.
    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost



# Parent function to run GNN training given input config
def run_gnn_training(q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered

    # initialize optimal solution
    best_bitstring = torch.zeros((len(q_torch),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)

    t_gnn_start = time.time()

    # Training logic
    for epoch in range(number_epochs):

        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model

        # build cost value with QUBO cost function
        loss = loss_func(torch.cat((probs, torch.tensor([1]))), q_torch)
        loss_ = loss.detach().item()


        # bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = probs.detach()


        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time.time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    # final_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, best_bitstring   





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxIter',       type=int,   default=200)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--n',             type=int,   default=910)
    parser.add_argument('--graph',         type=str,   default='SBM')
    parser.add_argument('--b_mode',        type=str,   default='uniform')
    parser.add_argument('--beta_mode',     type=str,   default='homophily')
    parser.add_argument('--b_var',         type=float, default=0.1)
    parser.add_argument('--beta_var',      type=float, default=0.1)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--tol',           type=float, default=1e-4)
    parser.add_argument('--patience',      type=int,   default=100)
    parser.add_argument('--epochs',        type=int,   default=10000)
    parser.add_argument('--nComm',         type=int,   default=10)
    parser.add_argument('--fPath',         type=str,   default='../result/tmp.txt')
    args  = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # seed torch RNG
    random.seed(args.seed)
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32
    PROB_THRESHOLD  = 0.5
    UB = 1


    G = gen_graph(n=args.n, graph=args.graph, seed=args.seed)
    A = nx.adjacency_matrix(G).todense()
    comms = extract_community(G, args.graph, nComm=args.nComm)
    n = len(G)
    print("Finished read input graph.")


    ### define individual-level games
    b_vec     = gen_b(G, var=args.b_var, comms=comms, mode=args.b_mode)
    beta_vec  = gen_beta(G, var=args.beta_var, comms=comms, mode=args.beta_mode)
    indivGame = LQGame(n, b_vec, beta_vec, G, ub=UB)
    print("Finished constructing games.")


    ### parameter for the GCN
    dim_embedding = 10
    hidden_dim    = 10
    graph_dgl  = dgl.from_networkx(nx_graph=G)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)
    Q_mat = construct_Q_mat(n, b_vec, beta_vec, A, TORCH_DTYPE, TORCH_DEVICE)
    

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': 0.1}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.5,
        'number_classes': 1,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs':  args.epochs,
        'tolerance':      args.tol,
        'patience':       args.patience
    }
    gnn_hypers.update(opt_params)


    ### instantiate and train GCN
    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)
    _, epochs, x_gcn = run_gnn_training(
        Q_mat, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])
    regret_gcn = indivGame.regret(x_gcn.numpy())


    ### compute a NE by gradient best response
    x_grad, _ = indivGame.grad_BR(maxIter=args.maxIter, lr=args.lr)
    regret_grad = indivGame.regret(x_grad)

    print(f"regret_gcn: {regret_gcn:.6f}  |  regret_grad: {regret_grad:.6f}")








