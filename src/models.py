import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nin, nout)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        x = self.gc1(x, adj)
        return x
        # return F.log_softmax(x, dim=1)