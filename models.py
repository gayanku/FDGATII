import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from lib_diago import * 

class GraphConvolution_G(nn.Module):
    def __init__(self, in_features, out_features, support, mode, heads, residual=False, adj=None):
        super().__init__() 
        self.support = support
        self.in_features = 2*in_features if support == 2 else in_features       
        self.out_features = out_features
        self.residual = residual

        self.mode = mode
        if mode == 'FDGATII': self.spgat_layer = SpGraphAttentionLayer_v2(in_features=in_features, out_features=in_features, dropout=0.6, alpha=0.2, concat=True)
 
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        if self.mode == 'GCNII': 
            hi = torch.sparse.mm(adj, input)
        else: 
            hi = self.spgat_layer(input, adj)
        
        if self.support == 0:
            output = hi
        if self.support == 1:
            support = (1-alpha)*hi+alpha*h0
            r = support
            output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.support == 2:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
            output = theta*torch.mm(support, self.weight)+(1-theta)*r

        if self.residual: output = output+input
        return output

class GCNII_BASE(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, support, mode, heads): 
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution_G(nhidden, nhidden,support, mode, heads))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


if __name__ == '__main__':
    pass






