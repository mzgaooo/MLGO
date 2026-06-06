from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchdiffeq
import numbers
import math


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x.shape = (batch, dim, nodes, seq_len)
        # A.shape = (node, node)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class wconv(nn.Module):
    def __init__(self):
        super(wconv, self).__init__()

    def forward(self, x, W):
        # x.shape = (batch, dim, nodes, seq_len)
        # w.shape = (dim, dim)
        x = torch.einsum('ncwl,vc->nvwl', (x, W))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        # s1,t1 = adj.topk(self.k,1)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)  # bug fixed
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class SGPFunc(nn.Module):
    def __init__(self, c_hidden, alpha, device):
        super(SGPFunc, self).__init__()
        self.c_hidden = c_hidden
        self.alpha = alpha
        self.adj = None
        self.device = device
        self.nconv = nconv()
        self.out = []
        

    def forward(self, t, x):
        adj = self.adj.to(self.device) + torch.eye(self.adj.size(0)).to(self.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)

        self.out.append(x)
        ax = self.nconv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x) 

        return f


class SGPODEBlock(nn.Module):
    def __init__(self, sgpfunc, method, step_size, rtol, atol, estimated_nfe, dropout):
        super(SGPODEBlock, self).__init__()
        self.odefunc = sgpfunc
        self.method = method
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol
        self.dropout = nn.Dropout(p=dropout)
        self.mlp =linear((estimated_nfe + 1) * self.odefunc.c_hidden, self.odefunc.c_hidden)

    def set_adj(self, adj):
        self.odefunc.adj = adj

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                 method=self.method, options=dict(step_size=self.step_size))
        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        h_out = self.mlp(torch.cat(outs,dim=1))
        return self.dropout(h_out)


class SGP(nn.Module):
    def __init__(self, c_hidden_channels, init_alpha=2.0, method='rk4', time=1.0, 
                 step_size=0.5, device='cuda:0', rtol=1e-4, atol=1e-3, dropout=0.3):
        super(SGP, self).__init__()
        self.c_hidden = c_hidden_channels     
        self.alpha = init_alpha
        self.device = device
        self.dropout = dropout
        self.integration_time = time
        if method == 'euler':
            self.estimated_nfe = round(self.integration_time / step_size)
        elif method == 'rk4':
            self.estimated_nfe = round(self.integration_time / (step_size / 4.0))

        self.SGPODE = SGPODEBlock(SGPFunc(self.c_hidden, self.alpha, self.device), 
                                  method, step_size, rtol, atol, self.estimated_nfe, self.dropout)

    def forward(self, x, adj):
        self.SGPODE.set_adj(adj)
        h = self.SGPODE(x, self.integration_time)
        return h


class TimeConv(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_hidden_layers, dropout):
        super(TimeConv, self).__init__()

        model = [nn.Linear(input_channels, hidden_channels)]
        model += [nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers - 2):
            model += [nn.Linear(hidden_channels, hidden_channels)]
            model += [nn.ReLU(), nn.Dropout(dropout)]
        model += [nn.Linear(hidden_channels, input_channels)]

        self.model = nn.Sequential(*model)

    def forward(self, inps):
        return self.model(inps)

