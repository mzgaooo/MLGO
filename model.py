from layer import ISIP, linear, graph_constructor, TimeConv
import torchdiffeq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class ODE_Block(nn.Module):
    def __init__(self, odefunc, method, step_size, rtol, atol):
        super(ODE_Block, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                 method=self.method, options=dict(step_size=self.step_size))
        self.odefunc.order = 0
        return out[-1]

class MLGO_Block(nn.Module):

    def __init__(self, hidden_channels, num_hidden_layers, device, method_2, time_2,
                 step_size_2, seq_length, alpha, rtol, atol):
        super(MLGO_Block, self).__init__()
        self.graph1 = None
        self.graph2 = None
        self.graph3 = None
        self.order = 0
        self.device = device

        self.mlp1 = linear(2*hidden_channels, hidden_channels)
        self.mlp2 = linear(2*hidden_channels, hidden_channels)
        self.mlp3 = linear(2*hidden_channels, hidden_channels)
        self.tconv = TimeConv(input_channels=seq_length, hidden_channels=hidden_channels, 
                              num_hidden_layers=num_hidden_layers)
        self.gconv_1 = ISIP(c_hidden_channels=hidden_channels, init_alpha=alpha, method=method_2, time=time_2, 
                            step_size=step_size_2, device=device, rtol=rtol, atol=atol)
        self.gconv_2 = ISIP(c_hidden_channels=hidden_channels, init_alpha=alpha, method=method_2, time=time_2, 
                            step_size=step_size_2, device=device, rtol=rtol, atol=atol)
        self.gconv_3 = ISIP(c_hidden_channels=hidden_channels, init_alpha=alpha, method=method_2, time=time_2, 
                            step_size=step_size_2, device=device, rtol=rtol, atol=atol)
        self.gconv_4 = ISIP(c_hidden_channels=hidden_channels, init_alpha=alpha, method=method_2, time=time_2, 
                            step_size=step_size_2, device=device, rtol=rtol, atol=atol)
        
        
    def forward(self, t, x):
        x[0] = F.relu(self.mlp1(torch.cat((x[2], x[0]), dim=1)))

        x0 = self.tconv(x[0])
        x[0] = self.gconv_1(x0, self.graph1[self.order])
        self.order += 1

        x1 = F.relu(self.mlp2(torch.cat((x[0], x[1]), dim=1)))
        x[1] = self.gconv_2(x1, self.graph2)

        x2 = F.relu(self.mlp3(torch.cat((x[1], x[2]), dim=1)))
        x[2] = self.gconv_3(x2, self.graph3) + self.gconv_4(x2, self.graph3.transpose(1, 0))

        return x

    def setGraph(self, graph1, graph2, graph3):
        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3

class MLGO(nn.Module):
    def __init__(self, num_nodes, device, predefined_A=None, static_feat=None, num_hidden_layers=3,
                 subgraph_size=20, node_dim=40, conv_channels=64, end_channels=64, seq_length=12, 
                 in_dim=1, tanhalpha=3, method_1='euler', time_1=1.0, step_size_1=0.25, 
                 method_2='euler', time_2=1.0, step_size_2=0.5, alpha=1.0, rtol=1e-4, atol=1e-3):
        super(MLGO, self).__init__()

        self.integration_time = time_1
        self.predefined_A = predefined_A
        self.seq_length = seq_length

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.idx = torch.arange(num_nodes).to(device)

        self.affine_weight = nn.Parameter(torch.Tensor(*(conv_channels, num_nodes)))  
        self.affine_bias = nn.Parameter(torch.Tensor(*(conv_channels, num_nodes)))  

        self.ODE = ODE_Block(MLGO_Block(hidden_channels=conv_channels, num_hidden_layers=num_hidden_layers, 
                                        device=device, method_2=method_2, time_2=time_2, step_size_2=step_size_2,
                                        seq_length=seq_length, alpha=alpha, rtol=rtol, atol=atol),
                            method_1, step_size_1, rtol, atol)

        self.end_conv_0 = nn.Conv2d(in_channels=conv_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_1 = nn.Conv2d(in_channels=end_channels, out_channels=end_channels//2, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels//2, out_channels=1, kernel_size=(1, 1), bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.affine_weight)
        init.zeros_(self.affine_bias)
        
    def forward(self, input, idx=None):
        seq_len = input.size(3)  #input：B*dim1*n*window
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        x = self.start_conv(input) #B*dim64*n*window

        adp1 = torch.from_numpy(self.predefined_A[0].astype(np.float32))  #dtw
        adp2 = torch.from_numpy(self.predefined_A[1].astype(np.float32))  #spatial
        if idx is None:
                adp3 = self.gc(self.idx)
        else:
                adp3 = self.gc(idx)

        self.ODE.odefunc.setGraph(adp1,adp2,adp3)

        x = torch.stack([x,x,x])
        x = self.ODE(x, self.integration_time)
        x = x[2]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        if idx is None:
            x = torch.add(torch.mul(x, self.affine_weight[:, self.idx].unsqueeze(-1)), self.affine_bias[:, self.idx].unsqueeze(-1))  # C*H
        else:
            x = torch.add(torch.mul(x, self.affine_weight[:, idx].unsqueeze(-1)), self.affine_bias[:, idx].unsqueeze(-1))  # C*H
        
        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x   
