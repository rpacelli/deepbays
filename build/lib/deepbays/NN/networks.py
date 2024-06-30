import os, os.path,time,argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.erf(x)

class Id(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Norm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, x):
        return x/self.norm

class FCNet: 
    def __init__(self, N0, N1,  L):
        self.N0 = N0 
        self.N1 = N1
        self.L = L
    def Sequential(self, bias, act_func):
        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "erf":
            act = Erf()
        elif act_func == "id":
            act = Id()
        modules = []
        first_layer = nn.Linear(self.N0, self.N1, bias=bias)
        init.normal_(first_layer.weight, std = 1)
        if bias:
            init.constant_(first_layer.bias,0)
        modules.append(Norm(np.sqrt(self.N0)))
        modules.append(first_layer)
        for l in range(self.L-1): 
            modules.append(act)
            modules.append(Norm(np.sqrt(self.N1)))
            layer = nn.Linear(self.N1, self.N1, bias = bias)
            init.normal_(layer.weight, std = 1)
            if bias:
                init.normal_(layer.bias,std = 1)
            modules.append(layer)
        modules.append(act)
        modules.append(Norm(np.sqrt(self.N1)))
        last_layer = nn.Linear(self.N1, 1, bias=bias)  
        init.normal_(last_layer.weight, std = 1) 
        if bias:
                init.normal_(last_layer.bias,std = 1)
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} dense hidden layer(s) of size {self.N1} with {act_func} actviation function', sequential)
        return sequential