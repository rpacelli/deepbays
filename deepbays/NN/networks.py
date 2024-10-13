import os, os.path,time,argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F

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
    
def relu(x):
    return x if x > 0 else 0

class FCNet: 
    def __init__(self, 
                 N0 : int, #input dimension
                 N1 : int, #number of hidden layer units
                 L : int, #number of hidden layers. depth = L+1
                 bias : bool = False, #if True, add bias for each layer
                 act : str = "erf" #activation function
                 ):
        self.N0, self.N1, self.L, self.act, self.bias = N0, N1, L, act, bias 

    def Sequential(self):
        if self.act == "relu":
            act = nn.ReLU()
        elif self.act == "erf":
            act = Erf()
        elif self.act == "id":
            act = Id()
        modules = []
        first_layer = nn.Linear(self.N0, self.N1, bias=self.bias)
        init.normal_(first_layer.weight, std = 1)
        if self.bias:
            init.constant_(first_layer.bias,0)
        modules.append(Norm(np.sqrt(self.N0)))
        modules.append(first_layer)
        for l in range(self.L-1): 
            modules.append(act)
            modules.append(Norm(np.sqrt(self.N1)))
            layer = nn.Linear(self.N1, self.N1, bias = self.bias)
            init.normal_(layer.weight, std = 1)
            if self.bias:
                init.normal_(layer.bias,std = 1)
            modules.append(layer)
        modules.append(act)
        modules.append(Norm(np.sqrt(self.N1)))
        last_layer = nn.Linear(self.N1, 1, bias=self.bias)  
        init.normal_(last_layer.weight, std = 1) 
        if self.bias:
                init.normal_(last_layer.bias,std = 1)
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} dense hidden layer(s) of size {self.N1} with {self.act} actviation function', sequential)
        return sequential
    
class ConvNet:
    def __init__(self, N0 : int,  #input size, necessary to compute normalization
                 Nc : int,  # number of channels in internal layers
                 mask : int, #mask
                 stride : int, #striding of convolution
                 bias : bool = False, #if True, add bias for each layer
                 act :str = "erf", #activation function
                 inputChannels : int = 1 #number of inputs channel, defaults for black and white image
                 ):
        self.Nc, self.N0, self.inputChannels, self.mask, self.stride, self.bias, self.act = Nc, N0, inputChannels, mask, stride, bias, act

    def Sequential(self):
        if self.act == "relu":
            act = nn.ReLU()
        elif self.act == "erf":
            act = Erf()
        modules = []
        # First convolutional layer
        first_layer = nn.Conv2d(self.inputChannels, self.Nc, self.mask, stride = self.stride, bias=self.bias)
        init.normal_(first_layer.weight, std=1)
        if self.bias:
            init.constant_(first_layer.bias, 0)
        modules.extend([Norm(self.mask), first_layer, act])
        # Flatten the tensor before the fully connected layer
        modules.append(nn.Flatten())
        with torch.no_grad(): 
            dummy_net_sequential = nn.Sequential(*modules)
            out = dummy_net_sequential(torch.randn(self.inputChannels,int(np.sqrt(self.N0)),int(np.sqrt(self.N0))).unsqueeze(0))
            FC_params = len(out.flatten())  # number of parameters in the last layer   
        # Fully connected layer
        last_layer = nn.Linear(FC_params, 1, bias=self.bias)
        init.normal_(last_layer.weight, std=1)
        if self.bias:
            init.constant_(last_layer.bias, 0)
        modules.append(Norm(np.sqrt(FC_params)))
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has 1 convolutional hidden layer with {self.Nc} kernels of size {self.mask} and {self.act} activation function', sequential)
        return sequential