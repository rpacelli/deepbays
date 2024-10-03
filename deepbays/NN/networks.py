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
    
class ConvNet:
    def __init__(self, K, mask, L):
        self.K = K
        self.mask = mask #mask
        self.L = L

    def Sequential(self, input_channels, input_size, stride, bias, act_func):
        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "erf":
            act = Erf()
        modules = []
        # First convolutional layer
        first_layer = nn.Conv2d(input_channels, self.K, self.mask, stride = stride, bias=bias)
        init.normal_(first_layer.weight, std=1)
        if bias:
            init.constant_(first_layer.bias, 0)
        modules.extend([Norm(self.mask), first_layer, act])
        # Calculate the output size of the last convolutional layer to determine the number of input features for the fully connected layer
        # Assume no padding 
        conv_output_size = np.sqrt(input_size)
        conv_output_size = (conv_output_size - self.mask) // stride + 1
        # Additional L-1 convolutional layers
        for l in range(self.L - 1):
            layer = nn.Conv2d(self.K, self.K, self.mask,stride = stride, bias=bias) 
            init.normal_(layer.weight, std=1)
            conv_output_size = (conv_output_size - self.mask) // stride + 1
            if bias:
                init.constant_(layer.bias, 0)
            modules.extend([Norm(np.sqrt(conv_output_size)),layer, act])
        # Flatten the tensor before the fully connected layer
        modules.append(nn.Flatten())
        with torch.no_grad(): 
            dummy_net_sequential = nn.Sequential(*modules)
            out = dummy_net_sequential(torch.randn(int(np.sqrt(input_size)),int(np.sqrt(input_size))).unsqueeze(0))
            FC_params = len(out.flatten())    
        print(f"Number of parameters in the fully connected layer: {FC_params * input_channels}")
        # Fully connected layer
        last_layer = nn.Linear(FC_params, 1, bias=bias)
        init.normal_(last_layer.weight, std=1)
        if bias:
            init.constant_(last_layer.bias, 0)
        modules.append(Norm(np.sqrt(FC_params)))
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} convolutional hidden layer(s) with {self.K} kernels of size {self.f} and {act_func} activation function', sequential)
        return sequential

