import os, os.path, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

class LangevinOpt(optim.Optimizer):
    def __init__(self, model: nn.Module, lr, temperature, priors):
        defaults = {'lr': lr, 'temperature': temperature}
        param_groups = []
        for layer in model.children():
            if isinstance(layer, (nn.Linear)):
                param_groups.append({'params': layer.parameters()})

        super().__init__(param_groups, defaults)

        for group, lambda_j in zip(self.param_groups, priors):
            group['lambda'] = lambda_j
            group['noise_std'] = (2 * group['lr'] * group['temperature']) ** 0.5

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.add_( -group['lr'] * (param.grad + group['temperature'] * group['lambda'] * param) + torch.randn_like(param) * group['noise_std'] )



    
def test(net, test_data, test_labels, criterion):
        net.eval()
        with torch.no_grad():
                outputs = net(test_data)
                loss = criterion(outputs, test_labels) 
        return loss.item()

def computeNormsAndOverlaps(Snet,Tnet):
    i = len(Snet)-1
    Snorm = torch.linalg.matrix_norm(Snet[i].weight)
    Tnorm = torch.linalg.matrix_norm(Tnet[i].weight)
    overlap = torch.dot(Snet[i].weight.flatten(),Tnet[i].weight.flatten())
    return Snorm, Tnorm, overlap

def train(net, data, labels, criterion, optimizer):
    optimizer.zero_grad()
    loss = criterion(net(data), labels)
    loss.backward()
    optimizer.step()

def regLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)