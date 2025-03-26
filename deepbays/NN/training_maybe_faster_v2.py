import os, os.path, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

class LangevinOpt(optim.Optimizer):
    def __init__(self, model: nn.Module, lr, temperature):
        defaults = {'lr': lr, 'temperature': temperature}
        super().__init__(model.parameters(), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, T = group['lr'], group['temperature']
            noise_std = (2 * lr * T) ** 0.5
            for param in group['params']:
                if param.grad is not None:
                    param.add_(-lr * (param.grad + T * param) + torch.randn_like(param) * noise_std)

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

def train(net, data, labels, criterion, optimizer, T, priors):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = net(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

def regLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)