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
        defaults = {
            'lr': lr, 
            'temperature': temperature
        }
        self.lr = lr
        self.T = temperature
        groups = []
        for i, layer in enumerate(model.children()):
            groups.append({'params': layer.parameters(), 
                           'lr': lr, 
                           'temperature': temperature})
        super(LangevinOpt, self).__init__(groups, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group['params']:
                parameter += - self.lr*parameter.grad -self.lr*self.T*parameter + torch.randn_like(parameter) * (2*self.lr*self.T)**0.5

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
    inputs, targets = data, labels#).unsqueeze(1)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs.float(), targets.float())
    loss.backward()
    optimizer.step()

#def regLoss(output, target,net,T,lambda0,lambda1):
#    loss = 0
#    for i in range(len(net)):
#        if i == len(net)-1:
#            loss += (0.5*lambda1*T)*(torch.linalg.matrix_norm(net[i].weight)**2)
#        else:
#            if (isinstance(net[i],nn.Linear) or isinstance(net[i],nn.Conv2d)):
#                loss += (0.5*lambda0*T)*(torch.linalg.matrix_norm(net[i].weight)**2)
#    loss += 0.5*torch.sum((output - target)**2)
#    return loss

def regLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)