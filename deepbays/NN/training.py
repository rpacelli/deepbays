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

        assert len(priors) == len(self.param_groups), "Lenght mismatch"

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad = param.grad.detach()
                    noise = torch.randn_like(param, memory_format=torch.preserve_format) * group['noise_std']
                    param.add_(-group['lr'] * (grad + group['temperature'] * group['lambda'] * param) + noise)



    
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
    loss.backward(retain_graph=False)
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

def regLoss(output, target, net, T, priors):
    loss, idx = 0, 0
    for i in range(1,len(net)):
        # if (i - 1) % 3 == 0:
        if i % 3 == 0:  # CK: adapted to different position of Linear layers due to changed Norm() position
            loss += (0.5*priors[idx]*T) * (torch.linalg.matrix_norm(net[i].weight)**2)
            idx += 1
    loss += 0.5 * torch.sum((output - target)**2)
    return loss

def train_AI(net, data, labels, optimizer, T, priors): # note: no criterion argument
    net.train()
    optimizer.zero_grad()
    dataloss = nonregLoss(net(data), labels)  # no reg in the graph
    dataloss.backward()
    for idx, p in enumerate(net.parameters()):
        p.grad.add_(p, alpha=priors[idx]*T)    # weight decay, adding priors[idx] * T * param to grad on param
    optimizer.step()
    return dataloss.detach().item()


def nonregLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)
    