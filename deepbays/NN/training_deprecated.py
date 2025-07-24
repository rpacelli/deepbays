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
        groups = []
        for i, layer in enumerate(model.children()):
            groups.append({'params': layer.parameters(), 
                           'lr': lr, 
                           'temperature': temperature})
        super(LangevinOpt, self).__init__(groups, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            learning_rate = group['lr']
            temperature = group['temperature']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                d_p = torch.randn_like(parameter) * (2*learning_rate*temperature)**0.5
                d_p.add_(parameter.grad, alpha=-learning_rate)
                parameter.add_(d_p)

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
    loss = criterion(outputs.float(), targets.float(),net,T,priors)
    loss.backward()
    optimizer.step()
    return loss.item() 


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
    print("Warning, contains bug where weight decay does not affect first layer. Use modregLoss or train_AI() instead")
    loss, idx = 0, 0
    for i in range(1,len(net)):
        # if (i - 1) % 3 == 0:
        if i % 3 == 0:  # CK: adapted to different position of Linear layers due to changed Norm() position
            loss += (0.5*priors[idx]*T) * (torch.linalg.matrix_norm(net[i].weight)**2)
            idx += 1
    loss += 0.5 * torch.sum((output - target)**2)
    return loss

def modregLoss(output, target, net, T, priors):
    loss, idx = 0, 0
    for module in net:
        # if (i - 1) % 3 == 0:
        if isinstance(module, nn.Linear):
        # if i % 3 == 0:  # CK: adapted to different position of Linear layers due to changed Norm() position
            loss += (0.5*priors[idx]*T) * (torch.linalg.matrix_norm(module.weight)**2)
            idx += 1  # counter over weight layers
    loss += 0.5 * torch.sum((output - target)**2)
    return loss

def train_AI(net, data, labels, optimizer, T, priors): # note: no criterion argument
    net.train()
    optimizer.zero_grad()
    dataloss = nonregLoss(net(data), labels)  # no reg in the graph
    dataloss.backward()
    with torch.no_grad():
        for idx, p in enumerate(net.parameters()):
            p.grad.add_(p, alpha=priors[idx]*T)    # weight decay, adding priors[idx] * T * param to grad on param
    optimizer.step()
    return dataloss.detach().item()


def nonregLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)
    