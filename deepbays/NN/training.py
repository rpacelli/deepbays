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

def multitaskTrain(Snet, Tnet, source_data, target_data, source_labels, target_labels, criterion, Soptimizer, Toptimizer, args):
    Snet.train()
    Tnet.train()
    Soptimizer.zero_grad()
    Toptimizer.zero_grad()
    source_outputs = Snet(source_data)
    target_outputs = Tnet(target_data)
    loss = criterion(source_outputs.float(), target_outputs.float(), source_labels.float(), target_labels.float(), Snet, Tnet, args)
    loss.backward()
    Soptimizer.step()
    Toptimizer.step()
    return loss.item() 

def test(net, test_data, test_labels, criterion):
        net.eval()
        with torch.no_grad():
                outputs = net(test_data)
                loss = criterion(outputs, test_labels) 
        return loss.item()

def multitaskMSE(Snet, Tnet, Sinputs, Tinputs, Stargets, Ttargets, criterion, args):
        Snet.eval()
        Tnet.eval()
        with torch.no_grad():
                Soutputs = Snet(Sinputs)
                Toutputs = Tnet(Tinputs)
                Sloss = criterion(Soutputs, Stargets) 
                Tloss = criterion(Toutputs, Ttargets) 
        loss = Sloss + Tloss 
        return loss.item()/(args.Ps + args.Pt)

def multitaskLoss(source_output, target_output, source_labels, target_labels, Snet, Tnet, args):
    loss = 0
    for i in range(len(Snet)):
        if i == len(Snet)-1:
            loss += (0.5*args.lambda1*args.T)*(torch.linalg.matrix_norm(Snet[i].weight)**2)
            loss += (0.5*args.lambda1*args.T)*(torch.linalg.matrix_norm(Tnet[i].weight)**2)
        else:
            if isinstance(Snet[i], nn.Linear):
                #print("hello")
                #loss += (0.5*(args.lambda0-args.gamma)*args.T)*(torch.linalg.matrix_norm(Snet[i].weight)**2)
                #loss += (0.5*(args.lambda0-args.gamma)*args.T)*(torch.linalg.matrix_norm(Tnet[i].weight)**2)
                loss += (0.5*(args.lambda0)*args.T)*(torch.linalg.matrix_norm(Snet[i].weight)**2)
                loss += (0.5*(args.lambda0)*args.T)*(torch.linalg.matrix_norm(Tnet[i].weight)**2)
                #loss += (0.5*args.gamma*args.T)*(torch.linalg.matrix_norm(Snet[i].weight-Tnet[i].weight)**2)
                loss -= (0.5*args.gamma*args.T)*(torch.dot(Snet[i].weight.flatten(),Tnet[i].weight.flatten()))
    loss += 0.5*torch.sum((source_output - source_labels)**2)
    loss += 0.5*torch.sum((target_output - target_labels)**2)
    return loss

def computeNormsAndOverlaps(Snet,Tnet):
    i = len(Snet)-1
    Snorm = torch.linalg.matrix_norm(Snet[i].weight)
    Tnorm = torch.linalg.matrix_norm(Tnet[i].weight)
    overlap = torch.dot(Snet[i].weight.flatten(),Tnet[i].weight.flatten())
    return Snorm, Tnorm, overlap

def train(net,data, labels, criterion, optimizer,T,lambda0,lambda1):
    net.train()
    inputs, targets = data, labels#).unsqueeze(1)
    optimizer.zero_grad()
    outputs = net(inputs)
    #loss = criterion(outputs.float(),targets.float())
    loss = criterion(outputs.float(), targets.float(),net,T,lambda0,lambda1)
    loss.backward()
    optimizer.step()
    return loss.item() 

def regLoss(output, target,net,T,lambda0,lambda1):
    loss = 0
    for i in range(len(net)):
        if i == len(net)-1:
            loss += (0.5*lambda1*T)*(torch.linalg.matrix_norm(net[i].weight)**2)
        else:
            if (isinstance(net[i],nn.Linear) or isinstance(net[i],nn.Conv2d)):
                loss += (0.5*lambda0*T)*(torch.linalg.norm(net[i].weight.flatten())**2)
    loss += 0.5*torch.sum((output - target)**2)
    return loss