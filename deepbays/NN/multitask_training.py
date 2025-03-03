import os, os.path, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

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