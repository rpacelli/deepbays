
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def filter_by_label_old(data_loader, labels, P, whichTask):
    data, target = next(iter(data_loader))
    mask = torch.zeros_like(target, dtype = torch.bool)
    for label in labels:
        mask |=  target  ==  label
    filtered_data = data[mask][P*whichTask:P*whichTask+P]
    filtered_labels = target[mask][P*whichTask:P*whichTask+P]
    zero_one_labels = [0 if x  ==  labels[0] else 1 for x in filtered_labels]
    return filtered_data, torch.tensor(zero_one_labels)

def filter_by_label(data_loader, labels, P, dataSeed):
    data, target = next(iter(data_loader))
    mask = torch.zeros_like(target, dtype = torch.bool)
    for label in labels:
        mask |=  target  ==  label
    filtered_data = data[mask]
    filtered_labels = target[mask]
    rng = np.random.RandomState(dataSeed)  
    rp = rng.permutation(len(filtered_labels))
    filtered_data = filtered_data[rp[:P]]
    filtered_labels = filtered_labels[rp[:P]]
    zero_one_labels = [0 if x  ==  labels[0] else 1 for x in filtered_labels]
    return filtered_data, torch.tensor(zero_one_labels)

def getTransforms(self):
        T = t.Compose([
            t.Resize(size = self.side_size), 
            t.ToTensor(), 
            t.Grayscale(), 
            #t.Normalize((0.5),(0.24)),
            t.Lambda(lambda x: torch.flatten(x))
        ])
        return T

def normalizeDataset(data, testData):
    mean = data.mean()
    std = data.std()
    data = data - mean
    testData = testData - mean
    data = data / std
    testData = testData /std
    return data, testData

class mnist_dataset: 
    def __init__(self, N, selectedLabels, dataSeed = 123, whichTask=0): #selectedLabels is a list, for example [2, 3] will select the labels 2 and 3 from mnist
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.whichTask = whichTask
        self.dataSeed = dataSeed
    def make_data(self, P, Ptest, batchSize = 60000):
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchSize)
        testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 10000, num_workers = 0)
        # Filter train and test datasets
        #data, labels = filter_by_label_old(trainloader, self.selectedLabels, P, self.whichTask)
        #testData, testLabels = filter_by_label_old(testloader, self.selectedLabels, Ptest, self.whichTask)
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, self.dataSeed)
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, self.dataSeed)
        data, testData  = normalizeDataset(data, testData)
        return data, labels.unsqueeze(1), testData, testLabels.unsqueeze(1)
 

class cifar_dataset: 
    def __init__(self, N, selectedLabels, dataSeed=123):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.dataSeed = dataSeed
    def make_data(self, P, Ptest, batchSize = 60000):
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchSize, num_workers = 0)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 10000, num_workers = 0)
        #all_data, targets = next(iter(trainloader))
        # Filter train and test datasets
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, self.dataSeed)
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, self.dataSeed)
        data, testData  = normalizeDataset(data, testData)
        return data, labels.unsqueeze(1), testData, testLabels.unsqueeze(1)
