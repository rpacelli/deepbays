
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def map_to_binary(vector, original_values):
    """
    Transform a vector with two original values to -1/1.
    
    Args:
        vector: array with two distinct values (e.g., [1, 3, 1, 3, ...])
        original_values: tuple/list of two values (first goes to -1, second to 1)
    
    Returns:
        array with values -1 and 1
    """
    return np.where(vector == original_values[0], -1, 1)

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
    #zero_one_labels = [0 if x  ==  labels[0] else 1 for x in filtered_labels]
    return filtered_data, filtered_labels #  torch.tensor(zero_one_labels)

def getTransforms(self):
        T = t.Compose([
            t.Resize(size = self.side_size), 
            t.ToTensor(), 
            t.Grayscale(), 
            #t.Normalize((0.5),(0.24)),
        ])
        if self.flatten:
            T = t.Compose([
                T, 
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

def oneHotEncoding(y, verbose=True):
    uniqueClasses = torch.unique(y);
    classToIndex = {cls.item(): idx for idx, cls in enumerate(uniqueClasses)};
    mappedTensor = torch.tensor([classToIndex[val.item()] for val in y]);
    numClasses = len(uniqueClasses);
    oneHotEncoded = F.one_hot(mappedTensor, num_classes=numClasses);
    if verbose:
        print("Performing One Hot Encoding of your data:")
        print(f"> num of classes: {numClasses}")
        print(f"> class dictionary (from old to new):")
        for newClass, oldClass in enumerate(classToIndex):
            tempArray = np.zeros(numClasses);
            tempArray[newClass] = 1;
            print(f"   old class : {oldClass} >> new class {tempArray}")
    return oneHotEncoded;

class mnist_dataset: 
    def __init__(self, N, selectedLabels, binaryLabels = True): 
        #selectedLabels is a list, for example [2, 3] will select the labels 2 and 3 from mnist
        #if binary labels = True, two class classification will be mapped into -1 and 1, otherwise the original labels will be kept.
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.binaryLabels = binaryLabels
    def make_data(self, P, Ptest, batchSize = 60000, dataSeed = 1234, flatten = False):
        self.flatten = flatten
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchSize)
        testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 10000, num_workers = 0)
        # Filter train and test datasets
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, dataSeed)
        
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, dataSeed)
        if self.binaryLabels:
            labels = map_to_binary(labels, self.selectedLabels)
            testLabels = map_to_binary(testLabels, self.selectedLabels)
        data, testData  = normalizeDataset(data, testData)
        return data.numpy(), labels.reshape(P,1), testData.numpy(), testLabels.reshape(Ptest,1)
 

class cifar_dataset: 
    def __init__(self, N, selectedLabels, binaryLabels = True):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.binaryLabels = binaryLabels
    def make_data(self, P, Ptest, batchSize = 60000, dataSeed=123, flatten = False):
        self.flatten = flatten
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchSize, num_workers = 0)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 10000, num_workers = 0)
        #all_data, targets = next(iter(trainloader))
        # Filter train and test datasets
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, dataSeed)
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, dataSeed)
        data, testData  = normalizeDataset(data, testData)
        if self.binaryLabels:
            labels = map_to_binary(labels, self.selectedLabels)
            testLabels = map_to_binary(testLabels, self.selectedLabels)
        return data.numpy(), labels.reshape(P,1), testData.numpy(), testLabels.reshape(Ptest,1)
