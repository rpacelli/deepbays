
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def relu(x):
    return x if x > 0 else 0

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

def accumulateClasses(X, Y, classesTask1, classesTask2):
    X1 = X[Y == classesTask1[0]]
    X2 = X[Y == classesTask2[0]]
    for c in range(1,len(classesTask1)):
        X1 = np.concatenate([X1, X[Y == classesTask1[c]]])
        X2 = np.concatenate([X2, X[Y == classesTask2[c]]])
    Y1, Y2  = np.zeros(len(X1)), np.ones(len(X2))
    Xnew, Ynew = np.concatenate([X1, X2]), np.concatenate([Y1, Y2])
    return Xnew, Ynew

def preprocessDataset(Xtrain, Ytrain, Xtest, Ytest, dataSeed, whichLabels, P,Ptest):
    rng = np.random.RandomState(dataSeed)   
    rp = rng.permutation(len(Ytrain))
    rptest = rng.permutation(len(Ytest))
    Xtrain, Ytrain = Xtrain[rp[:P]], Ytrain[rp[:P]]
    Xtest, Ytest =  Xtest[rptest[:Ptest]], Ytest[rptest[:Ptest]]
    if whichLabels[0] == -1 and whichLabels[1] == 1:
        Ytrain = 2 * Ytrain - 1
        Ytest = 2 * Ytest - 1
    return Xtrain, Ytrain, Xtest, Ytest

def randomProjectDataset(Xtrain, Xtest, D, randomProject):
    N0 = len(Xtrain[0])
    rng2 = np.random.RandomState(randomProject) 
    wmatrix = rng2.random(( N0, D))
    Xtrain = relu(np.dot(Xtrain, wmatrix)/np.sqrt(N0))
    Xtest = relu(np.dot(Xtest, wmatrix)/np.sqrt(N0))
    return Xtrain, Xtest

class emnistABEL_CHJS: 
    def __init__(self, N, whichLabels, randomProject, dataSeed):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.whichLabels = whichLabels
        if randomProject > 0:
            self.projectSeed = randomProject
            self.randomProject = True
        self.dataSeed = dataSeed
    def get_dataset_emnist(self, Xtrain, Ytrain, Xtest, Ytest):
        Xtrain, Ytrain = accumulateClasses(Xtrain, Ytrain, [1,2,5,12], [3,8,10,19])
        Xtest, Ytest = accumulateClasses(Xtest, Ytest, [1,2,5,12], [3,8,10,19])
        Xtrain, Ytrain, Xtest, Ytest = preprocessDataset(Xtrain, Ytrain, Xtest, Ytest, self.dataSeed, self.whichLabels, self.P, self.Ptest)
        Xtrain, Xtest = normalizeDataset(Xtrain, Xtest)
        if self.projectSeed > 0:
            Xtrain, Xtest = randomProjectDataset(Xtrain, Xtest, self.D, self.projectSeed)
        Xtrain, Ytrain, Xtest, Ytest = torch.tensor(Xtrain, dtype=torch.float), torch.tensor(Ytrain, dtype=torch.float), torch.tensor(Xtest, dtype=torch.float), torch.tensor(Ytest, dtype=torch.float)
        return Xtrain, Ytrain, Xtest, Ytest 
    def make_data(self, P, Ptest, batch_size, D):
        self.P, self.Ptest, self.D = P, Ptest, D
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = True,  transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0)
        testset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = False, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, num_workers = 0)
        data, labels = next(iter(trainloader))
        Xtest, Ytest = next(iter(testloader))
        data, labels, Xtest, Ytest = self.get_dataset_emnist(data, labels, Xtest, Ytest)
        return data, labels.unsqueeze(1), Xtest, Ytest.unsqueeze(1)
    
class emnistABFL_CHIS: 
    def __init__(self, N, whichLabels, randomProject, dataSeed):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.whichLabels = whichLabels
        if randomProject > 0:
            self.projectSeed = randomProject
            self.randomProject = True
        self.dataSeed = dataSeed
    def get_dataset_emnist(self, Xtrain, Ytrain, Xtest, Ytest):
        Xtrain, Ytrain = accumulateClasses(Xtrain, Ytrain, [1,2,6,12], [3,8,9,19])
        Xtest, Ytest = accumulateClasses(Xtest, Ytest, [1,2,6,12], [3,8,9,19])
        Xtrain, Ytrain, Xtest, Ytest = preprocessDataset(Xtrain, Ytrain, Xtest, Ytest, self.dataSeed, self.whichLabels, self.P, self.Ptest)
        Xtrain, Xtest = normalizeDataset(Xtrain, Xtest)
        if self.projectSeed > 0:
            Xtrain, Xtest = randomProjectDataset(Xtrain, Xtest, self.D, self.projectSeed)
        Xtrain, Ytrain, Xtest, Ytest = torch.tensor(Xtrain, dtype=torch.float), torch.tensor(Ytrain, dtype=torch.float), torch.tensor(Xtest, dtype=torch.float), torch.tensor(Ytest, dtype=torch.float)
        return Xtrain, Ytrain, Xtest, Ytest 
    def make_data(self, P, Ptest, batch_size, D):
        self.P, self.Ptest, self.D = P, Ptest, D
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = True,  transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0)
        testset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = False, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, num_workers = 0)
        data, labels = next(iter(trainloader))
        Xtest, Ytest = next(iter(testloader))
        data, labels, Xtest, Ytest = self.get_dataset_emnist(data, labels, Xtest, Ytest)
        return data, labels.unsqueeze(1), Xtest, Ytest.unsqueeze(1)