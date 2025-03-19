
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..NN import *

class random_dataset: 
    def __init__(self, N0):
        self.N0 = N0

    def make_data(self, P, Pt):
        inputs = torch.randn((P, self.N0))
        targets = torch.randn(P)
        test_inputs = torch.randn((Pt, self.N0))
        test_targets = torch.randn(Pt)
        return inputs, targets, test_inputs, test_targets
    
class linear_dataset:
    def __init__(self, N0, dataSeed = 1234):
        self.N0 = N0
        self.seed = dataSeed
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed)  
        # Generate a random normalized teacher weight vector (w)
        w = rng.randn(self.N0)
        w /= torch.linalg.norm(w)
        # Create training data
        X = rng.randn((P, self.N0))
        Y = torch.mv(X, w).reshape(P,1) #equivalent to unsqueeze
        # Create test data
        Xtest = rng.randn((Pt, self.N0))
        Ytest = torch.mv(Xtest, w).reshape(Pt,1)
        return X, Y, Xtest, Ytest

class synthetic_1hl_dataset: 
    def __init__(self, N0, hidden_dim, act, dataSeed = 1234):
        self.N0 = N0
        self.hidden_dim = hidden_dim
        model = FCNet(N0, hidden_dim, L=1)
        self.model = model.Sequential(bias = False, act_func=act)
        # Initialize the model parameters
        self.seed = dataSeed
        self.initialize_model()

    def initialize_model(self):
        # Set the model to evaluation mode and initialize with random weights
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                nn.init.normal_(param, mean=0, std=1)
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed) 
        inputs = torch.tensor(rng.randn(P, self.N0), dtype=torch.float)
        test_inputs = torch.tensor(rng.randn(Pt, self.N0), dtype=torch.float)
        with torch.no_grad():
            targets = self.model(inputs).squeeze()
            test_targets = self.model(test_inputs).squeeze()
        
        return inputs, targets.unsqueeze(1), test_inputs, test_targets.unsqueeze(1)
    