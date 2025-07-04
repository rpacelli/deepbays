
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
        w /= np.linalg.norm(w)
        # Create training data
        X = rng.randn(P, self.N0)
        Y = np.dot(X, w).reshape(P,1) #equivalent to unsqueeze
        # Create test data
        Xtest = rng.randn(Pt, self.N0)
        Ytest = np.dot(Xtest, w).reshape(Pt,1)
        return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float), torch.tensor( Xtest, dtype=torch.float), torch.tensor(Ytest, dtype=torch.float)

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
        # CK: use private rng for weight init seeded with dataSeed, to fix the realization of the target function and to avoid changing the state of the global torch rng which may be used elsewhere in the code!
        rng = torch.Generator()
        rng.manual_seed(self.seed) 
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                nn.init.normal_(param, mean=0, std=1, generator=rng)
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed) 
        inputs = torch.tensor(rng.randn(P, self.N0), dtype=torch.float)
        test_inputs = torch.tensor(rng.randn(Pt, self.N0), dtype=torch.float)
        with torch.no_grad():
            targets = self.model(inputs).squeeze()
            test_targets = self.model(test_inputs).squeeze()
        
        return inputs, targets.unsqueeze(1), test_inputs, test_targets.unsqueeze(1)
    


class pokerhand_dataset:
    """Todo: currently add function do download and preprocess data, currently need to ask CK to get the data files """
    
    def __init__(
        self,
        datadir: str = '/home/keup/torchvision/poker/',
        labels: list = None
        ):
        self.labels = labels
        self.train_file = datadir + 'train_reducedpoker.pt'
        self.test_file = datadir + 'test_reducedpoker.pt'
        

    def load_poker_data(
        self,
        pt_file: str,
        nsamples: int = -1,
        labels: list = None,
        regression: bool = True
        ):
        """
        Loads a .pt file saved by prepare_poker_data and returns a torch.Tensors X and y,
        with options to restrict number of samples and subset of labels (their values will be remapped to 0, 1, 2,...).
    
        Args:
            pt_file: path to the .pt file containing (features, labels).
            nsamples: number of samples to include (after label filtering). Use -1 for all.
            labels: list of class labels to include (e.g. [1,3,4]). Use None for all classes.
            regression: if True, return the labels as float in shape (nsamples, 1), 
                        if False, return the labels as long int in shape (nsamples), for use with crossentropy loss
    
        Returns:
            X: Tensor of shape (nsamples, 10)
            y: Tensor of shape (nsamples, 1)
        """
        X, y = torch.load(pt_file)
    
        # Filter by labels if provided
        if labels is not None:
            mask = torch.zeros_like(y, dtype=torch.bool)
            for lbl in labels:
                mask |= (y == lbl)
            X = X[mask]
            y = y[mask]
            
            # remap labels to 0, 1, 2 ...
            new_y = torch.empty_like(y)
            for new_idx, orig_lbl in enumerate(labels):
                new_y[y == orig_lbl] = new_idx
            y = new_y
    
        # Restrict number of samples
        if nsamples > 0:
            X = X[:nsamples]
            y = y[:nsamples]
        
        if regression:
            return X, y.unsqueeze(1).float()
        else:
            return X, y
    
    def make_data(self, P, Pt):
        inputs, targets = self.load_poker_data(self.train_file, nsamples=P, labels=self.labels, regression=True)
        test_inputs, test_targets = self.load_poker_data(self.test_file, nsamples=Pt, labels=self.labels, regression=True)
        return inputs, targets, test_inputs, test_targets
    
    

    