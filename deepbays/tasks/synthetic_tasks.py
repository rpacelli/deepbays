
import os
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..NN import *
from deepbays.NN.networks import make_act_module

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
        model = FCNet(N0, hidden_dim, L=1, bias = False, act=act)
        self.model = model.Sequential()
        # Initialize the model parameters
        self.seed = dataSeed
        self.initialize_model()

    def initialize_model(self):
        # Set the model to evaluation mode and initialize with random weights
        # CK: save current seed and reseed with dataSeed, to fix the realization of the target function. Then set the state of the rng back to how it was before!
        saveseed = torch.initial_seed()
        torch.manual_seed(self.seed) # use dataSeed for 
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                nn.init.normal_(param, mean=0, std=1)
        torch.manual_seed(saveseed) # restore rng seed to before initialization
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed) 
        inputs = torch.tensor(rng.randn(P, self.N0), dtype=torch.float)
        test_inputs = torch.tensor(rng.randn(Pt, self.N0), dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            targets = self.model(inputs).squeeze()
            test_targets = self.model(test_inputs).squeeze()
        
        return inputs, targets.unsqueeze(1), test_inputs, test_targets.unsqueeze(1)
    

class single_index_dataset:
    def __init__(self, N0, actfunc, dataSeed = 1234):
        self.N0 = N0
        self.actfunc = make_act_module(actfunc)
        self.seed = dataSeed
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed)  
        # Generate a random normalized teacher weight vector (w)
        w = rng.randn(self.N0)
        w /= np.linalg.norm(w)
        # Create training data
        X = rng.randn(P, self.N0)
        Y = np.dot(X, w).reshape(P,1) #equivalent to unsqueeze
        X = torch.Tensor(X)
        Y = self.actfunc(torch.Tensor(Y)) # nonlinearity
        # Create test data
        Xtest = rng.randn(Pt, self.N0)
        Ytest = np.dot(Xtest, w).reshape(Pt,1)
        Xtest = torch.Tensor(Xtest)
        Ytest = self.actfunc(torch.Tensor(Ytest))
        return X, Y, Xtest, Ytest
    
    
class hermite_dataset:
    '''Hermite single index task according to Fig5 in Rubin2025 (assumes physicists convention of hermite polynomial def) '''
    def __init__(self, N0=30, hermite_coefs=[0, 1, 0, -0.1], dataSeed = 1234, e1_teacher=False):
        self.N0 = N0
        self.actfunc = np.polynomial.hermite.Hermite(hermite_coefs)
        self.seed = dataSeed
        self.e1_teacher = e1_teacher
    
    def make_data(self, P, Pt):
        rng = np.random.RandomState(self.seed)  
        # Generate a random normalized teacher weight vector (w)
        w = rng.randn(self.N0)
        w /= np.linalg.norm(w)
        if self.e1_teacher:
            # overwrite (so as not to change random numbers) random teacher vector with e1 basis vector (dangerous but that is what Rubin2025 write in App.C.3)
            w = np.zeros_like(w)
            w[0] = 1.
        # Create training data
        X = rng.randn(P, self.N0)
        Y = self.actfunc(np.dot(X, w)) 
        X = torch.Tensor(X)
        Y = torch.Tensor(Y).unsqueeze(1) # shape (P,1)
        # Create test data
        Xtest = rng.randn(Pt, self.N0)
        Ytest = self.actfunc(np.dot(Xtest, w))
        Xtest = torch.Tensor(Xtest)
        Ytest = torch.Tensor(Ytest).unsqueeze(1) # shape (Pt,1)
        return X, Y, Xtest, Ytest


class pokerhand_dataset:
    """Todo: add function do download and preprocess data, currently need to ask CK to get the data files """
    
    def __init__(
        self,
        datadir: str = '~/torchvision/poker/',
        labels: list = None,
        fake_bias: bool = True
        ):
        self.labels = labels
        self.train_file = os.path.expanduser(datadir) + 'train_reducedpoker.pt'
        self.test_file = os.path.expanduser(datadir) + 'test_reducedpoker.pt'
        self.fake_bias = fake_bias
        

    def load_poker_data(
        self,
        pt_file: str,
        nsamples: int = -1,
        labels: list = None,
        regression: bool = True,
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
            fake_bias: adds an 11th dim to the samples which is always 1.0, 
                       corresponds to allowing biases in the first layer,
                       and makes sense because generalization on this task strongly depends on allowing biases.
    
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

        # Append dimension of ones for fake bias feature if requested
        if self.fake_bias:
            ones_col = torch.ones((X.size(0), 1), dtype=X.dtype, device=X.device)
            X = torch.cat((X, ones_col), dim=1)
        
        if regression:
            return X, y.unsqueeze(1).float()
        else:
            return X, y
    
    def make_data(self, P, Pt):
        inputs, targets = self.load_poker_data(self.train_file, nsamples=P, labels=self.labels, regression=True)
        test_inputs, test_targets = self.load_poker_data(self.test_file, nsamples=Pt, labels=self.labels, regression=True)
        return inputs, targets, test_inputs, test_targets
    
    

def add_first_layer_bias(X, Xtest):
    """ Adds a column of ones to the datamatrix, which is the classic trick of emulating trainable biases in the first layer. """
    Xnew = torch.concat([X, torch.ones(X.shape[0],1)], dim=1)
    Xtestnew = torch.concat([Xtest, torch.ones(Xtest.shape[0],1)], dim=1)
    return Xnew, Xtestnew


    