
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
    

class random_binary_Ksparse_dataset:
    """
    Generate random binary data from Gaussian distribution.
    
    Data generation process:
    1. Generate random numbers from N(0, 1)
    2. Take sign to get -1/1 values
    3. Labels are product of first K features: For K =4, the labels are Y = X[:,0] * X[:,1] * X[:,2] * X[:,3]
    """
    def __init__(self, N0, P, Ptest, K, seed):
        self.N0 = N0 #number of features
        self.P = P #number of training samples
        self.Ptest = Ptest
        self.seed = seed
        self.K = K #number of feature used in the labels. For example, if K=4, the labels are the product of the first 4 features.
    
    def product_of_first_k_features(self, X, K):
            """
            Compute the product of the first K features for each sample.

            Parameters:
            X : numpy array of shape (n_samples, n_features)
            K : integer, number of features to multiply

            Returns:
            Y : numpy array of shape (n_samples, 1)
            """
            # Take first K features and multiply along axis=1
            Y = np.prod(X[:, :K], axis=1, keepdims=True)
            return Y
    
    def make_data(self):
        """Generate training and test data."""
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Generate training data
        X = np.random.normal(0, 1, size=(self.P, self.N0))
        X = np.sign(X)
        
        # Generate test data
        np.random.seed(self.seed + 1000)
        Xtest = np.random.normal(0, 1, size=(self.Ptest, self.N0))
        Xtest = np.sign(Xtest)

        # Generate labels as product of first K features
        Y = self.product_of_first_k_features(X, self.K)
        Ytest = self.product_of_first_k_features(Xtest, self.K)

        return X, Y, Xtest, Ytest
    