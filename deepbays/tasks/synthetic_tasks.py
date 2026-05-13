
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..NN import FCNet

#All classes return X, Y, Xtest, Ytest as numpy arrays. This is for compatibility with the kernel code, which is written in numpy.
# X and Xtest are of shape (P, N0) and (Pt, N0) respectively. 
# Y and Ytest are of shape (P, 1) and (Pt, 1) respectively. This is for compatibility with pytorch network models.
# N0 is in the class init for consistency with classic_tasks

class random_dataset: 
    def __init__(self, N0):
        " This class generates random data with random labels"
        "Parameters: N0: input dimension, "
        self.N0 = N0

    def make_data(self, P, Pt, dataSeed=1234):
        "P: number of training samples, "
        "Pt: number of test samples, "
        "dataSeed: random seed for data generation"
        rng = np.random.RandomState(dataSeed)
        X = rng.randn((P, self.N0))
        Y = rng.randn(P).reshape(P,1)
        Xtest = rng.randn((Pt, self.N0))
        Ytest = rng.randn(Pt).reshape(P,1)
        return X,Y, Xtest, Ytest
        #return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float), torch.tensor( Xtest, dtype=torch.float), torch.tensor(Ytest, dtype=torch.float)

    
class linear_dataset:
    def __init__(self,N0):
        self.N0 = N0
    
    def make_data(self, P, Pt, dataSeed=1234):
        rng = np.random.RandomState(dataSeed)  
        # Generate a random normalized teacher weight vector (w)
        w = rng.randn(self.N0)
        w /= np.linalg.norm(w)
        # Create training data
        X = rng.randn(P, self.N0)
        Y = np.dot(X, w).reshape(P,1) #equivalent to unsqueeze
        # Create test data
        Xtest = rng.randn(Pt,self.N0)
        Ytest = np.dot(Xtest, w).reshape(Pt,1)
        #return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float), torch.tensor( Xtest, dtype=torch.float), torch.tensor(Ytest, dtype=torch.float)
        return X, Y, Xtest, Ytest
    

class synthetic_1hl_dataset: 
    def __init__(self, N0, hidden_dim, act, netSeed = 4321):
        """
        Generates random gaussian data with label given by a 1 hidden layer teacher network.
        Init takes parameter of the teacher 1hl network 
        N0: input dimension, this will also match data dimension,
        hidden_dim: number of teacher hidden units, 
        act: activation function of teacher's network, 
        netSeed: random seed for initializing the teacher network    
        """
        self.N0 = N0
        self.hidden_dim = hidden_dim
        model = FCNet(N0, hidden_dim, L=1,bias = False, act=act)
        self.model = model.Sequential()
        # Initialize the model parameters with netSeed
        self.initialize_model(netSeed)

    def initialize_model(self,netSeed):
        rng = np.random.RandomState(netSeed)
        # Set the model to evaluation mode and initialize with random weights
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # Generate numpy array with the same shape
                    numpy_values = rng.normal(0, 1, size=param.shape)
                    # Convert to torch tensor and assign
                    param.copy_(torch.from_numpy(numpy_values).float())
            self.model.eval()
    
    def make_data(self, P, Pt, dataSeed = 1234):
        rng = np.random.RandomState(dataSeed) 
        X = torch.tensor(rng.randn(P, self.N0), dtype=torch.float)
        Xtest = torch.tensor(rng.randn(Pt, self.N0), dtype=torch.float)
        with torch.no_grad():
            Y = self.model(X)
            Ytest = self.model(Xtest)
        return X.numpy(), Y.numpy(), Xtest.numpy(), Ytest.numpy()
    

class random_binary_Ksparse_dataset:
    """
    Generate random binary data from Gaussian distribution.
    
    Data generation process:
    1. Generate random numbers from N(0, 1)
    2. Take sign to get -1/1 values
    3. Labels are product of first K features: For K =4, the labels are Y = X[:,0] * X[:,1] * X[:,2] * X[:,3]
    """
    def __init__(self,N0, K):
        self.N0 = N0 
        self.K = K

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
    
    def make_data(self, P, Ptest, dataSeed):
        """
        Generate random binary data with labels as product of first K features. 
        Parameters: 
                   N0: number of features, 
                   P: number of training samples, Ptest: number of test samples, 
                   K: number of features used in labels, For example, if K=4, the labels are the product of the first 4 features.
                   seed: random seed for data generation"""
        # Set random seed for reproducibility
        rng = np.random.RandomState(dataSeed)
        # Generate training data
        X = rng.randn(P, self.N0)
        X = np.sign(X)
        # Generate test data
        Xtest = rng.randn(Ptest, self.N0)
        Xtest = np.sign(Xtest)
        # Generate labels as product of first K features
        Y = self.product_of_first_k_features(X, self.K)
        Ytest = self.product_of_first_k_features(Xtest, self.K)
        return X, Y, Xtest, Ytest    