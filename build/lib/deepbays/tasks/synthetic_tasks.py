
import torch, torchvision, torchvision.transforms as t 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class perceptron(nn.Module):
    def __init__(self, input_dim):
        super(perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x
      
# Define a simple 1-hidden-layer neural network
class Simple1HLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Simple1HLNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

def relu(x):
    return x if x > 0 else 0

class random_dataset: 
    def __init__(self, N):
        self.N = N

    def make_data(self, P, Ptest):
        inputs = torch.randn((P, self.N))
        targets = torch.randn(P)
        test_inputs = torch.randn((Ptest, self.N))
        test_targets = torch.randn(Ptest)
        return inputs, targets, test_inputs, test_targets
    
class hmm_dataset:
    def __init__(self, N, D, F, Wt, whichTask, q, eta, rho, actx):
        self.N = N
        self.D = D
        self.side_size = int(np.sqrt(self.N))
        self.save_data = bool("NaN")
        self.randmat_file = f"randmat_N0_{self.N}.txt"
        self.whichTask = whichTask
        self.q = q
        self.eta = eta
        self.rho = rho
        self.actx = actx
        self.Wt = Wt
        self.F = F

    def perturb_teachers(self):
        Wt_perturbed = np.copy(self.Wt)
        n_latent = len(Wt_perturbed)
        #print("latent dimension is", n_latent)
        Wt_perturbed = self.q * Wt_perturbed + np.sqrt(1 - self.q**2) * np.random.normal(0., 1., size = (n_latent))
        return Wt_perturbed
    
    def perturb_features(self):
        F_perturbed = np.copy(self.F)
        n_latent, n = np.shape(F_perturbed)
        n_rows = int(self.rho*n_latent)
        F_perturbed = self.eta * F_perturbed + np.sqrt(1 - self.eta**2) * np.random.normal(0, 1., size = (n_latent, n))
        F_perturbed[:n_rows] = np.random.normal(0, 1., size = (n_rows, n))
        return F_perturbed

    def get_hmm(self, P, P_test):
        if self.whichTask == 'target':
            Wt = self.perturb_teachers()
            F = self.perturb_features()
        else: 
            Wt = self.Wt
            F = self.F
        c_train = np.random.normal(0., 1., size = (P, self.D))
        c_test = np.random.normal(0., 1., size = (P_test, self.D))
        if self.actx == 'tanh':
            x_train = np.tanh(c_train@F/np.sqrt(self.D))
            x_test = np.tanh(c_test@F/np.sqrt(self.D))
        if self.actx == 'relu':
            x_train = relu(c_train@F/np.sqrt(self.D))
            x_test = relu(c_test@F/np.sqrt(self.D))
        y_train = np.sign(c_train @ Wt)
        y_test = np.sign(c_test @ Wt)
        x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
        return x_train, y_train, x_test, y_test

    def make_data(self, P, P_test, batch_size):

        if self.whichTask not in ["source", "target"]:
            raise Exception("Sorry, no task found!")
        else:
            data, labels, test_data, test_labels = self.get_hmm(P, P_test)
        #print("the dimension of data is ", len(data[0]))
        return data, labels, test_data, test_labels


class synthetic_1hl_dataset: 
    def __init__(self, N, hidden_dim):
        self.N = N
        self.hidden_dim = hidden_dim
        self.model = Simple1HLNet(N, hidden_dim)
        # Initialize the model parameters
        self.initialize_model()

    def initialize_model(self):
        # Set the model to evaluation mode and initialize with random weights
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                nn.init.normal_(param, mean=0, std=1)
    
    def make_data(self, P, Ptest):
        inputs = torch.randn((P, self.N))
        test_inputs = torch.randn((Ptest, self.N))
        
        with torch.no_grad():
            targets = self.model(inputs).squeeze()
            test_targets = self.model(test_inputs).squeeze()
        
        return inputs, targets.unsqueeze(1), test_inputs, test_targets.unsqueeze(1)
    
class perceptron_dataset: 
    def __init__(self, N):
        self.N = N
        self.model = perceptron(N)
        # Initialize the model parameters
        self.initialize_model()

    def initialize_model(self):
        # Set the model to evaluation mode and initialize with random weights
        self.model.eval()
        with torch.no_grad():
            for param in self.model.parameters():
                nn.init.normal_(param, mean=0, std=1)
    
    def make_data(self, P, Ptest):
        inputs = torch.randn((P, self.N))
        test_inputs = torch.randn((Ptest, self.N))
        
        with torch.no_grad():
            targets = self.model(inputs).squeeze()
            test_targets = self.model(test_inputs).squeeze()
        
        return inputs, targets.unsqueeze(1), test_inputs, test_targets.unsqueeze(1)