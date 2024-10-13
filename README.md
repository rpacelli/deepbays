# Documentation
The `deepbays` package provides a comprehensive framework for defining and training a special class of Gaussian Processes, the Renormalized Kernel Gaussian Process (RKGP), that are equivalent to neural networks in certain regimes. 

In addition to RKGPs, the package includes:
- A variety of computer vision tasks (both synthetic and real-world),
- Different neural network (NN) architectures with the standard scaling,
- An optimizer for training NNs based on discrete Langevin Dynamics

This documentation provides an overview of the main features, and links to interactive python notebooks.

NB: This repository contains the code to reproduce the experiments contained in "Local Kernel Renormalization as a mechanism for feature learning in overparameterized Convolutional Neural Networks".

## Installation
To install the `deepbays` package, use the following command:
```bash
pip install deepbays
```
## Theoretical background for understanding RKGPs
This code refers to a series of articles, linked below, which contain all the theoretical background to understanding RKGPs and their exact link to neural networks. 
- A statistical mechanics framework for Bayesian deep neural networks beyond the infinite-width limit, [Nature machine intelligence, 2023](https://www.nature.com/articles/s42256-023-00767-6).
- Predictive Power of a Bayesian Effective Action for Fully Connected One Hidden Layer Neural Networks in the Proportional Limit [Phys. Rev. Lett., 2024](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.027301).
- Local Kernel Renormalization as a mechanism for feature learning in overparameterized Convolutional Neural Networks. [ArXiv, 2023](https://arxiv.org/abs/2307.11807)
## Core Functionality: `RKGP` Function
The `RKGP` function is the main functionality of `deepbays`. It returns an RKGP model with the specified parameters of the equivalent neural network. To see all available combinations of RKGP, see Package Structure. 
```python
#Constructs a Renormalized Kernel Gaussian Process model.
RKGP(L: int, # Number of hidden layers
	 width: int, # Size of all the hidden layers.
	 D: int, # Number of network outputs.
	 T: float = 0.01, # Temperature 
	 priors: List[float] = [], # Gaussian Priors for each layer
	 act: str = "erf", # Activation function
	 model_type: str = "FC", # Type of layers
	 kernel_type: str = "vanilla" # Compute corrections when possible
		 ): 
# Returns RKGP model with specified parameters.
```
Parameters:
- `L (int)`: Number of hidden layers. The depth of the networ is L+1. Must be an integer.
- `width (int)`: Size of all the hidden layers. If the model is `FC`, this is the number of hidden layer units, if the model is `CONV` this is the number of convolutional channels in the hidden layer. Must be an integer.
- `D (int)`: Number of network outputs. Must be an integer.
- `T (float, optional)`: Temperature 
- `priors (list of float, optional)`: Gaussian priors for each layer. Must be a list of length L. Defaults to 1. for all layers.
- `act (str, optional)`: Activation function. Choose between: Error Function (`erf`), Rectified Linear Unit (`relu`), Identity function (`id`).
- `model_type (str, optional)`: Type of model. Choose between: fully-connected `FC`, and convolutional `CONV`
- `kernel_type (str, optional)`: Kernel type. Defaults to `vanilla`. If kernel_type is `best` and the architecture is 1HL FC, first order corrections to the proportional limit will be computed. 
## Interactive Python Notebook
To help users get started quickly, a series of interactive Jupyter Notebooks are also provided. The notebooks include step-by-step instructions and code cells to run the examples interactively.
- Intro to RKGP. Solving two simple regression and classification problems with RKGP: [Tutorial intro](https://colab.research.google.com/drive/1bMh6-5H8ptmDIfsk6_v2X_30qzK4sPlq?usp=sharing)
- Training a convolutional shallow network with Langevin dynamics over the MNIST dataset of handwritten digits: [Tutorial CNN Langevin training](https://colab.research.google.com/drive/1YxWXd4hCKG_AfZN6N0LT550n2diNuWm2?usp=sharing)
## Package Structure
The `deepbays` package is organized into the following modules:
- `rkgp`: Contains the core functionality for defining and training Gaussian Processes. This module is called by the RKGP function. 
- `tasks`: Contains datasets for training and testing.
- `NN`: Contains neural network training and evaluation utilities. Can be used to compare the RKGP performance to that of the equivalent network model.
- `kernels`: Contains kernel functions used in RKGPs
### `rkgp` module
The `rkgp` module contains the core functionality for defining and training Gaussian Processes. You can directly define the RKGPs from this module, or you can use the RKGP function (which has a more standardized call for arguments). 
#### Available Functions and Classes
- `FC_1HL(N1, T, l0, l1, act, bias)`:
	- Returns: GP model equivalent to a single-output one-hidden-layer fully-connected network with odd activation function (`erf`,`id`).
- `FC_1HL_multiclass(N1, D, T, l0, l1, act)`:
	- Returns: GP model for multiclass classification, equivalent to a fully-connected shallow network with multiple outputs.
- `FC_deep(L, N1, T, l0, l1, act)`:
	- Returns: GP model equivalent to a deep fully-connected neural network
- `FC_1HL_nonodd(N1, T, l0, l1, act)`:
	- Returns: GP model equivalent to a single-output one-hidden-layer fully-connected network with nonodd activation function (`relu`).
- `CONV_1HL(N1, T, l0, l1, act, mask, stride)`:
	- Returns: Shallow convolutional model with dense readout
#### Parameters for 1HL fully-connected networks
- `N1 : int` : Number of units in the hidden layer;
- `T : float` : Temperature;
- `D : int` : Number of outputs;
- `l0 : float = 1.`: Gaussian Prior of first hidden layer;
- `l1 : float = 1.`: Gaussian Prior of second hidden layer
- `act : str = "erf"`: Activation function. 
-`bias : bool = False` Bias for all layers
#### Parameters for deep fully-connected networks
- `N1 : int` : Number of units in the hidden layers;
- `T : float` : Temperature;
- `L : int` : Number of hidden layers;
- `priors : list = [1,...,1]`: Gaussian Priors for each layer. List of length `L`.
- `act : str = "erf"`: Activation function. 
#### Parameters for 1HL convolutional networks
- `Nc : int`: Number of channels (for the convolutional model)
- `T : float` : Temperature;
- `l0 : float = 1.`: Gaussian Prior of first hidden layer;
- `l1 : float = 1.`: Gaussian Prior of second hidden layer;
- `act : str = "erf"`: Activation function.
- `mask : int ` Size of convolutional mask (convolutional kernel);
- `stride : int ` Striding of convolution. 
### `tasks` module
The `tasks` module provides various datasets for training and testing. The avaliable datasets are listed as follows: 
- `cifar_dataset`. The CIFAR10 dataset.
- `mnist_dataset`. The MNIST dataset of handwritten digits.
- `synthetic_1hl_dataset`. A dataset of random data with labels given by a random 1HL network.
Each dataset class typically includes the following methods:
- `__init__(self, N0, classes, seed)`: Initializes the dataset with the specified parameters.
	- `N0`: Input dimension of the dataset.
	- classes: Tuple specifying the classes of interest.
	- seed: Seed for reproducibility.
- `make_data(self, P, Pt)`: Generates the data and labels for training and testing. Returns tuple containing training data, training labels, test data, and test labels
	- `P`: Number of training examples.
	- `Pt`: Number of test examples.
 Example:
 ```python
 import deepbays as db
 data_class = db.tasks.mnist_dataset(N0=784, classes=(0, 1), seed=1234) #select only classes with labels 0 and 1 from MNIST
 data, labels, test_data, test_labels = data_class.make_data(P=5, Pt=5)
 ```
### NN module
The `NN` module provides utilities for Bayesian training of standard scaled neural networks. Two tutorials on how to train the available models with the Bayesian sampler can be found LINK.
#### Available Functions
- `train` A function that trains a neural network using a specified dataset, optimizer, and loss function.
- `regLoss` This function computes the loss of the network, incorporating regularization terms rescaled by the temperature (gaussian priors).
- `test`A function that evaluates the MSE of the trained model on a test dataset.
- `LangevinOpt` A custom optimizer based on Langevin dynamics to simulate sampling from a posterior Gibbs distribution. This ensures Bayesian training. 
#### Available Classes
- `FCNet` A fully connected neural network consisting of one or more dense layers.
- `ConvNet`A convolutional neural network typically used for image-based tasks, consisting of one convolutional layer followed by a fully connected layer.
##### Example fully-connected network
```python
import torch
import deepbays as db
  
# Example: Define a fully connected network
input_dim = 784 # Input dimension (e.g., flattened 28x28 images)
hidden_units = 128 # Number of units in the hidden layers
num_hidden_layers = 2 # Number of hidden layers (L)
activation_function = "relu" # Activation function ('erf', 'relu')

# Instantiate the FCNet
model = db.NN.FCNet(N0=input_dim, N1=hidden_units, L=num_hidden_layers, act=activation_function)

# Get the sequential model for the network
net = model.Sequential()

# Example input (batch of 64, each with 784 features)
x = torch.randn(64, input_dim)

# Forward pass
output = net(x)
print(output.shape) # torch.Size([64, 1])
```
##### Example convolutional network
```python
import torch
import deepbays as db

# Example: Define a convolutional network for CIFAR-10 (input size 3x32x32, 10 output classes)
input_channels = 3 # CIFAR-10 images have 3 color channels (RGB)
N0 = 32*32 # Size of input in each channel
Nc = 20 # Number of channels in the second layer
mask = 8 # convolutional kernel size
stride = mask #set stride equal mask

# Instantiate the CONVNet
conv_model = db.NN.ConvNet(inputChannels = input_channels, N0 = N0, Nc = Nc, mask = mask, stride = stride )
conv_net = conv_model.Sequential()

# Example input (batch of 64 images, each with 3 channels and size 32x32)
x = torch.randn(64, input_channels, 32, 32)

# Forward pass
output = conv_net(x)
print(output.shape) # Output: torch.Size([64, 1])
```
#### Kernels Module
The `kernels` module provides various kernel functions used in Gaussian Processes. The kernels are in correspondance to (and depend on) the activation function. Available activation functions are the Error Function `erf` , Rectified Linear Unit `relu` and the identity function `id`. 
