# Documentation
The `deepbays` package provides a comprehensive framework for defining and training a special class of Gaussian Processes, the Renormalized Kernel Gaussian Process (RKGP), that are equivalent to neural networks in certain regimes. 

In addition to RKGPs, the package includes:
- A variety of computer vision tasks (both synthetic and real-world),
- Different neural network (NN) architectures 
- An optimizer for training NNs based on discrete Langevin Dynamics

This documentation provides an overview of the main features. 
## Installation
To install the `deepbays` package, use the following command:
```bash
pip install deepbays
```
## Theoretical background for understanding RKGPs
The renormalized kernel gaussian process is ....
This is all the material needed to reproduce experiments in the following papers: 
## Core Functionality: `RKGP` Function
The `RKGP` function is the main functionality of `deepbays`. It returns an RKGP model based on the specified parameters.
```python
RKGP(L: int, 
	 widths: Union[int, List[int]], 
	 D: int, 
	 T: float, 
	 priors: List[float] = [], 
	 act: str = "erf", 
	 model_type: str = "FC", 
	 kernel_type: str = "best"):
    """
    Constructs a Renormalized Kernel Gaussian Process model.

    Parameters:
    - L (int): Number of hidden layers. Must be an integer.
    - widths (int): Size of the hidden layer. Must be an integer.
    - D (int): Number of network outputs. Must be an integer greater than 0.
    - T (float): Temperature parameter.
    - priors (list of float, optional): Gaussian priors for each layer. 
       Defaults to None.
    - act (str, optional): Activation function. Defaults to "erf".
    - model_type (str, optional): Type of model. Defaults to "FC".
    - kernel_type (str, optional): Kernel type. Defaults to "best". 
       If kernel_type is "vanilla", first order corrections to the proportional limit will not be computed. 

    Returns:
    - model: The constructed RKGP model with specified parameters.

    Raises:
    - ValueError: If L is not an integer or is less than 1.
    - ValueError: If widths is not an integer.
    - ValueError: If D is not an integer or is less than 1.
    - ValueError: If kernel_type is not "best" or "vanilla". 
    """
```
To see all available options for RKGP, see Package Structure [[#RKGP Module]]
## Package Structure
The `deepbays` package is organized into the following modules:
- `rkgp`: Contains the core functionality for defining and training Gaussian Processes. This module is called by the RKGP function. 
- `tasks`: Contains datasets for training and testing.
- `NN`: Contains neural network training and evaluation utilities. Can be used to compare the RKGP performance to that of the equivalent network model.
- `kernels`: Contains kernel functions used in RKGPs

## Modules
#### `rkgp` Module
The `rkgp` module contains the core functionality for defining and training Gaussian Processes. You can directly define the RKGPs from this module, or you can use the RKGP function (which has a more standardized call for arguments). 
##### Available Functions and Classes
- `FC_1HL`: Single hidden layer fully connected GP model.
    - Parameters: `N1`, `T`, `l0`, `l1`, `act`
- `FC_1HL_multiclass`: Single hidden layer fully connected GP model for multiclass classification.
    - Parameters: `N1`, `D`, `T`, `l0`, `l1`, `act`
- `FC_1HL_corrected`: Single hidden layer fully connected GP model with corrections.
    - Parameters: `N1`, `T`, `l0`, `l1`, `act`
- `FC_deep`: Deep fully connected GP model.
    - Parameters: (inferred from the rest of the code)
- `FC_1HL_nonodd`

- `FC_1HL_spectral`

- `FC_1HL_multitask`
#### `tasks` Module
The `tasks` module provides various datasets for training and testing.
##### Available Datasets
- `cifar_dataset`
- `mnist_dataset`
- `emnistABFL_CHIS`
- `emnistABEL_CHJS`
- `synthetic_1hl_dataset`
- `perceptron_dataset`
  
Each dataset class typically includes the following methods:
- `__init__(self, N0, classes, seed)`: Initializes the dataset with the specified parameters.
 - N0: Input dimension of the dataset.
 - classes: Tuple specifying the classes of interest.
 - seed: Seed for reproducibility.
 Example:
 ```python
 data_class = db.tasks.mnist_dataset(N0=784, classes=(0, 1), seed=1234)
 ```
- `make_data(self, P, Pt)`: Generates the data and labels for training and testing. Returns tuple containing training data, training labels, test data, and test labels
 - P: Number of training examples.
 - Pt: Number of test examples.

 Example:
 ```python
 data, labels, test_data, test_labels = data_class.make_data(P=5, Pt=5)
 ```
#### `NN Module`
The `NN` module provides utilities for training and evaluating neural networks.
##### Available Functions
- `train`
- `regLoss`
- `multitaskMSE`
- `multitaskTrain`
- `multitaskLoss`
- `test`
- `LangevinOpt`
##### Available Classes
- `FCNet`
#### Kernels Module
The `kernels` module provides various kernel functions used in Gaussian Processes.
##### Available Functions
- `kernel_erf`
- `kernel_erf_torch`
- `kernel_relu`
- `kernel_relu_bias`
- `mean_relu`
- `computeKmatrix`
### Interactive Python Notebook

To help users get started quickly, an interactive Jupyter Notebook is also provided. The notebook includes step-by-step instructions and code cells to run the examples interactively.

You can find the notebook here: DeepBays_Tutorial.ipynb

## Tutorial
Here's a step-by-step tutorial on how to use the `deepbays` package.
### Step 1: Import DeepBays and Necessary Libraries
```python
import deepbays as db 
import time import numpy as np
```
### Step 2: Define Your Training and Test Data
You can use one of the predefined tasks or your own data. Here, we use the MNIST dataset.
```python
N0 = 784  # Input dimension
classes = [1, 2, 3]  # MNIST classes to use
seed = 1234  # Seed for reproducibility
P = 50  # Number of training examples
Pt = 50  # Number of test examples

dataclass = db.tasks.mnist_dataset(N0, classes, seed = seed)
X, Y, Xtest, Ytest = dataclass.make_data(P, Pt)
```
### Step 3: Choose Specifications for the RKGP and instance it
Specify the parameters for the Gaussian Process model:
```python
L = 1  # Number of hidden layers D = 3  # Number of network outputs 
priors = [0.3, 0.1]  # Gaussian priors for each layer 
N1 = 800  # Size of the hidden layer 
T = 0.02  # Temperature  
gp = db.RKGP(L, N1, D, T, priors)
```
### Step 4: Preprocess the Data and Train the Model
Preprocess the data, minimize the action (that is the process of train the GP), and make predictions over test set.
```python
gp.preprocess(X, Y)
gp.minimizeAction() # alternatively, you can use gp.setIW() that sets
gp.predict(Xtest) 
gp.averageLoss(Ytest)
```  
