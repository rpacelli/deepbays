# Deepbays documentation
`deepbays` is a Python package that implements a specific class of Gaussian Process (GP) to solve inference problems. The package also includes functionalities for handling neural networks (NN) and various datasets (tasks) used for testing the theories. This documentation will provide an overview of the main features
## Installation 
To install `deepbays`, use pip:
```
bash pip install deepbays
```

## Renormalized Kernel Gaussian Process (RKGP)
#### Defining the model 
The main function for defining and working with RKGP models is **RKGP** function that initialises the model with the specified parameters.

- `RKGP(L, widths, D, priors, T, act = "erf", model_type = "FC")`
  - **Parameters**:
    - `L`: Number of layers. Must be integer and positive.
    - `N1`: Number of neurons in the hidden layer. If `L>1`, a list containing the `L` lengths. 
    - `l0`: Input layer Gaussian prior. If `L>1`, a list containing the `L` lengths.
    - `l1`: Hidden layer Gaussian prior. If `L>1`, a list containing the `L` lengths.
    - `act`: Activation function (choose among "erf", "relu").
    - `T`: Temperature parameter.
    - `model_type`: Type of model (default is "FC" for fully connected, choose "CNN" for convolutional).
  - **Returns**: An instance of the RKGP model.

The RKGP of a `1HL` fully-connected neural network with `N1 = 500` units at `T = 0.1` in the hidden layer is implemented as follows : 
```
gp = dp.RKGP(L = 1, N1 = 500, T = 0.1)
```
#### Second way to define your models
The `RKGP` function is a shortcut for the submodule of deepbays that handles computations in the most efficient way. The avaliable single modules are:
```
import deepbays.rkgp as gp

gp1hl = gp.FC_1HL() #equivalent to RKGP()

gp1hl_spectral = gp.FC_1HL_spectral() #equivalent to RKGP()

gp_deep = gp.FC_deep() #equivalent to RKGP()

gp_multitask = gp.FC_multitask() #equivalent to RKGP()
```

### Main methods of RKGP class

- **`preprocess(self, X, Y)`**: Preprocesses the input data and labels. Computes kernels that will be useful for prediction.
- **`minimizeAction(self, x0=1)`**: Minimizes the action for the GP model.
- **`computePrediction(self, test_data)`**: Computes predictions for the test set.
- **`computeAverageLoss(self, test_data, test_labels)`**: Computes the generalization error, its bias, and variance decomposition.

## Tasks
The `tasks` module includes various datasets used to test the theoretical models implemented in `deepbays`. Users can also use their own datasets, provided they follow the specified format.

**Available Datasets**

- **mnist_dataset**: Class for the MNIST dataset.
- **cifar10_dataset**: Class for the CIFAR-10 dataset.
- **emnistABFL_CHIS**: Class for the EMNIST dataset with ABFL and CHIS configurations.
- **emnistABEL_CHJS**: Class for the EMNIST dataset with ABEL and CHJS configurations.
- **synthetic_1hl_dataset**: Class for generating synthetic datasets with one hidden layer.
- **perceptron_dataset**: Class for generating synthetic perceptron datasets.

**Key Classes and Methods**

Each dataset class typically includes the following methods:

- `__init__(self, N0, classes, seed)`: Initializes the dataset with the specified parameters.
    
    - **Parameters**:
        - `N0`: Input dimension of the dataset.
        - `classes`: Tuple specifying the classes of interest.
        - `seed`: Seed for reproducibility.
    - Example:
        
        python
        
        Copia codice
        
        `data_class = db.tasks.mnist_dataset(N0=784, classes=(0, 1), seed=1234)`
        
- `make_data(self, P, Pt)`: Generates the data and labels for training and testing.
    
    - **Parameters**:
        - `P`: Number of training examples.
        - `Pt`: Number of test examples.
    - **Returns**: Tuple containing training data, training labels, test data, and test labels.
    - Example:
        
        python
        
        Copia codice
        
        `data, labels, test_data, test_labels = data_class.make_data(P=5, Pt=5)`
## Usage

Below is an example of how to use the `deepbays` package to implement a renormalized kernel Gaussian process (RKGP) model using the MNIST dataset.

python

Copia codice

```python
import deepbays as db

# Specifications for the network model
N1 = 5000
l0 = 1.0
l1 = 1.0
act = "erf"
T = 0.01

# Dataset specifications
N0 = 784 # input dimension
task = "mnist" # dataset
classes = (0, 1) # classes of interest
seed = 1234 # reproducibility seed
P = 5
Pt = 5

dataclass = eval(f"db.tasks.{task}_dataset({N0},({classes[0]},{classes[1]}), {seed})")
data, labels, test_data, test_labels = dataclass.make_data(P, Pt)

# Define the RKGP model
gp = db.RKGP(N1, l0, l1, act, T, 1, "FC")
gp.preprocess(data, labels)
gp.minimizeAction(x0=1.0)
print(f"optQ is {gp.optQ}")

yPred = gp.computePrediction(test_data)
testLoss, bias, var = gp.computeAverageLoss(test_data, test_labels)
print("average test loss on testset:", testLrr, "average bias on testset:", bias, "average variance on testset:", var)

```
