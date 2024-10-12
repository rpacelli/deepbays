import numpy as np
from typing import Union, List
from scipy.optimize import minimize
from . import rkgp

def RKGP(L: int, 
         width: int, 
         D: int, 
         T: float, 
         priors: List[float] = [], 
         act: str = "erf", 
         model_type: str = "FC", 
         mask : int = 1,
         stride : int = 1,
         kernel_type: str = "vanilla"):
    """
    Constructs a Gaussian Process model.

    Parameters:
    - L (int): Number of hidden layers. Must be an integer.
    - width (int): Size of the hidden layer (number of channels if model is convolutional). Must be an integer.
    - D (int): Number of network outputs. Must be an integer greater than 0.
    - T (float): Temperature parameter.
    - priors (list of float, optional): Gaussian priors for each layer. Defaults to None.
    - act (str, optional): Activation function. Defaults to "erf".
    - model_type (str, optional): Type of model. Defaults to "FC".
    - kernel_type (str, optional): Kernel type. Defaults to "best". Setting kernel_type = "vanilla" does not implement first order corrections to the proportional regime

    Returns:
    - model: The constructed RKGP model that is equivalent to the specified network.
    """
    
    morespecs = ""
    assert (isinstance(L, int) and L > 0 ), "Number of layers must be integer and greater than 0."
    initPriors = np.ones(L)

    assert width > 0 , "Size of hidden layer(s) must be greater than 0."
    argsDict = {"depth of the network (L+1)": L+1, "size of hidden layer(s)": width, "number of outputs": D, "temperature": T, "activation function": act}

    if L == 1: 
        netName = "1HL"
        if len(priors) != 0:
            l0, l1 = priors[0], priors[1]
        else:
            l0, l1 = initPriors # if the user choose no priors
        argsDict["Gaussian priors"] = (l0, l1)
        assert (isinstance(D, int) and D>=1), "Number of outputs must be integer and greater than 0."
        if D == 1: 
            args = [width, T, l0, l1, act]
            argNames = {"N1":width, "T": T, "l0": l0, "l1": l1, "act": act}
        else: 
            morespecs = "_multiclass"
            args = [width, D, T, l0, l1, act]
            argNames = {"N1":width, "D": D, "T": T, "l0": l0, "l1": l1, "act": act}
        if model_type == "CONV": 
            assert D == 1,  "No model for shallow convolutional networks with multiple outputs."
            args = [width, T, mask, stride, l0, l1, act ]
            argNames = {"Nc":width, "T": T, "mask":mask, "stride":stride,"l0": l0, "l1": l1, "act": act}
    elif L > 1: 
        argsDict[ "Gaussian priors"] = priors
        netName = "deep"
        assert D == 1, "No model for deep networks with multiple outputs."
        assert model_type != "CONV", "No model for deep convolutional networks."
        args = [L, width, T, priors, act]
        argNames = {"L": L, "N1":width, "T": T, "priors": priors, "act": act}
    if kernel_type == "best":
        assert netName == "1HL", "No model with corrections to proportional regime for deep network."
        assert D == 1, "No model with corrections to proportional regime for networks with multiple outputs."
        assert model_type != "CONV", "No model with corrections to proportional regim for convolutional networks."
        morespecs = morespecs + "_corrected"
    elif kernel_type != "vanilla": 
        raise ValueError("Invalid kernel type.")
    
    if model_type == "CONV":
        argsDict["type of hidden layers"] = "convolutional"
        argsDict["mask"] = mask
        argsDict["stride"] = stride
    elif model_type == "FC":
        argsDict["type of hidden layers"] = "fully-connected"

    for name, arg in argsDict.items():
        print(f"{name}: {arg}")
    argStr = ", ".join([f"{key}={value}" for key, value in argNames.items()])
    print(f"\nmodel name: rkgp.{model_type}_{netName}{morespecs} \nmodel args: {argStr} \n")
    

    model = eval(f"rkgp.{model_type}_{netName}{morespecs}")
    return model(*args)
