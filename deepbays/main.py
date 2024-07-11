import numpy as np
from typing import Union, List
from scipy.optimize import minimize
from . import rkgp

def RKGP(L: int, 
         widths: Union[int, List[int]], 
         D: int, 
         T: float, 
         priors: List[float] = [], 
         act: str = "erf", 
         model_type: str = "FC", 
         kernel_type: str = "best"):
    """
    Constructs a Gaussian Process model.

    Parameters:
    - L (int): Number of hidden layers. Must be an integer.
    - widths (int): Size of the hidden layer. Must be an integer.
    - D (int): Number of network outputs. Must be an integer greater than 0.
    - T (float): Temperature parameter.
    - priors (list of float, optional): Gaussian priors for each layer. Defaults to None.
    - act (str, optional): Activation function. Defaults to "erf".
    - model_type (str, optional): Type of model. Defaults to "FC".
    - kernel_type (str, optional): Kernel type. Defaults to "best". Setting kernel_type = "vanilla" does not implement first order corrections to the proportional regime

    Returns:
    - model: The constructed RKGP model that is equivalent to the specified network.

    Raises:
    - ValueError: If L is not an integer or is less than 1.
    - ValueError: If widths is not an integer.
    - ValueError: If D is not an integer or is less than 1.
    - ValueError: If kernel_type is not "best" or "vanilla".
    """
    
    morespecs = ""
    assert (isinstance(L, int) and L > 0 ), "Number of layers must be integer and greater than 0."
    initPriors = np.ones(L)

    if isinstance(widths, int): ## if N1 is integer and L is greater than 1, assign N1 neurons to each layer.
        assert widths > 0 , "Size of hidden layer must be greater than 0."
        widths = [widths for _ in range(L)]
    else:
        assert len(widths) == L, "Widths must match number of hidden layers." 

    if L == 1: 
        netName = "1HL"
        if len(priors) != 0:
            assert len(priors) == 2, "For a 1HL network we need two gaussian priors."
            l0, l1 = priors[0], priors[1]
        else:
            l0, l1 = initPriors
        assert (isinstance(widths, int) or (len(widths) == 1 and isinstance(widths[0], int))), "Size of hidden layer must be integer."
        if isinstance(widths, int):
            N1 = widths
        else: 
            N1 = widths[0]
        assert (isinstance(D, int) and D>=1), "Number of outputs must be integer and greater than 0."
        if D == 1: 
            args = [N1, T, l0, l1, act]
        else: 
            morespecs = "_multiclass"
            args = [N1, D, T, l0, l1, act]
    elif L > 1: 
        netName = "deep"
        l0, l1 = priors[0], priors[1] # each layer has prior l0 except for the last layer which has prior lambda1 ___ TO CHANGE 
        args = [L, widths, T, l0, l1, act]
    if kernel_type == "best":
        if netName == "1HL" and D == 1 :
            morespecs = morespecs + "_corrected"
    elif kernel_type != "vanilla": 
        raise ValueError("Invalid kernel type.")

    model = eval(f"rkgp.{model_type}_{netName}{morespecs}")
    return model(*args)
