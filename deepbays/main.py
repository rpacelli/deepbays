import numpy as np
from scipy.optimize import minimize
from . import rkgp
import time

#L needs to be the first parameter

def RKGP(L, widths, D, priors, T, act = "erf", model_type = "FC"):
    morespecs = ""
    if not isinstance(L, (int)):
        raise ValueError("Number of layers must be integer")
    else:  
        if L == 1: 
            netName = "1HL"
            assert len(priors) == 2, "For a 1HL network we need two gaussian priors."
            assert (isinstance(widths, int) or (len(widths) == 1 and isinstance(widths[0], int))), "Size of hidden layer must be integer."
            if isinstance(widths, int):
                N1 = widths
            else: 
                N1 = widths[0]
            l0 = priors[0]
            l1 = priors[1]
            assert (isinstance(D, int) and D>=1), "Number of outputs must be integer and greater than 0."
            if D == 1: 
                args = [N1, l0, l1, act, T]
            else: 
                morespecs = "_multiclass"
                args = [N1, l0, l1, act, T, D]
        elif L > 1: 
            netName = "deep"
        else: 
            raise ValueError("Cannot build a network with 0 hidden layers.")

    model = eval(f"rkgp.{model_type}_{netName}{morespecs}")
    print(model)
    return model(*args)
