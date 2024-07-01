import numpy as np
from scipy.optimize import minimize
from . import rkgp
import time

#L needs to be the first parameter

def RKGP(N1, l0, l1, act, T, L, model_type = "FC"):
    if L == 1 :
        netName = "1HL"
    elif L > 1: 
        netName = "deep"
    model = eval(f"rkgp.{model_type}_{netName}")
    return model(N1,l0,l1,act,T)
