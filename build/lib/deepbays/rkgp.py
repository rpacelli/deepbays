import numpy as np
from scipy.optimize import minimize
from . import rkgp
import time

def RKGP(N1, l0, l1, act, T, L, model_type = "FC"):
    return eval(f"rkgp.RKGP_{L}HL_{model_type}({N1},{l0},{l1},{act},{T})")
