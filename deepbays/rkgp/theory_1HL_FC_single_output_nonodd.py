import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
import torch

class FC_1HL_nonodd():
    def __init__(self, 
                 N1   : int, 
                 T    : float, 
                 l0   : float = 1.0,
                 l1   : float = 1.0,
                 act  : str = "erf", 
                 bias : bool = False):
        self.N1, self.l0, self.l1, self.T = N1, l0, l1, T
        self.kernel = eval(f"kernels.kernel_{act}")
        if bias: 
            self.kernel = eval(f"kernels.kernel_{act}_bias")
        self.mean_func = eval(f"kernels.mean_{act}")

    def effectiveAction(self, x):
        A_diag = (self.T + (x/self.l1) * self.eigvalK + (1-x)*np.outer(self.mean, self.mean)/self.l1)
        invA = np.diag(1/A_diag)
        return ( x - np.log(x)
            + (1/self.N1) * np.sum(np.log(self.T + x * self.eigvalK / self.l1 + (1-x)*np.outer(self.mean, self.mean)/self.l1))
            + (1/self.N1) * np.dot(self.yT, np.dot(invA, self.yT)) )

    def optimize(self, x0 = 1.): # x0 is initial condition
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,np.inf),) , tol=1e-12)
        self.optQ = (optQ.x).item()
        assert self.optQ > 0 , "Unphysical solution found (Q is negative)."

    def setIW(self):
        self.optQ = 1

    def preprocess(self, Xtrain, Ytrain):
        self.P, self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.N0*self.l0)
        self.C = np.dot(Xtrain, Xtrain.T) * self.corrNorm
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.CX = self.C.diagonal()
        self.K = kernels.computeKmatrix(self.C, self.kernel) 
        self.eigvalK, eigvecK = np.linalg.eigh(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, Ytrain.squeeze())
        self.mean = self.mean_fun(self.CX)

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
        self.m0 = self.mean_fun(self.C0)
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        A = self.orderParam * self.K + (self.T) * np.eye(self.P) + (1/self.l1-self.orderParam)*np.outer(self.mean, self.mean)
        invK = np.linalg.inv(A)
        self.rK0X = self.orderParam * self.K0X + (1/self.l1 - self.orderParam)* np.outer( self.m0, self.mean)
        self.K0_invK = np.matmul(self.rK0X, invK)
        self.Ypred =  np.dot(self.K0_invK, self.Ytrain)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        self.rK0 = self.orderParam * self.K0 + (1/self.l1 - self.orderParam)* np.outer( self.m0, self.m0)
        bias = Ytest - self.Ypred 
        var = self.rK0 - np.sum(self.K0_invK * self.rK0X, axis=1)
        predLoss = bias**2 + var 
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()