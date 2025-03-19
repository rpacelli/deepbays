import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
import torch
 
class FC_1HL_corrected():
    def __init__(self, N1, T, l0 = 1.0, l1 = 1.0, act = "erf"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernels.kernel_{act}")

    def effectiveAction(self, x):
        tildeK = self.T * np.identity(self.P) + (x/self.l1) * self.diagK
        invA = np.linalg.inv(tildeK)
        return ( x - np.log(x)
            + (1/self.N1) * np.sum(np.log(self.T + x * self.eigvalK / self.l1))
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
        self.eigvalK, eigvecK = np.linalg.eig(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, Ytrain.squeeze())

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        tildeK = self.orderParam * self.K + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(tildeK)
        self.corrTildeK = self.orderParam * self.K + (self.T) * np.eye(self.P)
        self.tildeY = np.matmul(invK, np.outer(self.Ytrain, self.Ytrain))
        self.corrToK = np.matmul(np.matmul(self.tildeY -np.eye(self.P), invK ), self.K)
        self.corrK = np.matmul(self.K, np.eye(self.P)+ self.orderParam * self.corrToK/self.N1)
        self.corrK0X =  np.matmul(self.K0X, np.eye(self.P)+ self.orderParam * self.corrToK/self.N1)
        self.corrToK0 = np.matmul( invK , self.K0X.T)
        self.corrK0 = self.K0 + (self.orderParam/self.N1) * (-np.matmul(self.K0X, self.corrToK0)+np.matmul(self.K0X, np.matmul(self.tildeY,self.corrToK0)))
        self.corrTildeK = self.orderParam * self.corrK + (self.T) * np.eye(self.P)
        invCorrK = np.linalg.inv(self.corrTildeK)
        self.rK0X = self.orderParam * self.corrK0X
        self.K0_invK = np.matmul(self.rK0X, invCorrK)
        self.Ypred =  np.dot(self.K0_invK, self.Ytrain)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        self.rK0 = self.orderParam * self.corrK0 
        bias = Ytest - self.Ypred 
        var = self.rK0 - np.sum(self.K0_invK * self.rK0X, axis=1)
        predLoss = bias**2 + var 
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()