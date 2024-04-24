import numpy as np
from scipy.optimize import minimize
from .kernels import  kernel_erf, kernel_relu, kernel_id, computeKmatrix
import time

class prop_width_GP_1HL_FC():
    def __init__(self, N1, l0, l1, act, T):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernel_{act}")

    def effectiveAction(self,x):
        A = self.T * np.identity(self.P) + (x/self.l1) * self.diagK
        invA = np.linalg.inv(A)
        return ( x - np.log(x)
            + (1/self.N1) * np.sum(np.log(self.T + x*self.eigvalK/self.l1))
            + (1/self.N1) * np.dot(self.yT, np.dot(invA, self.yT)) )

    def minimizeAction(self):
        x0 = 1.0   #initial condition and bounds
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,np.inf),) , tol=1e-20)
        self.optQ = (optQ.x).item()
        print(f"optQ is {self.optQ}")

    def preprocess(self, data, labels):
        self.P, self.N0 = data.shape
        self.corrNorm = 1/(self.N0*self.l0)
        self.C = np.dot(data, data.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.K = computeKmatrix(self.C, self.kernel)
        [self.eigvalK, eigvecK] = np.linalg.eig(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, labels.squeeze())

    def computeTestsetKernels(self, data, testData):
        self.Ptest = len(testData)
        self.C0 = np.dot(testData, testData.T).diagonal() * self.corrNorm
        self.C0X = np.dot(testData, data.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def computeAveragePrediction(self, data, labels, testData, testLabels):
        
        self.computeTestsetKernels(data, testData)
        self.orderParam = self.optQ / self.l1
        A = self.orderParam * self.K + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        rK0 = self.orderParam * self.K0 
        rK0X = self.orderParam * self.K0X
        K0_invK = np.matmul(rK0X, invK)
        bias = testLabels - np.dot(K0_invK, labels) 
        var = rK0 - np.sum(K0_invK * rK0X, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item()
