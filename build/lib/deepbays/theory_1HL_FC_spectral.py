import numpy as np
from scipy.optimize import minimize
from .kernels import  kernel_erf, kernel_relu, kernel_id, kernel_tanh, computeKmatrix
import time

class RKGP_1HL_FC_spectral():
    def __init__(self, N1, l0, l1, act, T, theta):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernel_{act}")
        #self.theta = theta #fraction of shut nodes
        self.theta = theta #fraction of open nodes 

    def effectiveAction(self,y): #y is Q in the calculations
        #x = self.theta*(1+y)/(self.theta+(1-self.theta)*np.sqrt(1+y))
        x = self.theta/((1+y)**(1.5)*(1-self.theta + self.theta/np.sqrt(1+y)))
        A = self.T * np.identity(self.P) + (x/self.l1) * self.diagK
        invA = np.linalg.inv(A)
        return (- y*x - 2 * np.log(1-self.theta + (self.theta)/np.sqrt(1+y))
            + (1/self.N1) * np.sum(np.log(self.T + x*self.eigvalK/self.l1))
            + (1/self.N1) * np.dot(self.yT, np.dot(invA, self.yT)) )

    def minimizeAction(self, x0): # x0 is initial condition
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,np.inf),) , tol=1e-20)
        self.optQ = (optQ.x).item()
        self.optbarQ = self.theta/((1+self.optQ)**(1.5)*(1-self.theta + self.theta/np.sqrt(1+self.optQ)))
        print(f"value of S at the saddle point {self.effectiveAction(self.optQ)}")
        print(f"optQ is {self.optQ}")
        print(f"optbarQ is {self.optbarQ}")

    def preprocess(self, Xtrain, Ytrain):
        self.P, self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.N0*self.l0)
        self.C = np.dot(Xtrain, Xtrain.T) * self.corrNorm
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.CX = self.C.diagonal()
        self.K = computeKmatrix(self.C, self.kernel)
        [self.eigvalK, eigvecK] = np.linalg.eig(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, Ytrain.squeeze())

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def computeAveragePrediction(self, Xtest, Ytest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        A = self.orderParam * self.K + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        rK0 = self.orderParam * self.K0 
        rK0X = self.orderParam * self.K0X
        K0_invK = np.matmul(rK0X, invK)
        bias = Ytest - np.dot(K0_invK, self.Ytrain) 
        var = rK0 - np.sum(K0_invK * rK0X, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item()
    
    def computePosteriorEigenvalues(self):
        norm = self.theta/np.sqrt(1+self.optQ) +1-self.theta
        self.p0 = (1-self.theta)/norm
        self.p1 = (self.theta/(np.sqrt(1+self.optQ)))/norm
        print(f"posterior prob of open node: {self.p1}")
        print(f"posterior prob of shut node: {self.p0}")


