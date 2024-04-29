import numpy as np
from scipy.optimize import fsolve
from .kernels import kernel_erf, kernel_relu, kernel_id, computeKmatrix, computeKmatrixTorch, kernel_erf_torch
import torch

class prop_width_GP_deep_FC():
    def __init__(self, N1, l0, l1, act, T, L):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernelTorch = eval(f"kernel_{act}_torch")
        self.kernel = eval(f"kernel_{act}")
        self.L = L
        
    def effectiveAction(self, Q):
        rKL = torch.tensor(self.C, dtype = torch.float64, requires_grad = False)
        for l in range(self.L):
            orderParam = Q[l] / self.l1
            #rKL = orderParam * computeKmatrixTorch(rKL, self.kernelTorch)
            rKL = orderParam * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        A = rKL +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return ( torch.sum(Q - torch.log(Q))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                 )
    
    def computeActionGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(Q)
        f.backward()
        return Q.grad.data.detach().numpy()

    def minimizeAction(self):
        Q0 = np.ones(self.L)*2
        self.optQ = fsolve(self.computeActionGrad, Q0, xtol = 1e-8)
        isClose = np.isclose(self.computeActionGrad(self.optQ), np.zeros(self.L)) 
        self.converged = isClose.all()
        print("\nis exact solution close to zero?", isClose)   
        print(f"{self.L} hidden layer optQ is {self.optQ}")

    def preprocess(self, data, labels):
        self.P, self.N0 = data.shape
        self.corrNorm = 1/(self.N0 * self.l0)
        self.C = np.dot(data, data.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.y = labels.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, data, testData):
        self.Ptest = len(testData)
        self.C0 = np.dot(testData, testData.T).diagonal() * self.corrNorm
        self.C0X = np.dot(testData, data.T) * self.corrNorm
    
    def computeAveragePrediction(self, data, labels, testData, testLabels):
        self.computeTestsetKernels(data, testData)
        rKL = self.C
        rK0L = self.C0 
        rK0XL = self.C0X
        for l in range(self.L):
            orderParam = self.optQ[l] / self.l1
            rKXL = rKL.diagonal() 
            rK0XL = orderParam * self.kernel(rK0L[:,None], rK0XL, rKXL[None, :])
            rK0L = orderParam * self.kernel(rK0L, rK0L, rK0L)
            rKL = orderParam * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        A = rKL + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XL, invK)
        bias = testLabels - np.dot(K0_invK, labels) 
        var = rK0L - np.sum(K0_invK * rK0XL, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item()