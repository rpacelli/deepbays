import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
import torch

class FC_1HL_nonodd():
    def __init__(self, N1, T, l0 = 1.0, l1 = 1.0, act = "relu"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.L = 1
        self.kernel = eval(f"kernels.kernel_{act}")
        self.mean = eval(f"kernels.mean_{act}")

    def preprocess(self, X, Y):
        self.X = X
        self.Y = Y.squeeze().clone().detach().double()
        self.P, self.N0 = X.shape
        self.corrNorm = 1/(self.N0 * self.l0)
        self.C = np.dot(X, X.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False
        self.K = torch.tensor(kernels.computeKmatrix(self.C, self.kernel), dtype = torch.float64, requires_grad = False)
        self.Km = torch.tensor(self.mean(self.CX), dtype = torch.float64, requires_grad = False)
        self.M = torch.outer(self.Km, self.Km)

    def optimize(self, x0 = [1., 0.], lr=0.001, tolerance = 1e-7, max_epochs = 5000, maxCheck = 50, reg = 1000.0, verbose=True):

        Q0 = torch.tensor(x0)
        Q = Q0.clone().detach().requires_grad_(True)
        self.alpha = self.P / self.N1
        def S(Qvec):
            KR = (Q[0]/self.l1)*self.K - (Q[0]-1/(1+Q[1]))*self.M / self.l1
            gpKernel = self.T*torch.eye(self.P) + KR
            gpKernel_inv = torch.linalg.inv(gpKernel)
            dgp_dQ = - self.M / (self.l1 * (1+Q[1])**2)
            dgp_dbarQ = self.K / self.l1 - self.M / self.l1
            term1 = - Q[1] + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dbarQ)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dbarQ, torch.matmul( gpKernel_inv , self.Y ))) )
            term2 = - Q[0] + 1/(1+Q[1]) + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dQ)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dQ, torch.matmul( gpKernel_inv , self.Y ))) )
            return term1**2 + term2**2
        optimizer = torch.optim.Adam([Q], lr)
        previous_loss = 100.
        self.optState = False
        self.optEpochs = 0
        check = 0
        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = S(Q)
            loss.backward()
            optimizer.step()
            loss_change = abs(loss.item() - previous_loss)
            if (loss_change < tolerance):
                check += 1
            if (loss_change < tolerance) and (check > maxCheck):
                self.optState = True
                break
            previous_loss = loss.item()
            self.optEpochs +=1
        self.optQ = Q.detach().numpy()
        if verbose: print(f"opt state: {self.optState}, checks: {check}, epochs: {self.optEpochs}, Qs: {self.optQ}")
    
    def setIW(self):
        self.optQ[0] = 1.
        self.optQ[1] = 0.
    
    def computeTestsetKernels(self, X, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, X.T) * self.corrNorm
        self.Km0 = torch.tensor(self.mean(self.C0), dtype = torch.float64, requires_grad = False)
        self.MXX = self.M.detach().numpy()
        self.M0X = torch.outer(self.Km0, self.Km).detach().numpy()
        self.M0 = torch.outer(self.Km0, self.Km0).diagonal().detach().numpy()
    
    def predict(self, Xtest):
        self.computeTestsetKernels(self.X, Xtest)
        rKL = self.C
        rK0L = self.C0 
        rK0XL = self.C0X

        rKXL = rKL.diagonal() 

        rKL = (self.optQ[0] / self.l1) * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:]) - (self.optQ[0]-1/(1+self.optQ[1])) * self.MXX / self.l1
        rK0XL = (self.optQ[0] / self.l1) * self.kernel(rK0L[:,None], rK0XL, rKXL[None, :]) - (self.optQ[0]-1/(1+self.optQ[1])) * self.M0X / self.l1
        rK0L = (self.optQ[0] / self.l1) * self.kernel(rK0L, rK0L, rK0L) - (self.optQ[0]-1/(1+self.optQ[1])) * self.M0 / self.l1

        A = rKL + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XL, invK)
        self.rK0L = rK0L
        self.K0_invK = K0_invK
        self.rK0XL = rK0XL
        self.Ypred = np.dot(K0_invK, self.Y).reshape(-1, 1)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred  
        var = self.rK0L - np.sum(self.K0_invK * self.rK0XL, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()