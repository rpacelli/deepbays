import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
import torch

class FC_2HL_nonodd():
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

        self.thetaC = torch.tensor(kernels.computeKmatrix(self.C, self.kernel), dtype = torch.float64, requires_grad = False)
        self.theta_thetaC = torch.tensor(kernels.computeKmatrix(self.thetaC.detach().numpy(), self.kernel), dtype = torch.float64, requires_grad = False)

        self.muC = torch.tensor(self.mean(self.CX), dtype = torch.float64, requires_grad = False)
        self.MC = torch.outer(self.muC, self.muC)
        self.mu_thetaC = torch.tensor(self.mean(self.thetaC.diagonal().numpy()), dtype = torch.float64, requires_grad = False)
        self.M_thetaC = torch.outer(self.mu_thetaC.clone().detach(), self.mu_thetaC.clone().detach())

    def optimize(self, x0 = [1., 0., 1., 0.], lr=0.001, tolerance = 1e-7, max_epochs = 5000, maxCheck = 50, reg = 1000.0, verbose=True):
                            #[Q0, Q1, Q2, Q3] = [barQ_1, Q_1, barQ2, Q2]
        Q0 = torch.tensor(x0)
        Q = Q0.clone().detach().requires_grad_(True)
        self.alpha = self.P / self.N1
        def S(Qvec):

            KR = Q[0] * Q[2] * self.theta_thetaC - (Q[2]-1/(Q[3]+1)) * self.M_thetaC - Q[2] * (Q[0]-1/(Q[1]+1)) * self.M_thetaC
            gpKernel = self.T*torch.eye(self.P) + KR
            gpKernel_inv = torch.linalg.inv(gpKernel)

            dgp_dQ1 = - Q[2] * self.M_thetaC / ((Q[1]+1)**2)
            dgp_dQ2 = - self.M_thetaC / ((Q[3]+1)**2)
            dgp_dQbar1 = Q[2] * self.theta_thetaC - Q[2] * self.M_thetaC
            dgp_dQbar2 = Q[0] * self.theta_thetaC - (1+Q[0]-1/(Q[1]+1)) * self.M_thetaC

            dS_dQ1 = - Q[0] + 1/(Q[1]+1) + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dQ1)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dQ1, torch.matmul( gpKernel_inv , self.Y ))) )
            dS_dQ2 = - Q[2] + 1/(Q[3]+1) + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dQ2)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dQ2, torch.matmul( gpKernel_inv , self.Y ))) )
            dS_dQbar1 = - Q[1] + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dQbar1)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dQbar1, torch.matmul( gpKernel_inv , self.Y ))) )
            dS_dQbar2 = - Q[3] + (self.alpha/self.P) * torch.trace(torch.matmul(gpKernel_inv,dgp_dQbar2)) - (self.alpha/self.P)*torch.matmul( self.Y, torch.matmul(gpKernel_inv, torch.matmul( dgp_dQbar2, torch.matmul( gpKernel_inv , self.Y ))) )

            return dS_dQ1**2 + dS_dQ2**2 + dS_dQbar1**2 + dS_dQbar2**2
        
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
            # print(f"epochs: {i}, loss: {loss}, Qs: {Q}")
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
    
    def predict(self, Xtest):
        self.computeTestsetKernels(self.X, Xtest)
        rKL = self.C
        rK0L = self.C0 
        rK0XL = self.C0X

        rKXL = rKL.diagonal()

        self.thetaC = self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        self.theta_thetaC = self.kernel(self.thetaC.diagonal()[:,None], self.thetaC, self.thetaC.diagonal()[None,:])

        self.thetaC0 = self.kernel(rK0L, rK0L, rK0L)
        self.theta_thetaC0 = self.kernel(self.thetaC0, self.thetaC0, self.thetaC0)

        self.thetaC0X = self.kernel(rK0L[:,None], rK0XL, rKXL[None, :])
        self.theta_thetaC0X = self.kernel(self.thetaC0[:,None], self.thetaC0X, self.thetaC.diagonal()[None, :])

        self.mu_thetaC = self.mean(self.thetaC.diagonal())
        self.mu_thetaC0 = self.mean(self.thetaC0)

        self.M_thetaC = np.outer(self.mu_thetaC, self.mu_thetaC)
        self.M_thetaC0 = np.outer(self.mu_thetaC0, self.mu_thetaC0).diagonal()
        self.M_thetaC0X = np.outer(self.mu_thetaC0, self.mu_thetaC)

        rKL = self.optQ[0] * self.optQ[2] * self.theta_thetaC - (self.optQ[2]-1/(self.optQ[3]+1)) * self.M_thetaC - self.optQ[2] * (self.optQ[0]-1/(self.optQ[1]+1)) * self.M_thetaC

        rK0XL = self.optQ[0] * self.optQ[2] * self.theta_thetaC0X - (self.optQ[2]-1/(self.optQ[3]+1)) * self.M_thetaC0X - self.optQ[2] * (self.optQ[0]-1/(self.optQ[1]+1)) * self.M_thetaC0X

        rK0L = self.optQ[0] * self.optQ[2] * self.theta_thetaC0 - (self.optQ[2]-1/(self.optQ[3]+1)) * self.M_thetaC0 - self.optQ[2] * (self.optQ[0]-1/(self.optQ[1]+1)) * self.M_thetaC0

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