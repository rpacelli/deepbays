import numpy as np
from scipy.optimize import fsolve
from .. import kernels
import torch

class FC_2HL_full():
    def __init__(self, 
                 L      : int, 
                 N1     : int, 
                 T      : float, 
                 priors : list = [1.0, 1.0], 
                 act    : str = "erf"):
        self.N1, self.T, self.L, self.l0, self.l1 = N1, T, L, priors[0],priors[1:] 
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernelTorch_derivative_xx = eval(f"kernels.kernel_{act}_derivative_xx_torch")
        self.kernelTorch_derivative_yy = eval(f"kernels.kernel_{act}_derivative_yy_torch")
        self.kernelTorch_derivative_xy = eval(f"kernels.kernel_{act}_derivative_xy_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        
    def effectiveAction(self, Q):
        rKL = Q[1]*self.K_det + Q[1]*Q[0]*self.K_wis 
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

    def optimize(self, Q0 = 1.):
        if isinstance(Q0, float):
            Q0 = np.ones(self.L)
        self.optQ = fsolve(self.computeActionGrad, Q0, xtol = 1e-8)
        isClose = np.isclose(self.computeActionGrad(self.optQ), np.zeros(self.L)) 
        self.converged = isClose.all()
        #print("\nis exact solution close to zero?", isClose)   
        #print(f"{self.L} hidden layer optQ is {self.optQ}")

    def setIW(self):
        self.optQ = np.ones(self.L)

    def preprocess(self, X, Y):
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.corrNorm = 1/(self.N0 * self.l0)

        self.C = np.dot(X, X.T) * self.corrNorm
        self.C = torch.tensor(self.C, dtype = torch.float64, requires_grad = False)

        self.K_NNGP_1 = (1. / self.l1[0]) * self.kernelTorch( self.C.diagonal()[:,None], self.C, self.C.diagonal()[None,:] )
        self.K_NNGP_2 = (1. / self.l1[1]) * self.kernelTorch( self.K_NNGP_1.diagonal()[:,None], self.K_NNGP_1, self.K_NNGP_1.diagonal()[None,:] )

        self.F_xx = self.kernelTorch_derivative_xx( self.K_NNGP_1.diagonal()[:,None], self.K_NNGP_1, self.K_NNGP_1.diagonal()[None,:] )
        self.F_yy = self.kernelTorch_derivative_yy( self.K_NNGP_1.diagonal()[:,None], self.K_NNGP_1, self.K_NNGP_1.diagonal()[None,:] )
        self.F_xy = self.kernelTorch_derivative_xy( self.K_NNGP_1.diagonal()[:,None], self.K_NNGP_1, self.K_NNGP_1.diagonal()[None,:] )

        self.K_det = self.K_NNGP_2 - (1. / self.l1[1]) * ( self.F_xx * self.K_NNGP_1.diag().unsqueeze(1) + self.F_yy * self.K_NNGP_1.diag().unsqueeze(0) + self.F_xy * self.K_NNGP_1  )
        self.K_wis = (1. / self.l1[1]) * ( self.F_xx * self.K_NNGP_1.diag().unsqueeze(1) + self.F_yy * self.K_NNGP_1.diag().unsqueeze(0) + self.F_xy * self.K_NNGP_1  )

        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, X, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, X.T) * self.corrNorm
    
    def predict(self, Xtest):
        self.computeTestsetKernels(self.X, Xtest)
        self.C0X = torch.tensor(self.C0X, dtype = torch.float64, requires_grad = False)
        self.C0 = torch.tensor(self.C0, dtype = torch.float64, requires_grad = False)

        self.K_NNGP_1_0X = (1. / self.l1[0]) * self.kernelTorch( self.C0[:,None], self.C0X, self.C.diagonal()[None,:] )
        self.K_NNGP_1_0  = (1. / self.l1[0]) * self.kernelTorch( self.C0, self.C0, self.C0 )

        self.K_NNGP_2_0X = (1. / self.l1[1]) * self.kernelTorch( self.K_NNGP_1_0[:,None], self.K_NNGP_1_0X, self.K_NNGP_1.diagonal()[None,:] )
        self.K_NNGP_2_0  = (1. / self.l1[1]) * self.kernelTorch( self.K_NNGP_1_0, self.K_NNGP_1_0, self.K_NNGP_1_0 )

        self.F_xx_0X = self.kernelTorch_derivative_xx( self.K_NNGP_1_0[:,None], self.K_NNGP_1_0X, self.K_NNGP_1.diagonal()[None,:] )
        self.F_yy_0X = self.kernelTorch_derivative_yy( self.K_NNGP_1_0[:,None], self.K_NNGP_1_0X, self.K_NNGP_1.diagonal()[None,:] )
        self.F_xy_0X = self.kernelTorch_derivative_xy( self.K_NNGP_1_0[:,None], self.K_NNGP_1_0X, self.K_NNGP_1.diagonal()[None,:] )

        self.F_xx_0 = self.kernelTorch_derivative_xx( self.K_NNGP_1_0, self.K_NNGP_1_0, self.K_NNGP_1_0 )
        self.F_yy_0 = self.kernelTorch_derivative_yy( self.K_NNGP_1_0, self.K_NNGP_1_0, self.K_NNGP_1_0 )
        self.F_xy_0 = self.kernelTorch_derivative_xy( self.K_NNGP_1_0, self.K_NNGP_1_0, self.K_NNGP_1_0 )


        self.K_det_0X = self.K_NNGP_2_0X - (1. / self.l1[1]) * ( self.F_xx_0X * self.K_NNGP_1_0.unsqueeze(1) + self.F_yy_0X * self.K_NNGP_1.diag().unsqueeze(0) + self.F_xy_0X * self.K_NNGP_1_0X )


        self.K_wis_0X = (1. / self.l1[1]) * ( self.F_xx_0X * self.K_NNGP_1_0.unsqueeze(1) + self.F_yy_0X * self.K_NNGP_1.diag().unsqueeze(0) + self.F_xy_0X * self.K_NNGP_1_0X )

        self.K_det_0 = self.K_NNGP_2_0 - (1. / self.l1[1]) * ( self.F_xx_0 * self.K_NNGP_1_0 + self.F_yy_0 * self.K_NNGP_1_0 + self.F_xy_0 * self.K_NNGP_1_0  )

        self.K_wis_0 = (1. / self.l1[1]) * ( self.F_xx_0 * self.K_NNGP_1_0 + self.F_yy_0 * self.K_NNGP_1_0 + self.F_xy_0 * self.K_NNGP_1_0  )

        rKL   = self.optQ[1]*self.K_det    + self.optQ[1]*self.optQ[0]*self.K_wis 
        rK0XL = self.optQ[1]*self.K_det_0X + self.optQ[1]*self.optQ[0]*self.K_wis_0X 
        rK0L  = self.optQ[1]*self.K_det_0  + self.optQ[1]*self.optQ[0]*self.K_wis_0 

        A = rKL + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XL, invK)
        self.rKL = rKL
        self.rK0L = rK0L
        self.K0_invK = K0_invK
        self.rK0XL = rK0XL
        self.Ypred = np.dot(K0_invK, self.Y)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred  
        var = self.rK0L - torch.sum(self.K0_invK * self.rK0XL, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
    
