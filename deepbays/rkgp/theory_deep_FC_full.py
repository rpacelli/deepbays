import numpy as np
from scipy.optimize import fsolve
from .. import kernels
import torch

class FC_deep_full():
    def __init__(self, 
                 L      : int, 
                 N1     : int, 
                 T      : float, 
                 priors : list = [1.0, 1.0], 
                 act    : str = "erf"):
        self.N1, self.T, self.L = N1, T, L
        self.l1 = np.array(priors)
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        
    def effectiveAction(self, Q):
        C = torch.tensor(self.C, dtype = torch.float64, requires_grad=True)
        orderParam = Q / self.l1
        #rKL = torch.zeros(self.P, self.P, dtype = torch.float64, requires_grad=True)
        self.kR = eval(f"fR_{self.L}")
        rKL = self.kR(C.diagonal()[:,None], C, C.diagonal()[None,:], *Q)
            #orderParam = Q[l] / self.l1[l]
        A = rKL +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return ( torch.sum(Q - torch.log(Q))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                 )
    
    def computeKernelGrad(self, cxx,cxy,cyy, *args):
        cxx = torch.tensor(cxx, dtype = torch.float64, requires_grad = True)
        cxy = torch.tensor(cxy, dtype = torch.float64, requires_grad = True)
        cyy = torch.tensor(cyy, dtype = torch.float64, requires_grad = True)
        z = self.fR_l(cxx,cxy,cyy, *args)
        grad = torch.autograd.grad(outputs=z, inputs=(cxx,cxy,cyy),
                                    grad_outputs=torch.ones_like(z),
                                    create_graph=True)
        #f = self.fR(cxx, cxy, cyy)
        #f.backward()
        return grad

    def fR_1(self,cxx,cxy,cyy,Q1): 
        return Q1 * self.kernelTorch(cxx, cxy, cyy)
    
    def fR_2(self, cxx,cxy,cyy,Q1,Q2): 
        k_xx = self.kernelTorch(cxx, cxx, cxx)
        k_yy = self.kernelTorch(cyy, cyy, cyy)
        k_xy = self.kernelTorch(cxx, cxy, cyy)
        self.fR_l = self.fR_1
        d_mm, d_mn, d_nn = self.computeKernelGrad(k_xx, k_xy, k_yy)
        return self.fR_l(k_xx, k_xy, k_yy, Q1) + (Q2-1)* (d_mm* k_xx + d_nn* k_yy + d_mn* k_xy )

    def fR_3(self, cxx,cxy,cyy, Q1,Q2,Q3): 
        k_xx = self.kernelTorch(cxx, cxx, cxx)
        k_yy = self.kernelTorch(cyy, cyy, cyy)
        k_xy = self.kernelTorch(cxx, cxy, cyy)
        self.fR_l = self.fR_2
        d_mm, d_mn, d_nn = self.computeKernelGrad(k_xx, k_xy, k_yy)
        return self.fR_l(k_xx, k_xy, k_yy, Q1, Q2) + (Q3-1) * (d_mm* k_xx + d_nn* k_yy + d_mn* k_xy )
    
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
        self.corrNorm = 1/(self.N0 * self.l1[0])
        self.C = np.dot(X, X.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, X, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, X.T) * self.corrNorm
    
    def predict(self, Xtest):
        self.computeTestsetKernels(self.X, Xtest)
        rKL = self.C
        rK0L = self.C0 
        rK0XL = self.C0X
        for l in range(self.L):
            orderParam = self.optQ[l] / self.l1[l]
            rKXL = rKL.diagonal() 
            rK0XL = orderParam * self.kernel(rK0L[:,None], rK0XL, rKXL[None, :])
            rK0L = orderParam * self.kernel(rK0L, rK0L, rK0L)
            rKL = orderParam * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        A = rKL + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XL, invK)
        self.rK0L = rK0L
        self.K0_invK = K0_invK
        self.rK0XL = rK0XL
        self.Ypred = np.dot(K0_invK, self.Y)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred  
        var = self.rK0L - np.sum(self.K0_invK * self.rK0XL, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()