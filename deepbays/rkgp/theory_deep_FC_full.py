import numpy as np
from scipy.optimize import fsolve
#from .. import kernels
from deepbays import kernels
import torch

class FC_deep_full():
    def __init__(self, 
                 L      : int, 
                 N1     : int, 
                 T      : float, 
                 priors : list = [1.0, 1.0], 
                 act    : str = "erf"):
        self.N1, self.T, self.L = N1, T, L
        self.priors = np.array(priors)
        if len(self.priors) != L+1: 
            print("\nnumber of priors doesn't match total number of layers (L+1)")
            self.priors = np.ones(L+1)
        self.l0 = self.priors[0]
        self.l1 = self.priors[1:]
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        
    def effectiveAction(self, Q):
        C = torch.tensor(self.C, dtype = torch.float64, requires_grad=False)
        rKL = torch.zeros(self.P, self.P, dtype = torch.float64, requires_grad=False)
        self.kR = eval(f"self.fR_{self.L}")
        for mu in range(self.P):
            for nu in range(self.P): 
                rKL[mu,nu] = self.kR(C[mu,mu], C[mu,nu], C[nu,nu], *Q)
        A = rKL +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return ( torch.sum(Q - torch.log(Q))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                 )
    
    def computeKernelGrad(self, cxx, cxy, cyy, *args):
        cxx = cxx.clone().detach().requires_grad_(True)
        cxy = cxy.clone().detach().requires_grad_(True)
        cyy = cyy.clone().detach().requires_grad_(True)
        z = self.fR_l(cxx,cxy,cyy, *args)
        grad = torch.autograd.grad(outputs=z, inputs=(cxx,cxy,cyy),
                                    grad_outputs=torch.ones_like(z),
                                    create_graph=True, allow_unused=True)
        return grad

    def fR_1(self, cxx, cxy, cyy, Q1): 
        return Q1 * self.kernelTorch(cxx, cxy, cyy) /self.l1[0]
    
    def fR_2(self, cxx,cxy,cyy,Q1,Q2):
        k_xx = self.kernelTorch(cxx, cxx, cxx)/self.l1[0]
        k_yy = self.kernelTorch(cyy, cyy, cyy)/self.l1[0]
        k_xy = self.kernelTorch(cxx, cxy, cyy)/self.l1[0]
        self.fR_l = self.fR_1
        d_mm, d_mn, d_nn = self.computeKernelGrad(k_xx, k_xy, k_yy, Q1)
        return (self.fR_l(k_xx, k_xy, k_yy, Q1) + (Q2-1)* (d_mm* k_xx + d_nn* k_yy + d_mn* k_xy )) /self.l1[1]

    def fR_3(self, cxx,cxy,cyy, Q1,Q2,Q3): 
        k_xx = self.kernelTorch(cxx, cxx, cxx)
        k_yy = self.kernelTorch(cyy, cyy, cyy)
        k_xy = self.kernelTorch(cxx, cxy, cyy)
        self.fR_l = self.fR_2
        d_mm, d_mn, d_nn = self.computeKernelGrad(k_xx/self.l1[0], k_xy/self.l1[0], k_yy/self.l1[0], Q1, Q2)
        self.fR_l = self.fR_2
        return (self.fR_l(k_xx/self.l1[0], k_xy/self.l1[0], k_yy/self.l1[0], Q1, Q2)+ (Q3-1) * (d_mm* k_xx/self.l1[0] + d_nn* k_yy/self.l1[0] + d_mn* k_xy/self.l1[0] ))*self.l1[1]  /self.l1[2]

    def fR_4(self, cxx,cxy,cyy, Q1, Q2, Q3, Q4): 
        k_xx = self.kernelTorch(cxx, cxx, cxx)/self.l1[0]
        k_yy = self.kernelTorch(cyy, cyy, cyy)/self.l1[0]
        k_xy = self.kernelTorch(cxx, cxy, cyy)/self.l1[0]
        self.fR_l = self.fR_3
        d_mm, d_mn, d_nn = self.computeKernelGrad(k_xx, k_xy, k_yy, Q1, Q2, Q3)
        self.fR_l = self.fR_3
        return (self.fR_l(k_xx, k_xy, k_yy, Q1, Q2, Q3) + (Q4 - 1) * (d_mm* k_xx + d_nn* k_yy + d_mn* k_xy ) )*self.l1[2] /self.l1[3]
    
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

    def setIW(self):
        self.kR = eval(f"self.fR_{self.L}")
        self.optQ = torch.tensor(np.ones(self.L)) 

    def preprocess(self, X, Y):
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.corrNorm = 1/(self.N0 * self.l0)
        self.C = np.dot(X, X.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.X.T) * self.corrNorm
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        C = torch.tensor(self.C, dtype = torch.float64, requires_grad=False)
        C0 = torch.tensor(self.C0, dtype = torch.float64, requires_grad=False)
        C0X = torch.tensor(self.C0X, dtype = torch.float64, requires_grad=False)
        rKL = np.zeros(C.shape)
        rK0L = np.zeros(C0.shape) 
        rK0XL = np.zeros(C0X.shape)
        for mu in range(self.P):
            for nu in range(self.P): 
                rKL[mu,nu] = self.kR(C[mu,mu], C[mu,nu], C[nu,nu], *self.optQ)
        for mu in range(self.Ptest):
                rK0L[mu] = self.kR(C0[mu], C0[mu], C0[mu], *self.optQ)
        for mu in range(self.Ptest):
            for nu in range(self.P): 
                rK0XL[mu,nu] = self.kR(C0[mu], C0X[mu,nu], C[nu,nu], *self.optQ)
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