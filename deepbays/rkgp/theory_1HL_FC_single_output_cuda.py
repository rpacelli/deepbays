import cupy as cp
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
from cupy.cuda import Device
import torch

class FC_1HL_cuda():
    def __init__(self, 
                 N1   : int, 
                 T    : float, 
                 l0   : float = 1.0,
                 l1   : float = 1.0,
                 act  : str = "erf", 
                 bias : bool = False,
                 device_id: int = 0):
        self.device = Device(device_id)
        self.N1, self.l0, self.l1, self.T = N1, l0, l1, T
        self.kernel = eval(f"kernels.kernel_{act}_cuda")
        if bias: 
            self.kernel = eval(f"kernels.kernel_{act}_bias_cuda")

    def effectiveAction(self, x):
        with self.device:
            A = self.T * cp.identity(self.P) + (x/self.l1) * self.diagK
            invA = cp.linalg.inv(A)
            return ( x - cp.log(x)
                + (1/self.N1) * cp.sum(cp.log(self.T + x * self.eigvalK / self.l1))
                + (1/self.N1) * cp.dot(self.yT, cp.dot(invA, self.yT)) )

    def optimize(self, x0 = 1.): # x0 is initial condition
        with self.device:
            optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,cp.inf),) , tol=1e-12)
            self.optQ = (optQ.x).item()
            assert self.optQ > 0 , "Unphysical solution found (Q is negative)."

    def setIW(self):
        self.optQ = 1

    def preprocess(self, Xtrain, Ytrain):
        with self.device:
            self.Xtrain = cp.asarray(Xtrain)
            self.Ytrain = cp.asarray(Ytrain)
            self.P, self.N0 = self.Xtrain.shape
            self.corrNorm = 1/(self.N0*self.l0)
            self.C = cp.dot(self.Xtrain, self.Xtrain.T) * self.corrNorm        
            #self.Xtrain = Xtrain
            #self.Ytrain = Ytrain
            self.CX = self.C.diagonal()
            self.K = kernels.computeKmatrix(self.C, self.kernel)
            self.eigvalK, eigvecK = cp.linalg.eigh(self.K)
            self.diagK = cp.diagflat(self.eigvalK)
            self.Udag = eigvecK.T
            self.yT = cp.matmul(self.Udag, self.Ytrain.squeeze())

    def computeTestsetKernels(self, Xtest):
        with self.device:
            self.Ptest = len(Xtest)
            self.Xtest = cp.asarray(Xtest)
            self.C0 = cp.dot(self.Xtest, self.Xtest.T).diagonal() * self.corrNorm
            self.C0X = cp.dot(self.Xtest, self.Xtrain.T) * self.corrNorm
            self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
            self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def predict(self, Xtest):
        with self.device:
            self.computeTestsetKernels(Xtest)
            self.orderParam = self.optQ / self.l1
            A = self.orderParam * self.K + (self.T) * cp.eye(self.P)
            invK = cp.linalg.inv(A)
            self.rK0X = self.orderParam * self.K0X
            self.K0_invK = cp.matmul(self.rK0X, invK)
            self.Ypred =  cp.dot(self.K0_invK, self.Ytrain)
            return self.Ypred
    
    def averageLoss(self, Ytest):
        with self.device:
            self.Ytest = cp.asarray(Ytest)
            self.rK0 = self.orderParam * self.K0 
            bias = self.Ytest - self.Ypred 
            var = self.rK0 - cp.sum(self.K0_invK * self.rK0X, axis=1)
            predLoss = bias**2 + var 
            return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()

## FROM HERE  
class FC_1HL_corrected():
    def __init__(self, N1, T, l0 = 1.0, l1 = 1.0, act = "erf"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernels.kernel_{act}")

    def effectiveAction(self, x):
        tildeK = self.T * cp.identity(self.P) + (x/self.l1) * self.diagK
        invA = cp.linalg.inv(tildeK)
        return ( x - cp.log(x)
            + (1/self.N1) * cp.sum(cp.log(self.T + x * self.eigvalK / self.l1))
            + (1/self.N1) * cp.dot(self.yT, cp.dot(invA, self.yT)) )

    def optimize(self, x0 = 1.): # x0 is initial condition
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,cp.inf),) , tol=1e-12)
        self.optQ = (optQ.x).item()
        assert self.optQ > 0 , "Unphysical solution found (Q is negative)."
    
    def setIW(self):
        self.optQ = 1

    def preprocess(self, Xtrain, Ytrain):
        self.P, self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.N0*self.l0)
        self.C = cp.dot(Xtrain, Xtrain.T) * self.corrNorm
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.CX = self.C.diagonal()
        self.K = kernels.computeKmatrix(self.C, self.kernel)
        self.eigvalK, eigvecK = cp.linalg.eig(self.K)
        self.diagK = cp.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = cp.matmul(self.Udag, Ytrain.squeeze())

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = cp.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = cp.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        tildeK = self.orderParam * self.K + (self.T) * cp.eye(self.P)
        invK = cp.linalg.inv(tildeK)
        self.corrTildeK = self.orderParam * self.K + (self.T) * cp.eye(self.P)
        #self.tildeY = cp.matmul(invK, cp.outer(self.Ytrain, self.Ytrain))
        self.tildeY = cp.matmul(invK, cp.mutliply(self.Ytrain[:, None], self.Ytrain[:, None]))
        self.corrToK = cp.matmul(cp.matmul(self.tildeY -cp.eye(self.P), invK ), self.K)
        self.corrK = cp.matmul(self.K, cp.eye(self.P)+ self.orderParam * self.corrToK/self.N1)
        self.corrK0X =  cp.matmul(self.K0X, cp.eye(self.P)+ self.orderParam * self.corrToK/self.N1)
        self.corrToK0 = cp.matmul( invK , self.K0X.T)
        self.corrK0 = self.K0 + (self.orderParam/self.N1) * (-cp.matmul(self.K0X, self.corrToK0)+cp.matmul(self.K0X, cp.matmul(self.tildeY,self.corrToK0)))
        self.corrTildeK = self.orderParam * self.corrK + (self.T) * cp.eye(self.P)
        invCorrK = cp.linalg.inv(self.corrTildeK)
        self.rK0X = self.orderParam * self.corrK0X
        self.K0_invK = cp.matmul(self.rK0X, invCorrK)
        self.Ypred =  cp.dot(self.K0_invK, self.Ytrain)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        self.rK0 = self.orderParam * self.corrK0 
        bias = Ytest - self.Ypred 
        var = self.rK0 - cp.sum(self.K0_invK * self.rK0X, axis=1)
        predLoss = bias**2 + var 
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
    

class FC_1HL_nonodd(): # the order parameter associated with average norm is called x, the other one si called Bx
    def __init__(self, N1, T, l0 = 1.0, l1 = 1.0, act = "relu"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernels.kernel_{act}")
        self.mean = eval(f"kernels.mean_{act}")

    def effectiveAction(self, Q):
        x = Q[0]
        Bx = Q[1]
        rK = self.K * x/self.l1
        rKm = (x - (1/(1+Bx)))/self.l1 
        A = rK - rKm +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return ( - x*Bx 
                + torch.log(1 + Bx)
                + (1/self.N1) * torch.matmul(self.Ytrain, torch.matmul(invA, self.Ytrain)) 
                + (1/self.N1) * torch.logdet(A)
                 )

    def computeGrad(self, x):
        xTemp = torch.tensor(x, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(xTemp)
        f.backward(retain_graph=True)
        return xTemp.grad.detach().numpy()   
    
    def preprocess(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.P , self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.l0* self.N0)
        self.Ytrain = torch.tensor(Ytrain.squeeze()).type('torch.DoubleTensor')
        #self.Ytrain.requires_grad = False
        self.C = cp.dot(Xtrain,Xtrain.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.K = torch.tensor(kernels.computeKmatrix(self.C, self.kernel), dtype = torch.float64, requires_grad = False)
        self.Km = torch.tensor(self.mean(self.CX), dtype = torch.float64, requires_grad = False)
        #Km =  torch.tensor(kernel(m), dtype = torch.float64, requires_grad = False) 

    def optimize(self, x0 = [1,0.3]): # the order parameter associated with average norm is called x, the other one si called Bx
        assert len(x0) == 2, "I need initial conditions for two variables. Choose a list with lenght = 2."
        self.optQ = fsolve(self.computeGrad, x0, xtol = 1e-8)
        self.Q = self.optQ[0]
        self.bQ = self.optQ[1]
        assert self.Q > 0, "Unphysical solution found (Q is negative)."
        #print(f"\nPoint where the gradient is approximately zero: {self.optQ}")
        isClose = cp.isclose(self.computeGrad(self.optQ), [0.0, 0.0]) 
        self.converged = isClose.all()
        assert self.converged, "Wrong solution found, gradient is not 0. Try changing initial condition."
        #print("\nis exact solution close to zero?", isClose) 
    
    def setIW(self):
        self.Q = 1.
        self.bQ = 0.
    
    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = cp.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = cp.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0)
        self.Km0 = self.mean(self.C0)
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        rK = self.K * self.Q/self.l1
        rKm = (self.Q - (1/(1+self.bQ))) * cp.outer(self.Km, self.Km)/self.l1
        A = rK - rKm +  self.T * torch.eye(self.P) 
        invK = cp.linalg.inv(A)
        rK0X = self.K0X * self.Q /self.l1
        rK0Xm = (self.Q - (1/(1+self.bQ))) * cp.outer(self.Km, self.Km0) /self.l1
        self.rK0X = rK0X - rK0Xm
        self.K0_invK = cp.matmul(self.rK0X, invK)
        self.Ypred =  cp.dot(self.K0_invK, self.Ytrain)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        rK0 = self.K0 * self.Q/self.l1 
        rK0m = (self.Q - (1/(1+self.bQ))) * cp.outer(self.Km0, self.Km0) /self.l1
        self.rK0 = rK0 - rK0m
        bias = Ytest - self.Ypred 
        var = self.rK0 - cp.sum(self.K0_invK * self.rK0X, axis=1)
        predLoss = bias**2 + var 
        return predLoss.mean().item(), (bias**2).mean().item(), var 