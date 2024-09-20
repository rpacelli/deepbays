import numpy as np
from scipy.optimize import fsolve
from .. import kernels
import torch

class CONV_deep():
    def __init__(self, L, N1, T, l0 = 1., l1 = 1., act = "erf", mask = 7, stride = 7):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        self.L = L
        self.mask = mask
        self.corrNorm = 1/(self.mask**2 * self.l0)
        self.stride = stride

    def effectiveAction(self, Q):
        Q = Q.reshape(self.numPatches, self.numPatches)
        rKCumulate = torch.zeros((self.P,self.P), dtype = torch.float64)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                Kij = torch.tensor(self.CXX[i][j])
                Kii = torch.tensor(self.CXX[i][i])
                Kjj = torch.tensor(self.CXX[j][j])
                rKij =  Q[i][j] * kernels.computeKmatrixMultipleCTorch(Kii, Kij, Kjj, self.kernelTorch) / (self.l1*self.numPatches)
                rKCumulate = torch.add(rKCumulate , rKij)
        A = rKCumulate +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return (  torch.trace(Q) -  torch.logdet(Q)
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                 )
    
    def computeActionGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(Q)
        f.backward()
        return Q.grad.data.detach().numpy()

    def minimizeAction(self, Q0 = 1., minimizationTol = 1e-6, nStep = 100000, gradTol = 1e-3):
        if isinstance(Q0, float):
            Q0 = np.eye(self.numPatches)
        self.optQ = fsolve(self.computeActionGrad, Q0.ravel(), xtol = minimizationTol, maxfev = int(nStep*(self.numPatches**2)))
        self.optQ = np.reshape(self.optQ, (self.numPatches,self.numPatches))
        self.grad = self.computeActionGrad(self.optQ)
        isClose = np.isclose(self.grad, np.zeros(self.L), atol=gradTol)
        self.converged = isClose.all()
        print("\nis exact solution close to zero?\n", isClose)
        print("\ngrad:\n", self.grad)
        print(f"{self.L} hidden layer convolutional optQ is\n {self.optQ}")

    def setIW(self):
        self.optQ = np.eye(self.numPatches)

    def preprocess(self, X, Y):
        self.P, self.N0 = X.shape
        self.side = int(np.sqrt(self.N0))
        self.X = X
        self.Y = Y
        self.numPatches = int(self.N0 / (self.mask**2))
        patchX = np.array([kernels.divide2dImage(X[mu].reshape(self.side,self.side), self.mask) for mu in range(self.P)])
        self.Xtrain = np.zeros((self.numPatches), dtype = object )
        for i in range(self.numPatches):
            #print(patchX[:,i].reshape(self.P, -1))
            self.Xtrain[i] = patchX[:,i].reshape(self.P, -1)

        CXX = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range( self.numPatches):
            for j in range(i, self.numPatches):
                CXX[i][j] = np.dot(self.Xtrain[i], self.Xtrain[j].T) * self.corrNorm
                CXX[j][i] = CXX[i][j].T
        self.CXX = CXX
        self.CX = self.CXX.diagonal()
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        patchXtest = np.array([kernels.divide2dImage(Xtest[mu].reshape(self.side,self.side), self.mask) for mu in range(self.Ptest)])
        Xtest = np.zeros((self.numPatches), dtype = object )
        for i in range(self.numPatches):
            #print(patchX[:,i].reshape(self.P, -1))
            Xtest[i] = patchXtest[:,i].reshape(self.Ptest, -1)
        #self.C00 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        C00 = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range( self.numPatches):
            for j in range(i, self.numPatches):
                C00[i][j] = np.dot(Xtest[i], Xtest[j].T)  * self.corrNorm
                C00[j][i] = C00[i][j]
        self.C00 = C00
        CX0 = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range( self.numPatches):
            for j in range(i, self.numPatches):
                CX0[i][j] = np.dot(self.Xtrain[i], Xtest[j].T) * self.corrNorm
                CX0[j][i] = CX0[i][j]
        self.CX0 = CX0
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / (self.l1 * self.numPatches)
        #rK = self.CXX
        #rK0 = self.C00 
        #rK0X = self.CX0
        rK0XCumulate = np.zeros((self.P,self.Ptest), dtype = np.float64)
        rK0Cumulate= np.zeros((self.Ptest,self.Ptest))
        rKCumulate = np.zeros((self.P,self.P), dtype = np.float64)
        #for l in range(self.L):
        for i in range(self.numPatches):
            for j in range(i, self.numPatches):
                rKXii = self.CXX[i,i].diagonal()
                rKXXij = self.CXX[i,j]
                rKXjj = self.CXX[j,j].diagonal()
                rK0ii = self.C00[i,i].diagonal()
                rK0ij = self.C00[i,j].diagonal()
                rK0jj = self.C00[j,j].diagonal()
                rKX0ij = self.CX0[i,j]
                rKX0 = self.orderParam[i,j] *  self.kernel(rKXii[:, None], rKX0ij, rK0jj[None, :]) 
                rK0 = self.orderParam[i,j] * self.kernel(rK0ii, rK0ij, rK0jj) 
                rK = self.orderParam[i,j] * self.kernel(rKXii[:, None], rKXXij, rKXjj[None, :])
                rKCumulate = np.add(rKCumulate, rK)
                rK0Cumulate = np.add(rK0Cumulate, rK0)
                rK0XCumulate = np.add(rK0XCumulate, rKX0)
                if i!= j :
                    rKCumulate = np.add(rKCumulate, rK)
                    rK0Cumulate = np.add(rK0Cumulate, rK0)
                    rK0XCumulate = np.add(rK0XCumulate, rKX0)
        A =rKCumulate + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XCumulate.T, invK)
        self.K0_invK = K0_invK
        self.Ypred = np.dot(K0_invK, self.Y)
        self.rK0Cumulate = rK0Cumulate
        self.rK0XCumulate = rK0XCumulate

    def averageLoss(self, Ytest):
        bias = Ytest -  self.Ypred
        var = self.rK0Cumulate - np.sum( self.rK0XCumulate.T * self.K0_invK , axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()