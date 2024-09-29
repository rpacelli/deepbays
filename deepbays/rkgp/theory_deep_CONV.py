import numpy as np
from scipy.optimize import fsolve
from .. import kernels
import torch

class CONV_deep():
    def __init__(self, L, N1, T, l0 = 1., l1 = 1., act = "erf", mask = 7, stride = 7):
        self.N1, self.L = N1, L
        self.l0, self.l1, self.T = l0, l1, T
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        self.mask, self.stride = mask, stride
        self.corrNorm = 1/(self.mask**2 * self.l0)

    def effectiveAction(self, Q):
        Q = Q.reshape(self.numPatches, self.numPatches)
        rKTemp = torch.zeros((self.P,self.P), dtype = torch.float64)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                Kij = torch.tensor(self.CXX[i][j])
                Kii = torch.tensor(self.CXX[i][i])
                Kjj = torch.tensor(self.CXX[j][j])
                rKij =  Q[i][j] * kernels.computeKmatrixMultipleCTorch(Kii, Kij, Kjj, self.kernelTorch) / (self.l1*self.numPatches)
                rKTemp = torch.add(rKTemp , rKij)
        A = rKTemp +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return (torch.trace(Q) - torch.logdet(Q)
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
        self.grad = self.computeActionGrad(self.optQ) # gradient of the action 
        self.isClose = np.isclose(self.grad, np.zeros((self.numPatches, self.numPatches)), atol=gradTol) #checks if gradient is smaller than gradTol
        self.converged = self.isClose.all()
        print("\nis grad close to zero?\n", self.isClose)
    
    def forceMinimize(self, nStepForce = 5, minimizationTol = 1e-6, nStepSolver = 100000, gradTol = 1e-3):
        i = 0
        while (not self.converged) and (i < nStepForce):
            print(f"\nsolution not found. Try #{i+1}:") 
            self.minimizeAction(self.optQ, minimizationTol = minimizationTol, nStep = nStepSolver, gradTol = gradTol)
            i += 1     
    
    def setIW(self):
        self.optQ = np.eye(self.numPatches)

    def preprocess(self, X, Y):
        self.P, self.N0 = X.shape
        self.side = int(np.sqrt(self.N0))
        self.X, self.Y = X, Y
        self.numPatches = int(self.N0 / (self.mask**2))
        patchX = np.array([kernels.divide2dImage(X[mu].reshape(self.side,self.side), self.mask) for mu in range(self.P)])
        self.Xtrain = np.zeros((self.numPatches), dtype = object )
        for i in range(self.numPatches):
            self.Xtrain[i] = patchX[:,i].reshape(self.P, -1) #the i-th element of Xtrain is an object with dimension (P, m^2), where m^2 is the number of elements in the single patch
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
            Xtest[i] = patchXtest[:,i].reshape(self.Ptest, -1) # Each i element of Xtest is a P* times m^2 matrix, where m^2 is the number of elements in one patch 
        C00 = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range( self.numPatches):
            for j in range(i, self.numPatches):
                C00[i][j] = np.dot(Xtest[i], Xtest[j].T)  * self.corrNorm
                C00[j][i] = C00[i][j].T
        self.C00 = C00
        CX0 = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                # Each (i,j) element of the matrix CX0 is a P x P* matrix
                CX0[i][j] = np.dot(self.Xtrain[i], Xtest[j].T) * self.corrNorm 
        self.CX0 = CX0
    
    def predict(self, Xtest, pool = False):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / (self.l1 * self.numPatches)
        if pool:
            self.optQ = np.ones((self.numPatches, self.numPatches))
            self.orderParam = self.optQ / (self.l1 * self.numPatches**2)
        rKX0Temp = np.zeros((self.P,self.Ptest), dtype = np.float64)
        rK0Temp  = np.zeros((self.Ptest,self.Ptest))
        rKTemp   = np.zeros((self.P,self.P), dtype = np.float64)
        #for l in range(self.L):
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                KXii  = self.CXX[i,i].diagonal() #diagonal in train data (mu, mu)
                KXXij = self.CXX[i,j]
                KXjj  = self.CXX[j,j].diagonal() #diagonal in train data (mu, mu)
                KX0ij = self.CX0[i,j]
                K0ii  = self.C00[i,i].diagonal() #diagonal in test data (mu*, mu*)
                K0ij  = self.C00[i,j].diagonal() #diagonal in test data (mu*, mu*)
                K0jj  = self.C00[j,j].diagonal() #diagonal in test data (mu*, mu*)
                rKij   = self.orderParam[i,j] * self.kernel(KXii[:, None], KXXij, KXjj[None, :])
                rKX0ij = self.orderParam[i,j] * self.kernel(KXii[:, None], KX0ij, K0jj[None, :]) 
                rK0ij  = self.orderParam[i,j] * self.kernel(K0ii, K0ij, K0jj) 
                rKTemp   = np.add(rKTemp, rKij)
                rK0Temp  = np.add(rK0Temp, rK0ij)
                rKX0Temp = np.add(rKX0Temp, rKX0ij)
        self.rK = rKTemp
        self.rK0 = rK0Temp
        self.rKX0 = rKX0Temp
        A = self.rK + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        self.K0_invK = np.matmul(self.rKX0.T, invK)
        self.Ypred = np.dot(self.K0_invK, self.Y)

    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred
        var = self.rK0 - np.sum( self.rKX0.T * self.K0_invK , axis=1) #equivalent to the line below
        #var = self.rK0 - np.matmul(self.K0_invK, self.rKX0).diagonal()
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()