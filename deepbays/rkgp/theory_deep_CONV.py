import numpy as np
from scipy.optimize import fsolve
from .. import kernels
import torch

class CONV_deep():
    def __init__(self, L, N1, T, l0 = 1., l1 = 1., act = "erf", mask = 7, stride = 7):
        self.N1, self.L, self.l0, self.l1, self.T, self.mask, self.stride = N1, L, l0, l1, T, mask, stride
        self.pool = False
        self.kernel, self.kernelTorch = eval(f"kernels.kernel_{act}"), eval(f"kernels.kernel_{act}_torch")
        self.corrNorm = 1 / (self.mask**2 * self.l0)
        self.corrNormPool = 1 / (self.mask**4 * self.l0)
        
    def effectiveAction(self, Q):
        Q = Q.reshape(self.numPatches, self.numPatches)
        rKTemp = torch.zeros((self.P,self.P), dtype = torch.float64)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                Kij    = torch.tensor(self.CXX[i][j])
                Kii    = torch.tensor(self.CXX[i][i])
                Kjj    = torch.tensor(self.CXX[j][j])
                rKij   = Q[i][j] * kernels.computeKmatrixMultipleCTorch(Kii, Kij, Kjj, self.kernelTorch) / (self.l1 * self.numPatches)
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
            self.minimizeAction(self.optQ + np.random.randn(*self.optQ.shape)*0.01, minimizationTol = minimizationTol, nStep = nStepSolver, gradTol = gradTol)
            i += 1     
    
    def setIW(self):
        self.optQ = np.eye(self.numPatches)
    
    def setPooling(self):
        self.pool = True
        self.optQ = np.ones((self.numPatches, self.numPatches))

    def preprocess(self, X, Y):
        self.P, self.N0 = X.shape
        self.side = int(np.sqrt(self.N0))
        self.X, self.Y = X, Y
        self.numPatches = int(self.N0 / (self.mask**2))
        patchX = np.array([kernels.divide2dImage(X[mu].reshape(self.side,self.side), self.mask) for mu in range(self.P)])
        self.Xtrain = np.zeros(self.numPatches, dtype = object)
        for i in range(self.numPatches):
            self.Xtrain[i] = patchX[:,i].reshape(self.P, -1) #the i-th element of Xtrain is an object with dimension (P, m^2), where m^2 is the number of elements in the single patch
        CXX = np.zeros((self.numPatches,self.numPatches), dtype = object)
        for i in range(self.numPatches):
            for j in range(i, self.numPatches):
                CXX[i][j] = np.dot(self.Xtrain[i], self.Xtrain[j].T) * self.corrNorm
                CXX[j][i] = CXX[i][j].T
        self.CXX = CXX
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        patchXtest = np.array([kernels.divide2dImage(Xtest[mu].reshape(self.side,self.side), self.mask) for mu in range(self.Ptest)])
        self.Xtest = np.zeros((self.numPatches), dtype = object ) # Each i element of Xtest is a P* times m^2 matrix, where m^2 is the number of elements in one patch 
        for i in range(self.numPatches):
            self.Xtest[i] = patchXtest[:,i].reshape(self.Ptest, -1) 
        C00 = np.zeros((self.numPatches, self.numPatches), dtype = object) # Each (i,j) element of the matrix C00 is a P* x P* matrix
        CX0 = np.zeros((self.numPatches, self.numPatches), dtype = object) # Each (i,j) element of the matrix CX0 is a P x P* matrix
        C0X = np.zeros((self.numPatches, self.numPatches), dtype = object)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                C00[i][j] = np.dot(self.Xtest[i], self.Xtest[j].T)  * self.corrNorm
                CX0[i][j] = np.dot(self.Xtrain[i], self.Xtest[j].T) * self.corrNorm
                C0X[i][j] = np.dot(self.Xtest[i], self.Xtrain[j].T) * self.corrNorm 
        self.CX0, self.C00, self.C0X = CX0, C00, C0X
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / (self.l1 * self.numPatches)
        if self.pool:
            self.orderParam /= self.numPatches
        #rKX0Temp = np.zeros((self.P, self.Ptest), dtype = np.float64)
        rK0XTemp = np.zeros((self.Ptest, self.P), dtype = np.float64)
        rK0Temp  = np.zeros((self.Ptest, self.Ptest), dtype = np.float64) #to check why this is a matrix and then becomes vector
        rKTemp   = np.zeros((self.P, self.P), dtype = np.float64)
        KijTemp, K0XijTemp, K0ijTemp = [], [], []
        #for l in range(self.L):
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                KXii  = self.CXX[i,i].diagonal() #diagonal in train data (mu, mu)
                KXXij = self.CXX[i,j]
                KXjj  = self.CXX[j,j].diagonal() #diagonal in train data (mu, mu)
                #KX0ij = self.CX0[i,j]
                K0Xij = self.C0X[i,j]
                K0ii  = self.C00[i,i].diagonal() #diagonal in test data (mu*, mu*)
                K0ij  = self.C00[i,j].diagonal() #diagonal in test data (mu*, mu*)
                K0jj  = self.C00[j,j].diagonal() #diagonal in test data (mu*, mu*)
                Ktrain = self.kernel(KXii[:, None], KXXij, KXjj[None, :])
                Kmix   = self.kernel(K0ii[:, None], K0Xij, KXjj[None, :])
                Ktest  = self.kernel(K0ii, K0ij, K0jj)
                KijTemp.append(Ktrain)
                K0ijTemp.append(Ktest)
                K0XijTemp.append(Kmix)
                rKij   = self.orderParam[i,j] * Ktrain
                rK0Xij = self.orderParam[i,j] * Kmix
                rK0ij  = self.orderParam[i,j] * Ktest
                #print(self.orderParam[i,j])
                #print(self.kernel(K0ii, K0ij, K0jj))
                #rKX0ij = self.orderParam[i,j] * self.kernel(KXii[:, None], KX0ij, K0jj[None, :]) 
                rKTemp   = np.add(rKTemp, rKij)
                rK0Temp  = np.add(rK0Temp, rK0ij)
                rK0XTemp  = np.add(rK0XTemp, rK0Xij)
                #rKX0Temp = np.add(rKX0Temp, rKX0ij)
        self.rK, self.rK0, self.rK0X = rKTemp, rK0Temp, rK0XTemp
        self.Kij, self.K0Xij, self.K0ij = KijTemp, K0XijTemp, K0ijTemp
        #self.rKX0 = rKX0Temp
        #print("how different are they?\n", self.rKX0[0, :10] , "\n", self.rK0X.T[0, :10]) #they are the same
        #assert (self.rKX0 == self.rK0X.T).all()
        A = self.rK + self.T * np.eye(self.P)
        invK = np.linalg.inv(A)
        #self.K0_invK = np.matmul(self.rKX0.T, invK)
        self.K0_invK = np.matmul(self.rK0X, invK)
        self.Ypred = np.matmul(self.K0_invK, self.Y)

    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred
        #var = self.rK0 - np.sum( self.rK0X * self.K0_invK , axis=1)
        var = self.rK0 - np.matmul(self.K0_invK, self.rK0X.T).diagonal()
        #var = self.rK0 - np.sum( self.rKX0.T * self.K0_invK , axis=1) #equivalent to #var = self.rK0 - np.matmul(self.K0_invK, self.rKX0).diagonal() 
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
    
class validation_CONV():
    def __init__(self, Kij, K0Xij, K0ij, Yval, Ytrain,  T, l0 = 1., l1 = 1., act = "erf", epsilon = 0.01):
        self.Kij, self.K0ij, self.K0Xij = torch.tensor(Kij, requires_grad = False), torch.tensor(K0ij, requires_grad = False), torch.tensor(K0Xij, requires_grad = False)
        self.numPatches = int(np.sqrt(len(Kij)))
        self.P = len(self.Kij[0])
        self.Pval = len(self.K0ij[0])
        self.l0, self.l1, self.T, self.act = l0, l1, T, act
        self.pool = False
        #self.Yval = torch.tensor(Yval) #.squeeze()
        self.Yval = Yval.to(torch.float64)
        self.Ytrain = Ytrain.to(torch.float64)
        self.kernel, self.kernelTorch = eval(f"kernels.kernel_{act}"), eval(f"kernels.kernel_{act}_torch")
        self.epsilon = epsilon
        
    def bias(self, Q):
        rK0XTemp = torch.zeros((self.Pval, self.P), dtype = torch.float64)
        rK0Temp  = torch.zeros((self.Pval,self.Pval), dtype = torch.float64)
        rKTemp   = torch.zeros((self.P, self.P), dtype = torch.float64)
        for i in range(self.numPatches):
            for j in range(self.numPatches):
                index = int(self.indexMat[i,j])             
                rKij   = Q[index] * self.Kij[i*self.numPatches+j] / (self.l1 * self.numPatches)
                rK0Xij = Q[index] * self.K0Xij[i*self.numPatches+j] / (self.l1 * self.numPatches)
                rK0ij  = Q[index] * self.K0ij[i*self.numPatches+j] / (self.l1 * self.numPatches)
                rKTemp   = torch.add(rKTemp, rKij)
                rK0Temp  = torch.add(rK0Temp, rK0ij)
                rK0XTemp  = torch.add(rK0XTemp, rK0Xij)
        self.rK, self.rK0, self.rK0X = rKTemp, rK0Temp, rK0XTemp 
        A = rKTemp +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        self.K0_invK = torch.matmul(self.rK0X, invA)
        self.Ypred = torch.matmul(self.K0_invK, self.Ytrain)
        return torch.mean((self.Yval - self.Ypred)**2)
    
    def computeBiasGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, requires_grad = True)
        f = self.bias(Q)
        f.backward()
        return Q.grad.data.detach().numpy()

    def minimizeBias(self, Q0 = 1., minimizationTol = 1e-6, nStep = 100000, gradTol = 1e-3):
        if isinstance(Q0, float):
            Q0 = np.eye(self.numPatches)
        numVariables = int(self.numPatches*(self.numPatches+1)/2)
        self.indexMat = np.zeros((self.numPatches, self.numPatches))
        Qstart = np.zeros(numVariables)
        count = 0
        for i in range(self.numPatches): 
            for j in range(i, self.numPatches): 
                self.indexMat[i,j] = count
                self.indexMat[j,i] = count
                Qstart[count] = Q0[i,j]
                count +=1
        optQ = Qstart - self.epsilon * self.computeBiasGrad(Qstart)
        self.optQ = np.zeros((self.numPatches, self.numPatches))
        count=0
        for i in range(self.numPatches): 
            for j in range(i, self.numPatches): 
                self.optQ[i,j] = optQ[count]
                self.optQ[j,i] = optQ[count]
                count += 1
        self.grad = self.computeBiasGrad(optQ) # gradient of the action 
        self.isClose = np.isclose(self.grad, np.zeros(numVariables), atol=gradTol) #checks if gradient is smaller than gradTol
        self.converged = self.isClose.all()
        print("\nis grad close to zero?\n", self.isClose)
