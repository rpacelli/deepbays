import numpy as np
from scipy.optimize import fsolve
import torch
from torch.autograd import Variable 
from .. import kernels
from .. import tasks
## DEBUGGED VERSION. PASSED ALL TESTS ON 01/07/2024

class FC_1HL_multiclass():
    def __init__(self, 
                 N1 : int, 
                 D  : int, 
                 T  : float, 
                 l0 : float = 1.0, 
                 l1 : float = 1.0, 
                 act : str = "erf", 
                 oneHot : bool = False
                 ):
        self.N1, self.l0, self.l1, self.T, self.D, self.oneHot = N1, l0, l1, T, D, oneHot
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        self.numOfVariables = int(self.D*(self.D+1)/2)
        
    def effectiveAction(self, *args):
        Q = torch.zeros((self.D, self.D))
        count = 0;
        for i in range(self.D):
            for j in range(i, self.D):
                Q[i][j] = args[count]
                Q[j][i] = Q[i][j]
                count = count + 1;
        rK = torch.zeros((self.D*self.P, self.D*self.P))
        for i in range(self.D):
            for j in range(self.D):
                rK[i*self.P:i*self.P+self.P,j*self.P:j*self.P+self.P] = Q[i,j]*self.D/self.l1        
        A = self.T * torch.eye(self.D * self.P) + rK
        return torch.trace(Q) - torch.logdet(Q) + (1/self.N1) * torch.logdet(A) + (1/self.N1)*torch.matmul(self.y,torch.matmul(torch.inverse(A),self.y))
    
    def computeActionGrad(self, *args):
        variables = [Variable(torch.tensor(arg, requires_grad=True, dtype=torch.float), requires_grad=True) for arg in args]
        f = self.effectiveAction(*variables)
        f.backward()
        return [var.grad.data.detach().numpy() if var.grad is not None else 0 for var in variables]

    def gradientEq(self, vars):
        grads = self.computeActionGrad(*vars)
        return grads

    def fromQtoVariables(self, matrix):
        variables = np.zeros(self.numOfVariables)
        count = 0;
        for i in range(self.D):
            for j in range(i, self.D):
                variables[count] = matrix[i,j]
                count = count + 1;
        return variables

    def fromVariablesToQ(self, var):
        matrix = np.zeros((self.D, self.D))
        count = 0;
        for i in range(self.D):
            for j in range(i, self.D):
                matrix[i][j] = var[count]
                matrix[j][i] = matrix[i][j]
                count = count + 1;
        return matrix

    def optimize(self, x0 = 1):
        if isinstance(x0,(float,int)):
            x0 = np.eye(self.D)
        self.optVar = fsolve(self.gradientEq, self.fromQtoVariables(x0))
        self.optQ = self.fromVariablesToQ(self.optVar)
        print(self.gradientEq(self.optVar))
        isClose = np.isclose(self.gradientEq(self.optVar), np.zeros(self.numOfVariables), atol=1e-05) 
        self.converged = isClose.all()
        #print("\nis exact solution close to zero?", isClose)   
        #print(f"optQ is {self.optQ}")
    
    def setIW(self):
        self.optQ = np.eye(self.D)

    def preprocess(self, Xtrain, Ytrain):
        self.P, self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.N0 * self.l0)
        self.Xtrain = Xtrain
        self.C = np.dot(Xtrain, Xtrain.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.D = torch.tensor(self.kernel(self.C.diagonal()[:,None], self.C, self.C.diagonal()[None,:]), requires_grad = False)
        if self.oneHot:
            self.Ytrain = tasks.oneHotEncoding(Ytrain)
        else:
            self.Ytrain = Ytrain
        tempYReshaped = self.Ytrain[:,0];
        for i in range(1,self.D):
            tempYReshaped = np.concatenate((tempYReshaped, self.Ytrain[:,i]))
        self.y = torch.tensor(tempYReshaped, dtype=torch.float32)

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.Xtrain.T) * self.corrNorm 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
        self.K0 = self.kernel(self.C0, self.C0, self.C0)
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        rK = np.zeros((self.D*self.P, self.D* self.P)) #fast-varying index is mu.
        rK0 = np.zeros(self.D*self.Ptest)
        rK0X = np.zeros(( self.D*self.Ptest, self.D* self.P))
        for i in range(self.D):
            ind1 = i*self.P
            for j in range(self.D):
                ind2 = j*self.P
                rK[ind1:ind1+self.P,ind2:ind2+self.P] = self.optQ[i,j]*self.D/self.l1      
        for i in range(self.D):
            ind1 = i*self.Ptest
            rK0[ind1:ind1+self.Ptest] = self.optQ[i,i]*self.K0 /self.l1 
            for j in range(self.D):
                ind2 = j*self.P
                rK0X[ind1:ind1+self.Ptest,ind2:ind2+self.P] = self.optQ[i,j]*self.K0X  /self.l1
        self.rK0X = rK0X
        self.rK0 = rK0
        A = rK + (self.T) * np.eye(self.P*self.D)
        self.invK = np.linalg.inv(A)
        rKY = np.matmul(self.invK, self.y)
        self.prediction = np.matmul(rK0X, rKY)
        yPred = self.prediction.reshape(self.Ptest,self.D)
        return yPred

    def averageLoss(self, Ytest):
        if self.oneHot:
            self.Ytest = tasks.oneHotEncoding(Ytest, verbose=False)
        else:
            self.Ytest = Ytest
        tempYTestReshaped = self.Ytest[:,0]
        for i in range(1,self.D):
            tempYTestReshaped = np.concatenate((tempYTestReshaped, self.Ytest[:,i]))
        self.Ytest = torch.tensor(tempYTestReshaped)
        bias = (self.Ytest-self.prediction)**2
        biasPerClass = bias.reshape(self.Ptest,self.D).mean(axis = 0) # array of dimension k. each entry is average (on testset) bias for that class
        extVar = np.matmul(self.rK0X,np.matmul(self.invK, self.rK0X.T))
        var = self.rK0 - np.diag(extVar)
        varPerClass = var.reshape(self.Ptest,self.D).mean(axis = 0) # array of dimension k. each entry is average (on testset) var for that class
        predLoss = biasPerClass.mean().item() + varPerClass.mean().item()
        return predLoss, biasPerClass, varPerClass