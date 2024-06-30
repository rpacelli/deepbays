import numpy as np
from scipy.optimize import fsolve
from kernels import kernel_erf, kernel_relu, kernel_id, computeKmatrix, computeKmatrixTorch, kernel_erf_torch
import torch
from torch.autograd import Variable 

class FC_1HL_multiclass():
    def __init__(self, N1, l0, l1, act, T, k):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernelTorch = eval(f"kernel_{act}_torch")
        self.kernel = eval(f"kernel_{act}")
        self.k = k
        self.numOfVariables = int(self.k*(self.k+1)/2)
        
    def effectiveAction(self, *args):
        Q = torch.zeros((self.k, self.k))
        count = 0;
        for i in range(self.k):
            for j in range(i, self.k):
                Q[i][j] = args[count]
                Q[j][i] = Q[i][j]
                count = count + 1;
        rK = torch.zeros((self.k*self.P, self.k*self.P))
        for i in range(self.k):
            for j in range(self.k):
                rK[i*self.P:i*self.P+self.P,j*self.P:j*self.P+self.P] = Q[i,j]*self.K/self.l1        
        A = self.T * torch.eye(self.k * self.P) + rK
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
        for i in range(self.k):
            for j in range(i, self.k):
                variables[count] = matrix[i,j]
                count = count + 1;
        return variables

    def fromVariablesToQ(self, var):
        matrix = np.zeros((self.k, self.k))
        count = 0;
        for i in range(self.k):
            for j in range(i, self.k):
                matrix[i][j] = var[count]
                matrix[j][i] = matrix[i][j]
                count = count + 1;
        return matrix

    def minimizeAction(self, Q0):
        self.optVar = fsolve(self.gradientEq, self.fromQtoVariables(Q0))
        self.optQ = self.fromVariablesToQ(self.optVar)
        print(self.gradientEq(self.optVar))
        isClose = np.isclose(self.gradientEq(self.optVar), np.zeros(self.numOfVariables), atol=1e-05) 
        self.converged = isClose.all()
        print("\nis exact solution close to zero?", isClose)   
        print(f"{1} hidden layer optQ is {self.optQ}")

    def preprocess(self, data, labels):
        self.P, self.N0 = data.shape
        self.corrNorm = 1/(self.N0 * self.l0)
        self.C = np.dot(data, data.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.K = torch.tensor(self.kernel(self.C.diagonal()[:,None], self.C, self.C.diagonal()[None,:]), requires_grad = False) 
        y = np.array(labels.squeeze())
        self.labelsOfInterest = list(set(y))

        self.labelToIndex = {label: index for index, label in enumerate(self.labelsOfInterest)}

        yOneHot = torch.zeros((self.P, self.k))
        for i in range(self.P):
            labelIndex = self.labelToIndex[y[i].item()]
            yOneHot[i, labelIndex] = 1.0
            
        tempYReshaped = yOneHot[:,0];
        for i in range(1,self.k):
            tempYReshaped = np.concatenate((tempYReshaped, yOneHot[:,i]))
        self.y = torch.tensor(tempYReshaped)

    def computeTestsetKernels(self, data, testData):
        self.Ptest = len(testData)
        self.C0 = np.dot(testData, testData.T).diagonal() * self.corrNorm
        self.C0X = np.dot(testData, data.T) * self.corrNorm 
    
    def computeAveragePrediction(self, data, labels, testData, testLabels):
        self.computeTestsetKernels(data, testData)
        rK = np.zeros((self.k*self.P, self.k* self.P))
        rK0X = np.zeros((self.k*self.Ptest, self.k* self.P))
        rK0 = np.zeros((self.k* self.Ptest))
        
        for i in range(self.k):
            rK0temp = np.zeros((self.Ptest))
            for j in range(self.k):
                ind1 = i*self.P
                ind2 = j*self.P
                ind3 = i*self.Ptest
                rKX = self.C.diagonal() 
                rK0X[ind3:ind3+self.Ptest, ind2:ind2+self.P] = self.optQ[i,j] * self.kernel(self.C0[:,None], self.C0X, rKX[None, :]) /self.l1
                rK0temp = np.add( self.optQ[i,j] * self.kernel(self.C0, self.C0, self.C0)/self.l1, rK0temp)
                rK[ind1:ind1+self.P,ind2:ind2+self.P] = self.optQ[i,j]*self.K/self.l1  
            rK0[ind3:ind3+self.Ptest] = rK0temp
        A = rK + (self.T) * np.eye(self.P*self.k)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0X, invK)
        bias = testLabels - np.dot(K0_invK, self.y) 
        var = rK0 - np.sum(K0_invK * rK0X, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item()
