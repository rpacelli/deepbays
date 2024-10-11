import numpy as np
from scipy.optimize import minimize, fsolve
import torch
from .. import kernels

def kmatrix(Cmm,Cmn,Cnn,kernel):
    P1,P2 = Cmn.shape
    K = np.zeros((P1,P2))
    for i in range(P1): 
        for j in range(P2):         
            K[i][j] = kernel(Cmm[i][i], Cmn[i][j], Cnn[j][j])
            #K[j][i] = K[i][j]
    return K

class FC_1HL_spectral():
    def __init__(self, theta, N1, T, l0 = 1., l1 = 1., act = "erf"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernels.kernel_{act}")
        #self.theta = theta #fraction of shut nodes
        self.theta = theta #fraction of open nodes 

    def effectiveAction(self,y): #y is Q in the calculations
        #x = self.theta*(1+y)/(self.theta+(1-self.theta)*np.sqrt(1+y))
        x = self.theta/((1+y)**(1.5)*(1-self.theta + self.theta/np.sqrt(1+y)))
        A = self.T * np.identity(self.P) + (x/self.l1) * self.diagK
        invA = np.linalg.inv(A)
        return (- y*x - 2 * np.log(1-self.theta + (self.theta)/np.sqrt(1+y))
            + (1/self.N1) * np.sum(np.log(self.T + x*self.eigvalK/self.l1))
            + (1/self.N1) * np.dot(self.yT, np.dot(invA, self.yT)) )

    def minimizeAction(self, x0): # x0 is initial condition
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,np.inf),) , tol=1e-20)
        self.optQ = (optQ.x).item()
        self.optbarQ = self.theta/((1+self.optQ)**(1.5)*(1-self.theta + self.theta/np.sqrt(1+self.optQ)))
        print(f"value of S at the saddle point {self.effectiveAction(self.optQ)}")
        print(f"optQ is {self.optQ}")
        print(f"optbarQ is {self.optbarQ}")

    def preprocess(self, Xtrain, Ytrain):
        self.P, self.N0 = Xtrain.shape
        self.corrNorm = 1/(self.N0*self.l0)
        self.C = np.dot(Xtrain, Xtrain.T) * self.corrNorm
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.CX = self.C.diagonal()
        self.K = kernels.computeKmatrix(self.C, self.kernel)
        [self.eigvalK, eigvecK] = np.linalg.eig(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, Ytrain.squeeze())

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.dot(Xtest, Xtest.T).diagonal() * self.corrNorm
        self.C0X = np.dot(Xtest, self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def computeAveragePrediction(self, Xtest, Ytest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        A = self.orderParam * self.K + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        rK0 = self.orderParam * self.K0 
        rK0X = self.orderParam * self.K0X
        K0_invK = np.matmul(rK0X, invK)
        bias = Ytest - np.dot(K0_invK, self.Ytrain) 
        var = rK0 - np.sum(K0_invK * rK0X, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item()
    
    def computePosteriorEigenvalues(self):
        norm = self.theta/np.sqrt(1+self.optQ) +1-self.theta
        self.p0 = (1-self.theta)/norm
        self.p1 = (self.theta/(np.sqrt(1+self.optQ)))/norm
        print(f"posterior prob of open node: {self.p1}")
        print(f"posterior prob of shut node: {self.p0}")


class FC_1HL_multitask():
    def __init__(self, N1, T, l0 = 1., l1 = 1., act = "erf"):
        self.N1 = N1
        self.l0 = l0
        self.l1 = l1 
        self.T = T
        self.kernel = eval(f"kernels.kernel_{act}")

    def effectiveAction(self, barQ, K1, K2, Kmix,labels):
        Ptot = self.Ps + self.Pt
        rK = torch.zeros((Ptot, Ptot), dtype=torch.float64)
        #assert (Kmix == KmixT).all()
        rK[:self.Ps, :self.Ps] = K1 * barQ[0]
        rK[self.Ps:Ptot, self.Ps:Ptot] = K2 * barQ[3]
        assert rK[self.Ps:Ptot, self.Ps:Ptot].shape == (self.Pt, self.Pt)
        rK[self.Ps:Ptot, :self.Ps] = Kmix.T * barQ[2]
        rK[:self.Ps, self.Ps:Ptot] = Kmix * barQ[1]
        A = rK / self.l1  +  self.T * torch.eye(Ptot) 
        invA = torch.inverse(A)
        return (  barQ[0] + barQ[3]
                - torch.log(barQ[0]*barQ[3]- barQ[1]*barQ[2])
                + (1/self.N1) * torch.matmul(labels, torch.matmul(invA, labels)) 
                + (1/self.N1) * torch.logdet(A)
                 )

    def computeGrad(self, barQ, K1, K2, Kmix, labels):
        barQ = torch.tensor(barQ, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(barQ, K1, K2, Kmix, labels)
        f.backward(retain_graph=True)
        return barQ.grad.data.detach().numpy()   

    def computeAveragePrediction(self, inputs1, inputs2, labels1, labels2, testInputs1, testInputs2, testLabels1, testLabels2,kernel, args,theoryFile, barQ):    
        self.Ps = len(inputs1)
        self.Pt = len(inputs2)
        N0 = len(inputs1[0])  
        self.corrNorm = 1/(N0)
        #diag = self.gamma + self.l0
        #offDiag = -self.gamma
        offDiag = -self.gamma/2
        diag = self.l0
        regMat = [[diag, offDiag],[offDiag, diag]]
        invReg = np.linalg.inv(regMat)

        data = torch.concatenate((inputs1,inputs2))
        labels = torch.concatenate((labels1.squeeze().type(torch.DoubleTensor), labels2.squeeze().type(torch.DoubleTensor)))
        labels.requires_grad = False

        C1 = np.dot(inputs1,inputs1.T) * self.corrNorm * invReg[0][0]
        C2 = np.dot(inputs2,inputs2.T) * self.corrNorm * invReg[1][1]
        Cmix = np.dot(inputs1,inputs2.T) * self.corrNorm * invReg[0][1]

        K1 = torch.tensor(kmatrix(C1,C1,C1,kernel), dtype = torch.float64, requires_grad = False)
        K2 = torch.tensor(kmatrix(C2,C2,C2,kernel), dtype = torch.float64, requires_grad = False)
        Kmix = torch.tensor(kmatrix(C1,Cmix,C2,kernel), dtype = torch.float64, requires_grad = False)

        additionalArgs = (K1, K2, Kmix, args, labels)
        optQ = fsolve(self.computeGrad, barQ, args=additionalArgs, xtol = 1e-8)
        print(f"\nPoint where the gradient is approximately zero: {optQ}")
        isClose = np.isclose(self.scomputeGrad(optQ, *additionalArgs), [0.0, 0.0, 0.0, 0.0]) 
        print("\nis exact solution close to zero?", isClose)   
        #print(f"\nInteraction strength gamma: {self.gamma}")
        optQ = optQ.reshape(2,2)

        predLoss1,predLoss2 = 0,0
        for p in range(self.Ptest):
            x,y = np.array(testInputs1[p]), np.array(testLabels1[p])
            predLoss1 += self.computeSinglePrediction(data,labels,x,y,optQ, 1, kernel, K1, K2, Kmix, self.corrNorm, invReg)
            x,y = np.array(testInputs2[p]), np.array(testLabels2[p])
            predLoss2 += self.computeSinglePrediction(data,labels,x,y,optQ, 2, kernel, K1, K2, Kmix, self.corrNorm, invReg)

        with open(theoryFile, "a+") as f: 
            print(self.N1, self.Ps, self.Pt, self.T, self.gamma, self.l0, self.l1, optQ[0][0], optQ[0][1], optQ[1][1], predLoss1/self.Ptest, predLoss2/self.Ptest, self.Ptest, self.q, self.eta, self.rho, isClose.all(), file = f)

        print("test loss first task ", predLoss1 / self.Ptest , "\ntest loss second task ", predLoss2 / self.Ptest )

        return optQ
    
    def computeSinglePrediction(self, data, labels, x, y, optQ, task, kernel, K1, K2, Kmix, corrNorm, invReg):
        Ptot = self.Ps + self.Pt
        P = self.Ps
        corrXX = np.dot(x,x)*corrNorm*invReg[task-1][task-1]
        rKmu =  np.random.randn(Ptot)
        for mu in range(self.Ps):
            corrXData = np.dot(x,data[mu]) * corrNorm * invReg[task-1][0] 
            corrDataData = np.dot(data[mu],data[mu]) * corrNorm * invReg[0][0]
            rKmu[mu] = (optQ[task-1,0]/self.l1) * kernel(corrXX,corrXData,corrDataData) 
        for mu in range(self.Pt):
            corrXData2 = np.dot(x,data[self.Ps + mu]) * corrNorm * invReg[task-1][1]
            corrData2Data2 = np.dot(data[self.Ps+mu],data[self.Ps +mu]) * corrNorm * invReg[1][1]
            rKmu[self.Ps+mu] = (optQ[task-1,1]/self.l1) * kernel(corrXX,corrXData2,corrData2Data2)

        rKXX = (optQ[task-1,task-1]/self.l1) * kernel(corrXX,corrXX,corrXX) 
        rK = np.zeros((Ptot, Ptot), dtype=float)
        rK[:self.Ps, :self.Ps] = K1 * optQ[0, 0]
        rK[self.Ps:Ptot, self.Ps:Ptot] = K2 * optQ[1, 1]
        rK[self.Ps:Ptot, :self.Ps] = Kmix.T * optQ[0, 1]
        rK[:P, P:Ptot] = rK[P:Ptot, :P].T
        #rK[:P, P:Ptot] = rK[P:Ptot, :P]
        #assert rK[self.Ps:Ptot, self.Ps:Ptot].shape == (self.Pt, self.Pt)

        A = rK/self.l1 +  self.T * np.eye(Ptot) 
        invA = np.linalg.inv(A)
        rKmu_invA = np.matmul(rKmu, invA)
        bias = y - np.dot(rKmu_invA, labels)
        var = rKXX - np.dot(rKmu_invA, rKmu) 
        predErr = bias**2 + var
        return predErr.item()

