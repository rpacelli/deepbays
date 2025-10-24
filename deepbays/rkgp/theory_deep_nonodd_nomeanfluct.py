import numpy as np
from .. import kernels
import torch

class FC_deep_nonodd_nomeanfluct():
    def __init__(self, N1, T, L, priors = 1., act = "relu"):
        self.N1 = N1
        if type(priors) == float:
            self.priors = [priors] * (L+1)
        else:
            assert len(priors) == L+1, "priors misspecified"
        self.T = T
        self.L = L
        self.kernel = eval(f"kernels.kernel_{act}")
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.mean = eval(f"kernels.mean_{act}")
        

    def preprocess(self, X, Y):
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.alpha = self.P / self.N1
        self.corrNorm = 1/(self.N0 * self.priors[0])
        self.C = np.dot(X, X.T) * self.corrNorm
        self.CX = self.C.diagonal()
        self.y = Y.squeeze().to(torch.float64)
        self.y.requires_grad = False
        
        # precompute raw kernel and mean, as torch.tensors
        self.thetaL = torch.tensor(self.C, dtype = torch.float64, requires_grad = False)
        for l in range(self.L):
            self.thetaL = ((1. / self.priors[l+1])) * self.kernelTorch(self.thetaL.diagonal()[:,None], self.thetaL, self.thetaL.diagonal()[None,:])
            if l == self.L - 1:
                self.thetaLm1_diag = self.thetaL.diagonal() # needed for the mean vector muL
        self.muL = torch.tensor(self.mean(self.thetaLm1_diag), dtype = torch.float64, requires_grad = False)
        self.ML = torch.outer(self.muL, self.muL)


    def effectiveAction(self, Q):
        meanpref = 0.
        for l in range(self.L):
            meanpref += (Q[0,l] - 1./(1 + Q[1,l])) * torch.prod(Q[0,l+1])
        rKL = torch.prod(Q[0]) * self.thetaL - meanpref * self.ML
        A = rKL +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return (- torch.sum(Q[0]*Q[1])
                + torch.sum(torch.log(Q[1]))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                )
    
    def optimize_adam(self, Q0=None, lr=0.02, tolerance = 1e-4, max_epochs = 5000, verbose=True):
        
        if Q0 is not None:
            assert Q0.shape == (2, self.L), "Q0 init not in correct shape"
        else:
            Q0 = np.stack([np.ones(self.L), np.zeros(self.L)], axis=0)
        Q0 = torch.tensor(Q0)
        Q = Q0.clone().detach().requires_grad(True)
        
        opt = torch.optim.Adam([Q], lr=lr)
        self.optState = False
        self.optEpochs = 0
        
        for step in range(max_epochs):
            opt.zero_grad()
            S_val = self.effectiveAction(Q)
            (gradS,) = torch.autograd.grad(S_val, Q, create_graph=True) # important to keep graph to enable higher-order grads (next) !
            Loss = 0.5 * (gradS.pow(2).sum()) # optimization target: square norm of gradients dS/dQ, to find saddle-point
        
            Loss.backward()
            opt.step()
            self.optEpochs += 1
            if gradS.detach().norm() < tolerance:
                self.optState = True
                break
        
        self.optQ = Q.detach.numpy()
        self.meanpref = 0.
        for l in range(self.L):
            self.meanpref += (self.optQ[0,l] - 1./(1 + self.optQ[1,l])) * torch.prod(self.optQ[0,l+1])
        self.thetapref = torch.prod(self.optQ[0])
        if verbose: 
            print(f"opt state: {self.optState}, epochs: {self.optEpochs}, thetaprefactor: {self.thetapref}, meanprefactor: {self.meanpref}")
            print(f"Qs[0]: {self.optQ[0]}")
            print(f"Qs[1]: {self.optQ[1]}")
        
    def computeFullTrainTestKernel(self, X, Xtest):
        Xtt = np.stack[X, Xtest]
        Ctt = np.dot(Xtt, Xtt) * self.corrNorm
        # precompute raw kernel and mean, as torch.tensors
        self.thetaLtt = Ctt
        for l in range(self.L):
            self.thetaLtt = ((1. / self.priors[l+1])) * self.kernel(self.thetaLtt.diagonal()[:,None], self.thetaLtt, self.thetaLtt.diagonal()[None,:])
            if l == self.L - 1:
                self.thetaLm1tt_diag = self.thetaLtt.diagonal() # needed for the mean vector muL
        self.muLtt = self.mean(self.thetaLm1tt_diag)
        self.MLtt = np.outer(self.muLtt, self.muLtt)
        self.rKLtt = self.thetapref * self.thetaLtt - self.meanpref * self.MLtt
    
    
    def predict(self, Xtest):
        self.Ptest = len(Xtest)
        P = self.P
        Pt = self.Ptest

        self.computeFullTrainTestKernel(self.X, Xtest) # compute rKLtt
        A = self.rKLtt[:P,:P] + (self.T) * np.eye(P)
        self.invK = np.linalg.inv(A)
        self.K0_invK = np.matmul(self.rKLtt[-Pt:,:P], self.invK)
        self.Ypred = np.dot(self.K0_invK, self.Y).reshape(-1, 1)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred  
        var = self.rKLtt[-self.Ptest:,-self.Ptest:] - np.sum(self.K0_invK * self.rKLtt[-self.Ptest:,:self.P], axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()