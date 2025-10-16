import numpy as np
from scipy.optimize import fsolve, minimize_scalar, brent, brentq, newton
from .. import kernels
import torch

class FC_deep_vanilla():
    def __init__(self, 
                 L      : int, 
                 N1     : int, 
                 T      : float, 
                 priors : list = [1.0, 1.0], 
                 act    : str = "erf"):
        self.N1, self.T, self.L, self.l0, self.l1 = N1, T, L, priors[0],priors[1:] 
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.kernel = eval(f"kernels.kernel_{act}")
        
    def effectiveAction(self, Q):
        rKL = torch.tensor(self.C, dtype = torch.float64, requires_grad = False)
        for l in range(self.L):
            rKL = ((1. / self.l1[l])) * self.kernelTorch(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        for l in range(self.L):
            rKL *= Q[l]
        A = rKL +  self.T * torch.eye(self.P) 
        invA = torch.inverse(A)
        return ( torch.sum(Q - torch.log(Q))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y)) 
                + (1/self.N1) * torch.logdet(A)
                 )
    
    def computeActionGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(Q)
        f.backward()
        return Q.grad.data.detach().numpy()

    def LogLogEffective1DAction(self, logQscalar, offset=0.): 
        """
        Exploit that Qs are all equal due to symmetry. 
        Working with logQ makes big brentq brackets fast and is fine because Q < 0 not physical.
        If effective action should be negative, log-action may be NaN; a positive offset kwarg can be used to shift action upwards.
        """
        return torch.log(offset + self.effectiveAction(torch.exp(logQscalar).broadcast_to(self.L)))
    
    def LogLogEff1DActionPrime(self, logQ:float, offset=0.): # First derivative of loglog action, very benign for finding root logQstar.
        logQ = torch.tensor(logQ, dtype = torch.float64, requires_grad = True)
        g = torch.autograd.grad(self.LogLogEffective1DAction(logQ, offset=offset), logQ)[0]
        return g.item()
    
    # def LogLogEff1DActionPrime(self, logQ:float, offset=0.): # previous, equivalent impl
    #     logQ = torch.tensor(logQ, dtype = torch.float64, requires_grad = True)
    #     g = self.LogLogEffective1DAction(logQ, offset=offset)
    #     g.backward()
    #     return logQ.grad.detach().item()

    def optimize_LogLog(self, logQmin=-4., logQmax=7.): # fast and stable also for muP.
        """ 
        Find saddlepoint of effective action in log-log space using brentq bracket.
        Uses scalar Q, since N1 is the same across hidden layers and action is symm wrt. Qs.
        Notes: 
            - Lims are natural log -> defaults are exp(-4) = 0.018, and exp(7) = 1096 
            - Checks for negative eff. action at Q = 1, which may happen if Qstar close to 1, and attempts to shift log accordingly to avoid nan.
              If this should fail (not observed so far), fall back to standard optimize() without logs.
        """
        neg_offset = self.effectiveAction(torch.ones(self.L)).item()
        if neg_offset <= 0.: 
            offset = -2. * neg_offset # eff. action is negative close to Q=1, need global positive offset to make log action well defined. 
        else:
            offset = 0.
        logQstar = brentq(self.LogLogEff1DActionPrime, logQmin, logQmax, args=(offset,))
        self.optQ = np.exp(logQstar) * np.ones(self.L)
        isClose = np.isclose(self.computeActionGrad(self.optQ), np.zeros(self.L)) # tests additionally for smallness of grads, not only closeness to true root
        self.converged = isClose.all()

    def optimize(self, Q0 = 1.):
        if isinstance(Q0, float):
            Q0 = Q0 * np.ones(self.L)
        self.optQ = fsolve(self.computeActionGrad, Q0, xtol = 1e-8)
        isClose = np.isclose(self.computeActionGrad(self.optQ), np.zeros(self.L)) # tests additionally for smallness of grads, not only closeness to true root
        self.converged = isClose.all()
        #print("\nis exact solution close to zero?", isClose)   
        #print(f"{self.L} hidden layer optQ is {self.optQ}")

    def optimize_smart(self, logQmin=-4., logQmax=7., showdebugplots=True):
        """ Tries log-log optimizer first, if it fails finds approx minimum by line search and initializes optimize there. Add more bells and whistles here if need arises."""
        try:
            self.optimize_LogLog( logQmin=logQmin, logQmax=logQmax)
        except:
            print(f'brentq with logQ brackets a={logQmin} g(a)={self.LogLogEff1DActionPrime(logQmin)}, b={logQmax} g(b)={self.LogLogEff1DActionPrime(logQmax)} failed. ')
            self.converged = False
        if not self.converged:
            print('Plotting action to aid problem diagnosis and finding local init from plot values...')
            Q0new = self.debug_action(show=showdebugplots)
            self.optimize(Q0new)
            if not self.converged:
                print('... did not work - needs manual help or debugging.')
    
    def debug_action(self, Qminexp=-2., Qmaxexp=4., show=True):
        Qs = np.logspace(Qminexp, Qmaxexp, num=40)
        t_ones = torch.ones(self.L, dtype = torch.float64, requires_grad=False)
        logaction_vals =  [np.log(self.effectiveAction(Q*t_ones).item()) for Q in Qs]
        # loglogaction_vals =  [self.LogLogEffective1DAction(np.log10(Q)*t_ones).item() for Q in Qs]
        # loglogactionprime_vals =  [self.LogLogEff1DActionPrime(np.log10(Q)) for Q in Qs]
        if show:
            import matplotlib.pyplot as plt
            dfig, dax = plt.subplots() 
            dax.plot(Qs, logaction_vals)
            # dax.plot(Qs, loglogaction_vals, linestyle='dashed')
            # dax.plot(Qs, loglogactionprime_vals, linestyle='dotted')
            dax.set_xscale('log')
        Qinit_new = Qs[np.argmin(logaction_vals)]
        return Qinit_new

    def setIW(self):
        self.optQ = np.ones(self.L)

    def preprocess(self, X, Y):
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.corrNorm = 1/(self.N0 * self.l0)
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
            rKXL = rKL.diagonal() 
            rK0XL = (1. / self.l1[l]) * self.kernel(rK0L[:,None], rK0XL, rKXL[None, :])
            rK0L = (1. / self.l1[l]) * self.kernel(rK0L, rK0L, rK0L)
            rKL = (1. / self.l1[l]) * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        for l in range(self.L):
            rKXL = self.optQ[l] * rKXL
            rK0XL = self.optQ[l] * rK0XL
            rK0L = self.optQ[l] * rK0L
            rKL = self.optQ[l] * rKL
        A = rKL + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        K0_invK = np.matmul(rK0XL, invK)
        self.rK0L = rK0L
        self.K0_invK = K0_invK
        self.rK0XL = rK0XL
        self.Ypred = np.dot(K0_invK, self.Y)
        return self.Ypred
    
    def effectiveKernel(self):
        assert self.converged, "optimize() was not yet called or did not converge"
        rKL = self.C
        for l in range(self.L):
            rKL = (1. / self.l1[l]) * self.kernel(rKL.diagonal()[:,None], rKL, rKL.diagonal()[None,:])
        for l in range(self.L):
            rKL = self.optQ[l] * rKL
        A = rKL + (self.T) * np.eye(self.P)
        return rKL, A
    
    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred  
        var = self.rK0L - np.sum(self.K0_invK * self.rK0XL, axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
    
