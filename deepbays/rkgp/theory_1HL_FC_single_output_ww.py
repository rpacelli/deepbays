import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from .. import kernels
import torch


## MODIFIED VERSION WITH NEW FUNCTIONS IMPLEMENTED
class FC_1HL_metric():
    def __init__(self, 
                 N1   : int, 
                 T    : float, 
                 l0   : float = 1.0,
                 l1   : float = 1.0,
                 act  : str = "erf", 
                 bias : bool = False):
        self.N1, self.l0, self.l1, self.T = N1, l0, l1, T

        self.kernel = eval(f"kernels.kernel_{act}")
        if bias: 
            self.kernel = eval(f"kernels.kernel_{act}_bias")
        self.activation = act  # Store activation for later use

    def effectiveAction(self, x):
        A_diag = (self.T + (x/self.l1) * self.eigvalK)
        invA = np.diag(1/A_diag)
        return ( x - np.log(x)
            + (1/self.N1) * np.sum(np.log(self.T + x * self.eigvalK / self.l1))
            + (1/self.N1) * np.dot(self.yT, np.dot(invA, self.yT)) )

    def D_effectiveAction(self, Q):  # New function: first derivative of effective action
        """Compute the first derivative of the effective action"""
        term1 = np.sum(Q/self.l1 * self.eigvalK / (self.T + Q/self.l1 * self.eigvalK))
        term2 = np.dot(self.yT, np.dot(np.diag(Q/self.l1 * self.eigvalK / (self.T + Q/self.l1 * self.eigvalK)**2), self.yT))
        return (1/self.N1) * (1 - 1/Q + term1 - term2)

    def optimize(self, x0 = 1.): # x0 is initial condition
        optQ = minimize(self.effectiveAction, x0, bounds = ((1e-8,np.inf),) , tol=1e-12)
        self.optQ = (optQ.x).item()
        self.orderParam = self.optQ / self.l1
        assert self.optQ > 0 , "Unphysical solution found (Q is negative)."
        # Compute RK_diag_inv after optimization (similar to GaussianProcess)

    def setIW(self):
        self.optQ = 1
        self.orderParam = self.optQ / self.l1

    def preprocess(self, Xtrain, Ytrain, metric= None):
        self.P, self.N0 = Xtrain.shape
        self.metric = metric
        assert self.metric.shape[0] == self.N0, "Metric dimension does not match input dimension."
        #self.metric = np.eye(self.N0)
        self.corrNorm = 1/(self.N0*self.l0)
        self.C =np.matmul(np.matmul(Xtrain, self.metric),Xtrain.T) * self.corrNorm
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.CX = self.C.diagonal()
        self.K = kernels.computeKmatrix(self.C, self.kernel)
        self.eigvalK, eigvecK = np.linalg.eigh(self.K)
        self.diagK = np.diagflat(self.eigvalK)
        self.Udag = eigvecK.T
        self.yT = np.matmul(self.Udag, Ytrain.squeeze())
        self.eigvecK = eigvecK  # Store for later use
        self.computeDKernel()

    def computeTestsetKernels(self, Xtest):
        self.Ptest = len(Xtest)
        self.C0 = np.matmul(Xtest, np.matmul(self.metric, Xtest.T)).diagonal() * self.corrNorm
        self.C0X = np.matmul(np.matmul(Xtest, self.metric), self.Xtrain.T) * self.corrNorm
        self.K0 =  self.kernel(self.C0, self.C0, self.C0) 
        self.K0X = self.kernel(self.C0[:,None], self.C0X, self.CX[None, :])
    
    def predict(self, Xtest):
        self.computeTestsetKernels(Xtest)
        self.orderParam = self.optQ / self.l1
        A = self.orderParam * self.K + (self.T) * np.eye(self.P)
        invK = np.linalg.inv(A)
        self.rK0X = self.orderParam * self.K0X
        self.K0_invK = np.matmul(self.rK0X, invK)
        self.Ypred =  np.dot(self.K0_invK, self.Ytrain)
        return self.Ypred
    
    def averageLoss(self, Ytest):
        self.rK0 = self.orderParam * self.K0 
        bias = Ytest - self.Ypred 
        var = self.rK0 - np.sum(self.K0_invK * self.rK0X, axis=1)
        predLoss = bias**2 + var 
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()

    # New functions below - adapted from GaussianProcess class

    def computeDKernel(self):
        """Compute D_kernel (derivative of kernel) based on activation function"""
        
        if self.activation == 'linear':
            self.D_kernel = np.ones((self.C.shape[0], self.C.shape[0]))
        elif self.activation == 'erf':
            det_factor = (
                (1 + 2 * self.CX)[:, np.newaxis] *
                (1 + 2 * self.CX)[np.newaxis, :]
                - 4 * self.C ** 2
            )
            self.D_kernel = (4 / np.pi) / np.sqrt(np.maximum(det_factor, 1e-10)) 
        elif self.activation == 'relu':
            self.D_kernel = kernels.computeKmatrix(self.C, kernels.deriv_kernel_relu)
            #norm_complete = np.sqrt(self.CX[:, None] * self.CX[None, :])
            #rho_complete = self.C / norm_complete
            ##rho_complete = np.clip(rho_complete, -1.0 + 1e-12, 1.0 - 1e-12) #add numerical stability
            #self.D_kernel = (np.pi - np.arccos(rho_complete)) / (2 * np.pi)
            #self.D_kernel = 1/4 + 1/(2 * np.pi* np.arccos(rho_complete) )
        else:
            print("Warning: D_kernel not implemented for this activation function. Using default value of ones.")
            # Default: return ones
            self.D_kernel = np.ones((self.C.shape[0], self.C.shape[0]))

        return self.D_kernel

    def wwDominantContribution(self):
        """Compute dominant contribution to WW (weight-weight) correlation"""
        return (1/self.l0) * np.eye(self.N0)

    def wwSecondTerm(self):
        """Compute second term of WW contribution"""
        self.RK_diag_inv = 1 / (self.T + self.orderParam * self.eigvalK)
        prefactor = (1/self.l0) * self.optQ / (self.l1 * self.N1)
        
        # Construct K_tensor using eigen decomposition
        K_tensor = np.matmul(self.eigvecK, np.matmul(np.diag(self.RK_diag_inv), self.eigvecK.T)) * self.D_kernel
        
        var = np.matmul(self.Xtrain.T, np.matmul(K_tensor, self.Xtrain))
        return -prefactor * var / (self.l0 * self.N0)

    def wwThirdTerm(self):
        """Compute third term of WW contribution"""
        
        prefactor = (1/self.l0) * self.optQ / (self.l1 * self.N1)
        
        diagonal_K_r = np.diag(self.RK_diag_inv)
        m = np.matmul(self.eigvecK, np.matmul(diagonal_K_r, self.yT))
        mX = m[:, np.newaxis] * self.Xtrain
        var = np.matmul(mX.T, np.matmul(self.D_kernel[:self.P, :self.P], mX))
        
        return prefactor * var / (self.l0 * self.N0)
    
    def training_loss(self):
        inv_TK_R = self.T * self.RK_diag_inv
        inv_Kr_squared = np.diag(inv_TK_R**2)
        Term1 = np.dot(inv_TK_R, self.optQ / self.l1 * self.eigvalK)
        Term2 = np.dot(self.yT, np.dot(inv_Kr_squared, self.yT))
        return 1 / self.P * (Term1 + Term2)

