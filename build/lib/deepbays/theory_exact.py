import numpy as np
from scipy.optimize import minimize

def kmatrix(C,kernel):
    P = len(C)
    K = np.zeros_like(C)
    for i in range(P): 
        for j in range(i,P):         
            K[i][j] = kernel(C[i][i], C[i][j], C[j][j])
            K[j][i] = K[i][j]
    return K

def kernel_erf(k0xx,k0xy,k0yy):
    return (2/(np.pi))*np.arcsin((2*k0xy)/np.sqrt((1+2*k0xx)*(1+2*k0yy)))

def kappa1(u):
    return (1/(2*np.pi))*(u * (np.pi - np.arccos(u))+ np.sqrt(1-u**2))   

def kernel_relu(k0xx, k0xy, k0yy):
    if k0xy == k0xx:
        return np.sqrt(k0xx*k0yy)/(2)
    else:
        u = k0xy/np.sqrt(k0xx*k0yy)
        kappa = kappa1(u)
        return np.sqrt(k0xx*k0yy)*kappa

def effectiveAction(x, *params):
    T, P, N1, lambda1, diag_K, K_eigval, ytilde = params 
    A = T * np.identity(P) + (x/lambda1) * diag_K
    invA = np.linalg.inv(A)
    return ( x - np.log(x)
        + (1/N1) * np.sum(np.log(T + x*K_eigval/lambda1))
        + (1/N1) * np.dot(ytilde, np.dot(invA, ytilde)) )


class prop_width_GP_1HL():
    def __init__(self, N0, N1, lambda0, lambda1, act, T):
        self.N0 = N0 
        self.N1 = N1
        self.l0 = lambda0
        self.l1 = lambda1 
        self.T = T
        self.corrNorm = 1/(self.N0*self.l0)
        self.kernel = eval(f"kernel_{act}")

    def computeAveragePrediction(self, data, labels, testData, testLabels):
        self.P = len(data)
        self.Ptest = len(testData)
        C = np.dot(data, data.T)*self.corrNorm
        K = kmatrix(C, self.kernel)
        #print(K)
        [eigvalK, eigvecK] = np.linalg.eig(K)
        U = eigvecK
        Udag = np.transpose(U)
        diagK = np.diagflat(eigvalK)
        yT = np.matmul( Udag, labels.squeeze())
        bns = ((1e-8,np.inf),)
        params = self.T, self.P,self.N1, self.l1, diagK, eigvalK, yT
        x0 = 1.0 #initial condition
        res = minimize(effectiveAction, x0, bounds=bns, tol=1e-20, args = params)
        Qopt = (res.x).item()
        print(f"Qopt is {Qopt}")
        A = (Qopt/self.l1)*K+ (self.T)*np.eye(self.P)
        invK = np.linalg.inv(A)
        predLoss = 0
        for p in range(self.Ptest):
            x0, y0 = np.array(testData[p]), np.array(testLabels[p])
            predLoss += self.computeSinglePrediction(data, labels, x0, y0, Qopt, invK)
        predLoss = predLoss/self.Ptest
        return predLoss

    def computeSinglePrediction(self, data, labels, x0, y0, Qopt, invK):
        k0xx = np.dot(x0,x0)*self.corrNorm
        Kmu = np.random.randn(self.P)
        orderParam = Qopt/self.l1
        for i in range(self.P):
            k0xy = np.dot(x0,data[i])*self.corrNorm
            k0yy = np.dot(data[i],data[i])*self.corrNorm
            Kmu[i] = orderParam *self.kernel(k0xx,k0xy,k0yy) 
        k0xx = orderParam *self.kernel(k0xx,k0xx,k0xx) 
        K0_invK = np.matmul(Kmu, invK)
        bias = -np.dot(K0_invK, labels) + y0
        var = -np.dot(K0_invK, Kmu) + k0xx
        predErr = bias**2 + var
        return predErr.item()

    
