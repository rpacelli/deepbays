import numpy as np
from .. import kernels
import torch
from tqdm import tqdm

class FC_1HL_nonodd_nonzerotemp():
    """ 
    1 hidden layer theory following from a non-central Wishart Ansatz,
    (= taking into account that the prior activations have a non-zero mean, e.g. for ReLU).
    
    Here, compared to FC_1HL_nonodd(), a different way to write the action is used where 
    the dependence of the kernel on the mean term is expanded explicitly in the action,
    so as to render the terms interpretable and argue about the relative importance of the 
    mean-dependent terms compared to the vanilla (zero-mean) terms in the action.
             
    """
    def __init__(self, N1, T, priors=1., act="relu", device='cuda'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device()) # typically 'cuda:0'
        else:
            self.device = torch.device('cpu')
        self.N1 = N1
        self.T = T
        if type(priors) == float:
            self.priors = [priors] * 2
        else:
            assert len(priors) == 2, "priors misspecified"
            self.priors = priors
        self.kernel = eval(f"kernels.kernel_{act}")
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.mean = eval(f"kernels.mean_{act}")

    def preprocess(self, X, Y, tsolve=True): # tsolve uses torch.linalg.solve instead of torch.inverse, should be faster
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.alpha = self.P / self.N1
        self.corrNorm = 1/(self.N0 * self.priors[0])
        self.C = np.dot(X, X.T) * self.corrNorm
        self.y = Y.squeeze().to(torch.float32).to(self.device)
        # self.y.requires_grad = False

        # precompute raw kernel and [mu, y]^T stack on gpu, for calls to torch.linalg.solve() 
        self.theta = torch.tensor(self.kernel(self.C.diagonal()[:, None], self.C, self.C.diagonal()[None, :]), device=self.device)
        self.mu = torch.tensor(self.mean(self.C.diagonal()), device=self.device)
        self.muy = torch.stack([self.mu, self.y], axis=1) # shape (P, 2)
        
        # Tmu, Ty = torch.linalg.solve(regtheta, muy).T.to('cpu') # gives e.g. Tmu = inverse(regtheta) @ mu
        # # precompute scalars in action
        # self.muTmu = 1./self.P * np.dot(self.mu, Tmu)
        # self.muTy  = 1./self.P * np.dot(self.mu, Ty)
        # self.yTy   = 1./self.P * np.dot(self.y, Ty) 
        # self.logdettheta = torch.logdet(regtheta).to('cpu').item()
        
    def K(self, Q):
        return self.T * torch.eye(self.P, device=self.device) + Q[0] * self.theta / self.priors[1]
    
    def get_K_Kinv_muy(self, Q):
        K = self.K(Q)
        Kinv_mu, Kinv_y = torch.linalg.solve(K, self.muy).T # gives e.g. Tmu = inverse(regtheta) @ mu
        return K, Kinv_mu, Kinv_y
        
    def diffQ(self, Q):
        return Q[0] - 1. / (1 + Q[1])
        
    def term1(self, K):
        return self.alpha / self.P * torch.logdet(K)
    
    def term2(self, Q, Kinv_y):
        return self.alpha / self.P * self.y @ Kinv_y
    
    def term3(self, Q, Kinv_mu):
        return self.alpha / self.P * torch.log(1. - self.diffQ(Q) / self.priors[1] * self.mu @ Kinv_mu)
    
    def term4(self, Q, Kinv_mu):
        return self.alpha / self.P * self.diffQ(Q) / self.priors[1]   \
               * (self.y @ Kinv_mu)**2 / (1. - self.diffQ(Q) / self.priors[1] * self.mu @ Kinv_mu)

    def effectiveAction(self, Q):
        """
        Action as in Pacelli et al'23, eq.(105), but expanded the explicit dependence on mean quantities
        Note:
            - see Paolo's short write-up effective-action-with-mean-simplified.tex
            - typo-difference to action in Pacelli'23: here without the \beta factor in the TrLog which appears to be a typo in the paper).
        """
        assert type(Q) == torch.Tensor, f"Q should be type torch.Tensor but is {type.Q}"
        assert Q.device == self.device, f"Q is not on {self.device} but on {Q.device}"
        
        K, Kinv_mu, Kinv_y = self.get_K_Kinv_muy(Q)
        
        return (- Q[0] * Q[1]
                + torch.log(1. + Q[1])
                + self.term1(K)
                + self.term2(Q, Kinv_y)
                + self.term3(Q, Kinv_mu)
                + self.term4(Q, Kinv_mu)
                )
        
    def optimize_adam(self, Q0=None, lr=0.02, tolerance=1e-4, max_epochs=20000, verbose=True):
        if Q0 is not None:
            assert len(Q0) == 2, "Q0 init should e.g. be [1., 0.]"
        else:
            Q0 = np.array([1., 0.])
        Q0 = torch.tensor(Q0).to(self.device)
        Q = Q0.clone().detach().requires_grad_(True)

        opt = torch.optim.Adam([Q], lr=lr)
        # opt = torch.optim.SGD([Q], lr=lr)
        self.optState = False
        self.optEpochs = 0
        
        # print(f"Init t1: {self.term1(Q0):.3f}, t2: {self.term2(Q0):.3f}, t3: {self.term3(Q0):.3f}, t4: {self.term4(Q0):.3f}")
        
        tqiter = tqdm(range(max_epochs), desc='Adam opt.', disable=not verbose)
        for step in tqiter:
            opt.zero_grad()
            S_val = self.effectiveAction(Q)
            # important to keep graph to enable higher-order grads (next) !
            (gradS,) = torch.autograd.grad(S_val, Q, create_graph=True)
            # optimization target: square norm of gradients dS/dQ, to find saddle-point
            Loss = 0.5 * (gradS.pow(2).sum())
            Loss.backward()
            opt.step()
            
            tqiter.set_postfix({'loss':Loss.item()})
            # if step % 10 == 0:
            #     print(f'{step}: loss {Loss.item():.3f}, S {S_val.item():.3f}, grads: {gradS.detach().numpy()}')
            self.optEpochs += 1
            if gradS.detach().norm() < tolerance:
                self.optState = True
                break

        self.optQ = Q.detach().to('cpu').numpy()
        
        # print(f"Opt  t1: {self.term1(Q):.3f}, t2: {self.term2(Q):.3f}, t3: {self.term3(Q):.3f}, t4: {self.term4(Q):.3f}")
        if verbose:
            print(f"Qs: [{self.optQ[0]:.3f}, {self.optQ[1]:.3f}], opt state: {self.optState}, epochs: {self.optEpochs}, Qbar - 1/(1+Q): {self.diffQ(self.optQ):.3f}")

    def computeActionGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, device=self.device, requires_grad = True)
        f = self.effectiveAction(Q)
        f.backward()
        return Q.grad.data.detach().to('cpu').numpy()
    
    def optimize(self, Q0 = [1., 0.]):
        from scipy.optimize import fsolve
        assert len(Q0) == 2
        self.optQ = fsolve(self.computeActionGrad, Q0, xtol = 1e-9)
        resgrads = self.computeActionGrad(self.optQ)
        isClose = np.isclose(resgrads, np.zeros(2), rtol=1e-4, atol=1e-7) # tests additionally for smallness of grads, not only closeness to true root
        self.converged = isClose.all()
        print(f"is fsolve solution close to zero? {isClose}, {resgrads}")   
        print(f"fsolve optQ is {self.optQ}")

    def computeFullTrainTestKernel(self, X, Xtest):
        Xtt = torch.concat([X, Xtest], axis=0)
        Ctt = np.dot(Xtt, Xtt.T) * self.corrNorm
        # precompute raw kernel and mean, now we do things mostly in numpy
        self.thetatt = self.kernel(Ctt.diagonal()[:, None], Ctt, Ctt.diagonal()[None, :])
        self.mutt = self.mean(Ctt.diagonal())
        self.Mtt = np.outer(self.mutt, self.mutt)
        self.rKtt = (self.optQ[0] / self.priors[1] * self.thetatt  # self.T * np.eye(len(self.thetatt)) + 
                     - self.diffQ(self.optQ) / self.priors[1] * self.Mtt)
        
    def debug_plot(self, low=[0.1, -0.5], high=[8., 8.], n=50, grads=True):
        Qbars = np.linspace(low[0], high[0], num=n)
        Qs = np.linspace(low[1], high[1], num=n)
        Qgrid = torch.tensor(np.stack(np.meshgrid(Qbars, Qs)), device='cpu')
        Sgrid = self.effectiveAction(Qgrid)
        arcsinhSgrid = np.arcsinh(Sgrid)
        
        import matplotlib.pyplot as plt
        h = plt.contourf(Qbars, Qs, arcsinhSgrid)
        plt.axis('scaled')
        plt.colorbar()
        plt.title('arcsinh(action)')
        plt.ylabel('Q')
        plt.xlabel('Qbar')
        plt.show()
        
        if grads:
            gradlossgrid = np.zeros_like(Sgrid)
            for i in range(n):
                for j in range(n):
                    gradlossgrid[i,j] = np.sum(self.computeActionGrad(Qgrid[:,i,j])**2) # seems not to work out of box as vectorized op
            arcsinhgradlossgrid = np.arcsinh(gradlossgrid)
            
            h2 = plt.contourf(Qbars, Qs, arcsinhgradlossgrid)
            plt.axis('scaled')
            plt.colorbar()
            plt.title('arcsinh(||dS/dQs||^2)')
            plt.ylabel('Q')
            plt.xlabel('Qbar')
            plt.show()

    def predict(self, Xtest, tsolve=False):
        self.Ptest = len(Xtest)
        P = self.P
        Pt = self.Ptest
        self.computeFullTrainTestKernel(self.X, Xtest)  # compute rKtt
        if tsolve:   
            A = torch.tensor(self.T * np.eye(self.P) + self.rKtt[:P, :P], dtype=torch.float32, device=self.device)
            self.K0_invK = torch.linalg.solve(A, torch.tensor(self.rKtt[-Pt:, :P].T, device=self.device)).T.to('cpu').numpy()
        else:
            self.invK = np.linalg.inv(self.T * np.eye(self.P) + self.rKtt[:P, :P])
            self.K0_invK = np.matmul(self.rKtt[-Pt:, :P], self.invK)
        self.Ypred = np.dot(self.K0_invK, self.Y).reshape(-1, 1)
        return self.Ypred

    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred
        var = self.rKtt.diagonal()[-self.Ptest:] - \
            np.sum(self.K0_invK * self.rKtt[-self.Ptest:, :self.P], axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()




class FC_1HL_nonodd_zerotemp():
    """ 
    1 hidden layer theory following from a non-central Wishart Ansatz,
    (= taking into account that the prior activations have a non-zero mean, e.g. for ReLU),
    specialized to the T -> 0 case. 
    
    Here, compared to FC_1HL_nonodd(), by considering T=0 the action could be simplified into a form 
    where all matrix operations needed do not depend on Q, Qbar and therefore only need to be performed 
    a single time during preprocess() -> Computing the action and derivatives during optimization
    iterations only involves scalars and is much cheaper (i.e. no PxP matrix inversion per iteration).
             
    """
    def __init__(self, N1, priors=1., act="relu", regularization=1e-8, device='cuda'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N1 = N1
        if type(priors) == float:
            self.priors = [priors] * 2
        else:
            assert len(priors) == 2, "priors misspecified"
            self.priors = priors
        self.kernel = eval(f"kernels.kernel_{act}")
        self.kernelTorch = eval(f"kernels.kernel_{act}_torch")
        self.mean = eval(f"kernels.mean_{act}")
        self.reg = regularization

    def preprocess(self, X, Y, tsolve=True): # tsolve uses torch.linalg.solve instead of torch.inverse, should be faster
        self.X = X
        self.Y = Y
        self.P, self.N0 = X.shape
        self.alpha = self.P / self.N1
        self.corrNorm = 1/(self.N0 * self.priors[0])
        self.C = np.dot(X, X.T) * self.corrNorm
        self.y = Y.squeeze().to(torch.float64).numpy()
        # self.y.requires_grad = False

        # precompute raw kernel and mean vector, and inverse kernel with torch
        self.theta = self.kernel(self.C.diagonal()[:, None], self.C, self.C.diagonal()[None, :])
        regtheta = torch.tensor(self.theta + self.reg * np.eye(self.P), device=self.device)
        self.mu = self.mean(self.C.diagonal())
        if tsolve:
            muy = torch.tensor(np.stack([self.mu, self.y], axis=1), device=self.device) # shape (P, 2)
            Tmu, Ty = torch.linalg.solve(regtheta, muy).T.to('cpu') # gives e.g. Tmu = inverse(regtheta) @ mu
            # precompute scalars in action
            self.muTmu = 1./self.P * np.dot(self.mu, Tmu)
            self.muTy  = 1./self.P * np.dot(self.mu, Ty)
            self.yTy   = 1./self.P * np.dot(self.y, Ty)   # when using y here, the loss does not move...
        else:
            self.thetainv = torch.inverse(regtheta).to('cpu').numpy() # done with torch on gpu if avail, faster than numpy
            # precompute scalars in action
            self.muTmu = 1./self.P * self.mu.T @ self.thetainv @ self.mu
            self.muTy  = 1./self.P * self.mu.T @ self.thetainv @ self.y
            self.yTy   = 1./self.P * self.y.T @ self.thetainv @ self.y
        self.logdettheta = torch.logdet(regtheta).to('cpu').item()
        
    def diffQ(self, Q):
        return Q[0] - 1. / (1 + Q[1])
        
    def term1(self, Q):
        return self.alpha * torch.log(Q[0] / self.priors[1])
    
    def term2(self, Q):
        return self.alpha * self.priors[1] * self.yTy / Q[0]
    
    def term3(self, Q):
        return self.alpha * torch.log(1. - self.diffQ(Q) / Q[0] * self.P * self.muTmu)
    
    def term4(self, Q):
        return self.alpha * self.priors[1] * (self.diffQ(Q) / Q[0]) / (Q[0] - self.diffQ(Q) * self.P * self.muTmu) * self.P * self.muTy**2

    def effectiveAction(self, Q):
        """
        Action as in Pacelli et al'23, eq.(105), but for beta -> infty and simplified
        Note:
            - typo-difference to action in Pacelli'23: here without the \beta factor in the TrLog which appears to be a typo in the paper).
        """
        return (- Q[0] * Q[1]
                + torch.log(1. + Q[1])
                + self.alpha * self.logdettheta / self.P
                + self.term1(Q)
                + self.term2(Q)
                + self.term3(Q)
                + self.term4(Q)
                )
        
    def optimize_adam(self, Q0=None, lr=0.02, tolerance=1e-4, max_epochs=20000, verbose=True):
        if Q0 is not None:
            assert len(Q0) == 2, "Q0 init should e.g. be [1., 0.]"
        else:
            Q0 = np.array([1., 0.])
        Q0 = torch.tensor(Q0)
        Q = Q0.clone().detach().requires_grad_(True)

        # opt = torch.optim.Adam([Q], lr=lr)
        opt = torch.optim.SGD([Q], lr=lr)
        self.optState = False
        self.optEpochs = 0
        
        # print(f"Init t1: {self.term1(Q0):.3f}, t2: {self.term2(Q0):.3f}, t3: {self.term3(Q0):.3f}, t4: {self.term4(Q0):.3f}")
        
        tqiter = tqdm(range(max_epochs), desc='Adam opt.', disable=not verbose)
        for step in tqiter:
            opt.zero_grad()
            S_val = self.effectiveAction(Q)
            # important to keep graph to enable higher-order grads (next) !
            (gradS,) = torch.autograd.grad(S_val, Q, create_graph=True)
            # optimization target: square norm of gradients dS/dQ, to find saddle-point
            Loss = 0.5 * (gradS.pow(2).sum())
            Loss.backward()
            opt.step()
            
            tqiter.set_postfix({'loss':Loss.item()})
            # if step % 10 == 0:
            #     print(f'{step}: loss {Loss.item():.3f}, S {S_val.item():.3f}, grads: {gradS.detach().numpy()}')
            self.optEpochs += 1
            if gradS.detach().norm() < tolerance:
                self.optState = True
                break

        self.optQ = Q.detach().numpy()
        print(f"Opt  t1: {self.term1(Q):.3f}, t2: {self.term2(Q):.3f}, t3: {self.term3(Q):.3f}, t4: {self.term4(Q):.3f}")
        
        if verbose:
            print(f"Qs: [{self.optQ[0]:.3f}, {self.optQ[1]:.3f}], opt state: {self.optState}, epochs: {self.optEpochs}, Qbar - 1/(1+Q): {self.diffQ(self.optQ):.3f}")

    def computeActionGrad(self, Q):
        Q = torch.tensor(Q, dtype = torch.float64, requires_grad = True)
        f = self.effectiveAction(Q)
        f.backward()
        return Q.grad.data.detach().numpy()
    
    def optimize(self, Q0 = [1., 0.]):
        from scipy.optimize import fsolve
        assert len(Q0) == 2
        self.optQ = fsolve(self.computeActionGrad, Q0, xtol = 1e-8)
        isClose = np.isclose(self.computeActionGrad(self.optQ), np.zeros(2)) # tests additionally for smallness of grads, not only closeness to true root
        self.converged = isClose.all()
        print("is fsolve solution close to zero?", isClose)   
        print(f"fsolve optQ is {self.optQ}")

    def computeFullTrainTestKernel(self, X, Xtest):
        Xtt = torch.concat([X, Xtest], axis=0)
        Ctt = np.dot(Xtt, Xtt.T) * self.corrNorm
        # precompute raw kernel and mean, as torch.tensors
        self.thetatt = self.kernel(Ctt.diagonal()[:, None], Ctt, Ctt.diagonal()[None, :])
        self.mutt = self.mean(Ctt.diagonal())
        self.Mtt = np.outer(self.mutt, self.mutt)
        self.rKtt = (self.optQ[0] / self.priors[1] * self.thetatt 
                     - self.diffQ(self.optQ) / self.priors[1] * self.Mtt)
        
    def debug_plot(self, low=[0.1, -0.5], high=[8., 8.], n=50, grads=True):
        Qbars = np.linspace(low[0], high[0], num=n)
        Qs = np.linspace(low[1], high[1], num=n)
        Qgrid = torch.tensor(np.stack(np.meshgrid(Qbars, Qs)), device='cpu')
        Sgrid = self.effectiveAction(Qgrid)
        arcsinhSgrid = np.arcsinh(Sgrid)
        
        import matplotlib.pyplot as plt
        h = plt.contourf(Qbars, Qs, arcsinhSgrid)
        plt.axis('scaled')
        plt.colorbar()
        plt.title('arcsinh(action)')
        plt.ylabel('Q')
        plt.xlabel('Qbar')
        plt.show()
        
        if grads:
            gradlossgrid = np.zeros_like(Sgrid)
            for i in range(n):
                for j in range(n):
                    gradlossgrid[i,j] = np.sum(self.computeActionGrad(Qgrid[:,i,j])**2) # seems not to work out of box as vectorized op
            arcsinhgradlossgrid = np.arcsinh(gradlossgrid)
            
            h2 = plt.contourf(Qbars, Qs, arcsinhgradlossgrid)
            plt.axis('scaled')
            plt.colorbar()
            plt.title('arcsinh(||dS/dQs||^2)')
            plt.ylabel('Q')
            plt.xlabel('Qbar')
            plt.show()

    def predict(self, Xtest, tsolve=True):
        self.Ptest = len(Xtest)
        P = self.P
        Pt = self.Ptest
        self.computeFullTrainTestKernel(self.X, Xtest)  # compute rKtt
        if tsolve:   
            A = torch.tensor(self.rKtt[:P, :P] + self.reg * np.eye(P), dtype=torch.float32, device=self.device)
            self.K0_invK = torch.linalg.solve(A, torch.tensor(self.rKtt[-Pt:, :P].T, device=self.device)).T.to('cpu').numpy()
        else:
            A = self.rKtt[:P, :P] + (self.reg) * np.eye(P)
            self.invK = np.linalg.inv(A)
            self.K0_invK = np.matmul(self.rKtt[-Pt:, :P], self.invK)
        self.Ypred = np.dot(self.K0_invK, self.Y).reshape(-1, 1)
        return self.Ypred

    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred
        var = self.rKtt.diagonal()[-self.Ptest:] - \
            np.sum(self.K0_invK * self.rKtt[-self.Ptest:, :self.P], axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
