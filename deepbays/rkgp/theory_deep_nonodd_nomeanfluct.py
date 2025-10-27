import numpy as np
from .. import kernels
import torch
from tqdm import tqdm


class FC_deep_nonodd_nomeanfluct():
    """ 
    Theory following from a non-central Wishart Ansatz,
    (= taking into account that the prior activations have a non-zero mean, e.g. for ReLU),
    under the hypothesis that the fluctuations of the mean activations under the prior are neglegible.
    
    This assumption strongly simplifies the theory, by enabling the treatment of deeper layers beyond
    L=1 in a simple iteration of non-central Wishart Ansätzen which keep the form of the action as in 
    Pacelli et al. (2023) Nat. Mach. Int, only adding a series of L terms of the form 
    meanpref(Q,Qbar) * muL.dot(muL.T)
    to the rescaled kernel. Here muL is the vector of last-layer prior-activation means on the training data samples.
    
    Warning: Note however, that the assumption of neglegible mean fluctuations is not self-evidently 
             correct. Taking such fluctuations into account would require additional order parameters,
             and could be worth exploring. Yet the idea is here that if the means themselves only
             contribute a small change to the predictions, then the fluctuations of the means should
             correspondingly contribute less. Therefore, if including the fluctuations would make a large
             difference, also the present approximate theory should deviate significantly from the vanilla
             zero-mean theory.
             
    """
    def __init__(self, N1, T, L, priors=1., act="relu"):
        self.N1 = N1
        if type(priors) == float:
            self.priors = [priors] * (L+1)
        else:
            assert len(priors) == L+1, "priors misspecified"
            self.priors = priors
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
        self.thetaL = torch.tensor(self.C, dtype=torch.float64, requires_grad=False)
        for l in range(self.L):
            self.thetaL = ((1. / self.priors[l+1])) * self.kernelTorch(
                self.thetaL.diagonal()[:, None], self.thetaL, self.thetaL.diagonal()[None, :])
            if l == self.L - 2:
                self.thetaLm1_diag = self.thetaL.diagonal()  # needed for the mean vector muL when L>1
        if self.L == 1: # for one layer the mean activations are a direct function of C
            self.muL = torch.tensor(self.mean(self.C.diagonal()), dtype=torch.float64, requires_grad=False)
        else:
            self.muL = torch.tensor(self.mean(self.thetaLm1_diag), dtype=torch.float64, requires_grad=False)
        self.ML = torch.outer(self.muL, self.muL)

    def effectiveAction(self, Q):
        """
        Action as in Pacelli et al'23, eq.(105), but with extended definition of the rescaled kernel:
            rKL = thetapref * K - meanpref * muL.dot(muL.T)
        where
            thetapref = \prod_{l=1}^L \bar{Q}_l
            meanpref  = \sum_{l=1}^L [(\bar{Q}_l - 1 / (1 + Q_l)) \prod_{r=l+1}^L Q_r]
        Note:
            - in the last prod in meanpref only Qs of the layers above l contribute, for l=L the factor is set to 1.)
            - typo-difference to action in Pacelli'23: here without the \beta factor in the TrLog which appears to be a typo in the paper).
        """
        meanpref = 0.
        for l in range(self.L):
            if l == self.L-1:
                product = 1.
            else:
                product = torch.prod(Q[0, l+1:])
            meanpref += (Q[0, l] - 1./(1 + Q[1, l])) * product
        rKL = torch.prod(Q[0]) * self.thetaL - meanpref * self.ML
        A = rKL + self.T * torch.eye(self.P)
        invA = torch.inverse(A)
        return (- torch.sum(Q[0]*Q[1])
                + torch.sum(torch.log(1. + Q[1]))
                + (1/self.N1) * torch.matmul(self.y, torch.matmul(invA, self.y))
                + (1/self.N1) * torch.logdet(A)
                )

    def optimize_adam(self, Q0=None, lr=0.02, tolerance=1e-4, max_epochs=5000, verbose=True):

        if Q0 is not None:
            assert Q0.shape == (2, self.L), "Q0 init not in correct shape"
        else:
            Q0 = np.stack([np.ones(self.L), np.zeros(self.L)], axis=0)
        Q0 = torch.tensor(Q0)
        Q = Q0.clone().detach().requires_grad_(True)

        opt = torch.optim.Adam([Q], lr=lr)
        self.optState = False
        self.optEpochs = 0
        
        tqiter = tqdm(range(max_epochs), desc='Adam opt.', disable=not verbose)
        for step in tqiter:
            opt.zero_grad()
            S_val = self.effectiveAction(Q)
            # important to keep graph to enable higher-order grads (next) !
            (gradS,) = torch.autograd.grad(S_val, Q, create_graph=True)
            # optimization target: square norm of gradients dS/dQ, to find saddle-point
            Loss = 0.5 * (gradS.pow(2).sum())
            tqiter.set_postfix({'loss':Loss.item()})

            Loss.backward()
            opt.step()
            self.optEpochs += 1
            if gradS.detach().norm() < tolerance:
                self.optState = True
                break

        self.optQ = Q.detach().numpy()
        self.meanpref = 0.
        for l in range(self.L):
            if l == self.L-1:
                product = 1.
            else:
                product = np.prod(self.optQ[0, l+1:])
            self.meanpref += (self.optQ[0, l] -
                              1./(1 + self.optQ[1, l])) * product
        self.thetapref = np.prod(self.optQ[0])
        if verbose:
            print(f"opt state: {self.optState}, epochs: {self.optEpochs}, thetapref.: {self.thetapref:.3f}, meanpref.: {self.meanpref:.3f}")
            print(f"Qs[0]: {self.optQ[0]}")
            print(f"Qs[1]: {self.optQ[1]}")

    def computeFullTrainTestKernel(self, X, Xtest):
        Xtt = torch.concat([X, Xtest], axis=0)
        Ctt = np.dot(Xtt, Xtt.T) * self.corrNorm
        # precompute raw kernel and mean, as torch.tensors
        self.thetaLtt = Ctt
        for l in range(self.L):
            self.thetaLtt = ((1. / self.priors[l+1])) * self.kernel(self.thetaLtt.diagonal()[
                :, None], self.thetaLtt, self.thetaLtt.diagonal()[None, :])
            if l == self.L - 1:
                self.thetaLm1tt_diag = self.thetaLtt.diagonal()  # needed for the mean vector muL
        self.muLtt = self.mean(self.thetaLm1tt_diag)
        self.MLtt = np.outer(self.muLtt, self.muLtt)
        self.rKLtt = self.thetapref * self.thetaLtt - self.meanpref * self.MLtt

    def predict(self, Xtest):
        self.Ptest = len(Xtest)
        P = self.P
        Pt = self.Ptest

        self.computeFullTrainTestKernel(self.X, Xtest)  # compute rKLtt
        A = self.rKLtt[:P, :P] + (self.T) * np.eye(P)
        self.invK = np.linalg.inv(A)
        self.K0_invK = np.matmul(self.rKLtt[-Pt:, :P], self.invK)
        self.Ypred = np.dot(self.K0_invK, self.Y).reshape(-1, 1)
        return self.Ypred

    def averageLoss(self, Ytest):
        bias = Ytest - self.Ypred
        var = self.rKLtt.diagonal()[-self.Ptest:] - \
            np.sum(self.K0_invK * self.rKLtt[-self.Ptest:, :self.P], axis=1)
        predLoss = bias**2 + var
        return predLoss.mean().item(), (bias**2).mean().item(), var.mean().item()
