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
    
### quick LLM additions for better optimize

    def _build_rKL(self, Q):
        """Build rKL(Q) and return (rKL, meanpref) in torch."""
        meanpref = 0.0
        for l in range(self.L):
            product = 1.0 if (l == self.L - 1) else torch.prod(Q[0, l+1:])
            meanpref = meanpref + (Q[0, l] - 1.0/(1.0 + Q[1, l])) * product
        rKL = torch.prod(Q[0]) * self.thetaL - meanpref * self.ML
        return rKL, meanpref

    def effectiveAction_solve(self, Q, jitter0=1e-12, jitter_growth=10.0, max_tries=10,
                              require_pd=True):
        """
        Robust version:
          - uses cholesky_ex (no exception)
          - retries with increasing jitter
          - if still not PD:
              * if require_pd: raises RuntimeError (for current iterate, should not happen)
              * else: returns +inf (for line-search trial points)
        """
        rKL, _ = self._build_rKL(Q)
        A = rKL + self.T * torch.eye(self.P, dtype=torch.float64, device=Q.device)
        A = 0.5 * (A + A.T)  # symmetrize

        jitter = jitter0
        L = None
        info = None
        for _ in range(max_tries):
            tryA = A + jitter * torch.eye(self.P, dtype=A.dtype, device=A.device)
            L, info = torch.linalg.cholesky_ex(tryA)
            if int(info.item()) == 0 and torch.isfinite(L).all():
                A = tryA
                break
            jitter *= jitter_growth

        if L is None or int(info.item()) != 0:
            if require_pd:
                raise RuntimeError("A is not PD even after jitter escalation.")
            # for trial points in line search: mark as infeasible
            return torch.tensor(torch.inf, dtype=torch.float64, device=Q.device)

        sol = torch.cholesky_solve(self.y[:, None], L)[:, 0]
        quad = torch.dot(self.y, sol)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

        return (- torch.sum(Q[0]*Q[1])
                + torch.sum(torch.log(1. + Q[1]))
                + (1/self.N1) * quad
                + (1/self.N1) * logdet)
    
    def _Q_from_params(self, params, eps=1e-8):
        """
        params: torch tensor shape (2, L), unconstrained
        returns Q with constraints:
          Q0 > 0, 1+Q1 > 0
        """
        u, v = params[0], params[1]
        Q0 = torch.nn.functional.softplus(u) + eps
        Q1 = torch.nn.functional.softplus(v) - 1.0 + eps
        return torch.stack([Q0, Q1], dim=0)

    def optimize_newton(self, Q0=None, tol=1e-6, max_iter=80, damping=1e-3,
                        backtrack=0.5, ls_max=12, verbose=True):
        """
        Damped Newton on the root condition grad_Q S(Q) = 0, but in unconstrained params.
        Typically much fewer iterations than Adam on ||grad||^2.
        """
        device = self.y.device
        dtype = torch.float64

        if Q0 is None:
            Q0 = np.stack([np.ones(self.L), np.zeros(self.L)], axis=0)
        Q0 = torch.tensor(Q0, dtype=dtype, device=device)

        # initialize params so that Q_from_params(params) ~ Q0
        # softplus^{-1}(x) approx: log(exp(x)-1)
        def sp_inv(x):
            return torch.log(torch.expm1(torch.clamp(x, min=1e-8)))

        u0 = sp_inv(torch.clamp(Q0[0], min=1e-8))
        v0 = sp_inv(torch.clamp(Q0[1] + 1.0, min=1e-8))  # since Q1 = softplus(v)-1
        params = torch.stack([u0, v0], dim=0).detach().requires_grad_(True)

        self.optState = False
        self.optEpochs = 0

        for it in tqdm(range(max_iter), desc="Newton opt.", disable=not verbose):
            Q = self._Q_from_params(params)

            S = self.effectiveAction_solve(Q)
            g = torch.autograd.grad(S, params, create_graph=True)[0]  # shape (2,L)
            g_flat = g.reshape(-1)

            gnorm = torch.linalg.norm(g_flat).item()
            if verbose:
                tqdm.write(f"iter {it}: ||grad||={gnorm:.3e}")

            if gnorm < tol:
                self.optState = True
                break

            # Jacobian of g_flat wrt params_flat (i.e. Hessian of S in params)
            def grad_flat(p_flat):
                p = p_flat.reshape(2, self.L).requires_grad_(True)
                Qp = self._Q_from_params(p)
                Sp = self.effectiveAction_solve(Qp)
                gp = torch.autograd.grad(Sp, p, create_graph=True)[0]
                return gp.reshape(-1)

            p_flat = params.reshape(-1)
            J = torch.autograd.functional.jacobian(grad_flat, p_flat, create_graph=False)
            J = J.detach()  # we solve a linear system; no need to backprop through the solve

            # Damped solve: (J + λI) δ = -g
            lam = damping
            I = torch.eye(J.shape[0], dtype=dtype, device=device)
            rhs = (-g_flat.detach())
            # Try increasing damping if solve fails / is ill-conditioned
            for _ in range(6):
                try:
                    delta = torch.linalg.solve(J + lam * I, rhs)
                    if torch.isfinite(delta).all():
                        break
                except RuntimeError:
                    pass
                lam *= 10.0
            else:
                # fallback: steepest descent in params on 0.5||g||^2
                delta = rhs
            
            
            # Backtracking line search on F = 0.5||grad||^2 (robust)
            F0 = 0.5 * (g_flat.detach() @ g_flat.detach())
            step = 1.0
            accepted = False
            
            for _ in range(ls_max):
                new_params = (p_flat + step * delta).reshape(2, self.L)
            
                # IMPORTANT: trial eval should NOT crash if A not PD
                Qn = self._Q_from_params(new_params)
                Sn = self.effectiveAction_solve(Qn, require_pd=False)
            
                if not torch.isfinite(Sn):
                    step *= backtrack
                    continue
            
                gn = torch.autograd.grad(Sn, new_params, create_graph=False)[0].reshape(-1)
                if not torch.isfinite(gn).all():
                    step *= backtrack
                    continue
            
                Fn = 0.5 * (gn @ gn)
                if torch.isfinite(Fn) and Fn < F0:
                    params = new_params.detach().requires_grad_(True)
                    accepted = True
                    break
            
                step *= backtrack

            self.optEpochs += 1

        # store solution in same format as your Adam routine
        Qopt = self._Q_from_params(params).detach().cpu().numpy()
        self.optQ = Qopt

        # compute thetapref / meanpref like before
        self.meanpref = 0.0
        for l in range(self.L):
            product = 1.0 if (l == self.L - 1) else np.prod(self.optQ[0, l+1:])
            self.meanpref += (self.optQ[0, l] - 1.0/(1.0 + self.optQ[1, l])) * product
        self.thetapref = float(np.prod(self.optQ[0]))

        if verbose:
            print(f"opt state: {self.optState}, iters: {self.optEpochs}, thetapref.: {self.thetapref:.3f}, meanpref.: {self.meanpref:.3f}")
            print(f"Qs[0]: {self.optQ[0]}")
            print(f"Qs[1]: {self.optQ[1]}")

    def _project_Q(self, Q, q0_min=1e-6, q1_min=1e-6):
        Q0 = torch.clamp(Q[0], min=q0_min)
        Q1 = torch.clamp(Q[1], min=-1.0 + q1_min)
        return torch.stack([Q0, Q1], dim=0)

    def optimize_newton_projected(self, Q0=None, tol=1e-6, max_iter=25,
                                  damping=1e-3, backtrack=0.5, ls_max=12,
                                  q0_min=1e-5, q1_min=1e-5,
                                  verbose=True):
        """
        Newton on g(Q)=∇_Q S(Q)=0 directly in Q-space, with projection to keep feasibility.
        Avoids softplus saturation causing huge meanpref and stuck params.
        """
        device = self.y.device
        dtype = torch.float64

        if Q0 is None:
            Q0 = np.stack([np.ones(self.L), np.zeros(self.L)], axis=0)
        Q = torch.tensor(Q0, dtype=dtype, device=device)
        Q = self._project_Q(Q, q0_min=q0_min, q1_min=q1_min).detach().requires_grad_(True)

        self.optState = False
        self.optEpochs = 0

        for it in tqdm(range(max_iter), desc="Newton(Q) opt.", disable=not verbose):
            # Evaluate action; for the current iterate, require PD (if not PD even with jitter, that's real trouble)
            S = self.effectiveAction_solve(Q, require_pd=True)
            g = torch.autograd.grad(S, Q, create_graph=True)[0]  # shape (2,L)
            g_flat = g.reshape(-1)
            gnorm = torch.linalg.norm(g_flat).item()

            if verbose:
                tqdm.write(f"iter {it}: ||grad_Q||={gnorm:.3e}  "
                           f"min(1+Q1)={(1.0+Q[1].min()).item():.3e}  "
                           f"min(Q0)={Q[0].min().item():.3e}")

            if gnorm < tol:
                self.optState = True
                break

            # Jacobian of g_flat wrt Q_flat (i.e. Hessian of S in Q coords)
            def grad_flat(Q_flat):
                QQ = Q_flat.reshape(2, self.L)
                QQ = self._project_Q(QQ, q0_min=q0_min, q1_min=q1_min)
                QQ = QQ.requires_grad_(True)
            
                SS = self.effectiveAction_solve(QQ, require_pd=True)  # <-- important
                gg = torch.autograd.grad(SS, QQ, create_graph=True)[0]
                return gg.reshape(-1)
            
            Q_flat = Q.reshape(-1)
            J = torch.autograd.functional.jacobian(grad_flat, Q_flat, create_graph=False).detach()
            I = torch.eye(J.shape[0], dtype=dtype, device=device)

            # Solve (J + λI) δ = -g
            lam = damping
            rhs = (-g_flat.detach())
            for _ in range(6):
                try:
                    delta = torch.linalg.solve(J + lam * I, rhs)
                    if torch.isfinite(delta).all():
                        break
                except RuntimeError:
                    pass
                lam *= 10.0
            else:
                # fallback to steepest descent in Q-space
                delta = rhs

            # Line search on merit F=0.5||g(Q)||^2, rejecting non-PD trial points
            F0 = 0.5 * (g_flat.detach() @ g_flat.detach())
            step = 1.0
            accepted = False
            for _ in range(ls_max):
                Q_try = (Q_flat + step * delta).reshape(2, self.L)
                Q_try = self._project_Q(Q_try, q0_min=q0_min, q1_min=q1_min)

                S_try = self.effectiveAction_solve(Q_try, require_pd=False)
                if not torch.isfinite(S_try):
                    step *= backtrack
                    continue

                g_try = torch.autograd.grad(S_try, Q_try, create_graph=False)[0].reshape(-1)
                if not torch.isfinite(g_try).all():
                    step *= backtrack
                    continue

                F_try = 0.5 * (g_try @ g_try)
                if torch.isfinite(F_try) and F_try < F0:
                    Q = Q_try.detach().requires_grad_(True)
                    accepted = True
                    break

                step *= backtrack

            if not accepted:
                # tiny safe step
                Q = self._project_Q((Q_flat + 1e-2 * delta).reshape(2, self.L),
                                    q0_min=q0_min, q1_min=q1_min).detach().requires_grad_(True)

            self.optEpochs += 1

        # Store and compute prefactors as before
        self.optQ = Q.detach().cpu().numpy()

        self.meanpref = 0.0
        for l in range(self.L):
            product = 1.0 if (l == self.L - 1) else np.prod(self.optQ[0, l+1:])
            self.meanpref += (self.optQ[0, l] - 1.0/(1.0 + self.optQ[1, l])) * product
        self.thetapref = float(np.prod(self.optQ[0]))

        if verbose:
            print(f"opt state: {self.optState}, iters: {self.optEpochs}, thetapref.: {self.thetapref:.3f}, meanpref.: {self.meanpref:.3f}")
            print(f"Qs[0]: {self.optQ[0]}")
            print(f"Qs[1]: {self.optQ[1]}")            
            
    ### first needs debugging: 
    def optimize_broyden(self, Q0=None, tol=1e-6, max_iter=120, verbose=True):
        """
        Broyden's method to solve grad S(params)=0 in unconstrained params space.
        Uses inverse-Jacobian approximation; needs only grad evaluations.
        """
        device = self.y.device
        dtype = torch.float64

        if Q0 is None:
            Q0 = np.stack([np.ones(self.L), np.zeros(self.L)], axis=0)
        Q0 = torch.tensor(Q0, dtype=dtype, device=device)

        def sp_inv(x):
            return torch.log(torch.expm1(torch.clamp(x, min=1e-8)))

        u0 = sp_inv(torch.clamp(Q0[0], min=1e-8))
        v0 = sp_inv(torch.clamp(Q0[1] + 1.0, min=1e-8))
        params = torch.stack([u0, v0], dim=0).detach().requires_grad_(True)

        n = 2 * self.L
        B = torch.eye(n, dtype=dtype, device=device)  # inverse Jacobian approx

        # def g_of(p):
        #     Q = self._Q_from_params(p)
        #     S = self.effectiveAction_solve(Q)
        #     g = torch.autograd.grad(S, p, create_graph=False)[0]
        #     return g.reshape(-1)
        def g_of(p):
            p = p.detach().requires_grad_(True)
            Q = self._Q_from_params(p)
            S = self.effectiveAction_solve(Q, require_pd=False)
            if not torch.isfinite(S):
                return torch.full((2*self.L,), torch.inf, dtype=torch.float64, device=p.device)
            g = torch.autograd.grad(S, p, create_graph=False)[0]
            return g.reshape(-1).detach()

        p = params.detach()
        g = g_of(p)
        self.optState = False
        self.optEpochs = 0

        for it in tqdm(range(max_iter), desc="Broyden opt.", disable=not verbose):
            gnorm = torch.linalg.norm(g).item()
            if verbose:
                tqdm.write(f"iter {it}: ||grad||={gnorm:.3e}")
            if gnorm < tol:
                self.optState = True
                break

            # direction
            s = -B @ g

            # mild damping / line search on ||g|| (very simple but effective)
            step = 1.0
            accepted = False
            for _ in range(10):
                p_new = (p.reshape(-1) + step * s).reshape(2, self.L)
                g_new = g_of(p_new)
                if torch.isfinite(g_new).all() and torch.linalg.norm(g_new) < torch.linalg.norm(g):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                p_new = (p.reshape(-1) + 1e-2 * s).reshape(2, self.L)
                g_new = g_of(p_new)

            # Broyden update for inverse Jacobian
            y = (g_new - g)  # delta g
            s_eff = (p_new - p).reshape(-1)  # delta p
            denom = (s_eff @ (B @ y))
            if torch.abs(denom) > 1e-14 and torch.isfinite(denom):
                By = B @ y
                B = B + torch.outer((s_eff - By), (s_eff @ B)) / denom

            p, g = p_new.detach(), g_new.detach()
            self.optEpochs += 1

        Qopt = self._Q_from_params(p).detach().cpu().numpy()
        self.optQ = Qopt

        self.meanpref = 0.0
        for l in range(self.L):
            product = 1.0 if (l == self.L - 1) else np.prod(self.optQ[0, l+1:])
            self.meanpref += (self.optQ[0, l] - 1.0/(1.0 + self.optQ[1, l])) * product
        self.thetapref = float(np.prod(self.optQ[0]))

        if verbose:
            print(f"opt state: {self.optState}, iters: {self.optEpochs}, thetapref.: {self.thetapref:.3f}, meanpref.: {self.meanpref:.3f}")
            print(f"Qs[0]: {self.optQ[0]}")
            print(f"Qs[1]: {self.optQ[1]}")