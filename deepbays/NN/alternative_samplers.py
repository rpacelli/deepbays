import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch


# ---------- helpers ----------

def _iter_params(param_groups: list[dict]) -> Iterable[torch.nn.Parameter]:
    for g in param_groups:
        for p in g["params"]:
            if p.requires_grad:
                yield p

def _zero_grads(param_groups: list[dict]) -> None:
    for g in param_groups:
        for p in g["params"]:
            if p.grad is not None:
                p.grad.zero_()

def _loss_is_bad(x: torch.Tensor) -> bool:
    return (not torch.isfinite(x)) or torch.isnan(x).item()


# ---------- closure (training func) ------
def make_closure(model, data, targets, criterion, grads=True):
    # # Example usage:
    # closure = make_closure(model, Xtrain, ytrain, criterion)
    # accepted, log_alpha = optimizer.step(closure)
    def closure():
        if grads:
            with torch.enable_grad(): # this undoes locally the @torch.no_grad() decorator of step()
                out = model(data)
                loss = criterion(out, targets)      # ← likelihood only, NO weight prior
                loss.backward()                     # compute grads for ∇L
        else:
            out = model(data)
            loss = criterion(out, targets)      # ← likelihood only, NO weight prior
        return loss
    return closure

def train_MALA(model, X, y, criterion, optimizer):
    closure = make_closure(model, X, y, criterion, grads=True)
    accepted, log_alpha, loss = optimizer.step(closure)
    return loss

def train_pCN(model, X, y, criterion, optimizer):
    closure = make_closure(model, X, y, criterion, grads=False)
    accepted, log_alpha, loss = optimizer.step(closure)
    return loss


# ============================================================
#                    MALA (recommended)
# ============================================================

@dataclass
class MALAStats:
    tried: int = 0
    accepted: int = 0
    last_log_alpha: float = float("nan")

class MALAOpt(torch.optim.Optimizer):
    """
    Metropolis-Adjusted Langevin Algorithm for BNN weights at temperature T.

    Target (up to const):  log pi(theta) = -(1/T) * L(theta) - 0.5 * sum_j lambda_j ||theta_j||^2
    Proposal: theta' ~ N(theta + h * M * grad log pi(theta), 2h M), with h = lr * T.
      - Default M = I (Euclidean MALA). You can pass a fixed diagonal preconditioner per group.

    CONTRACT for `closure()` (caller-defined):
      * Must compute the full-batch likelihood loss (NO prior term),
      * Must call backward() to populate .grad with ∇L,
      * Must return a scalar 0-dim tensor (the loss value).

    Notes:
      * We evaluate loss+grads at current and proposed states (two forwards per step).
      * Preconditioner M must be treated as CONSTANT during sampling to preserve correctness.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        temperature: float,
        priors: Sequence[float],
        preconditioners: Optional[Sequence[float]] = None,
        # generator: Optional[torch.Generator] = None,
    ):
        # Build param groups: one group per (Linear) layer parameters
        param_groups = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                params = [p for p in layer.parameters() if p.requires_grad]
                if params:
                    param_groups.append({"params": params})

        if len(param_groups) == 0:
            # Fallback: single group with all trainable params
            params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{"params": params}]

        defaults = {"lr": float(lr), "temperature": float(temperature)}
        super().__init__(param_groups, defaults)

        assert len(priors) == len(self.param_groups), "priors length must match param groups"
        for g, lam in zip(self.param_groups, priors):
            assert lam > 0. , f"encountered flat (or even negative) prior precision {lam}"
            g["lambda"] = float(lam)

        if preconditioners is None:
            preconditioners = [1.0] * len(self.param_groups)
        assert len(preconditioners) == len(self.param_groups), "preconditioners length must match groups"
        for g, m in zip(self.param_groups, preconditioners):
            # Use a scalar diagonal metric per group (constant)
            g["mass"] = float(max(m, 1e-12))

        # self._gen = generator if generator is not None else torch.Generator(device=next(_iter_params(self.param_groups)).device)
        self.stats = MALAStats()

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.stats.tried == 0 else self.stats.accepted / self.stats.tried

    def _accumulate_prior_quadratic(self, param_groups: list[dict]) -> float:
        s = 0.0
        for g in param_groups:
            lam = g["lambda"]
            for p in g["params"]:
                s += 0.5 * lam * torch.sum(p.detach() * p.detach()).item()
        return s

    def _grad_log_posterior(self, temperature: float) -> list[torch.Tensor]:
        """Return list aligned with params: ∇ log π = -(1/T) ∇L - λ p."""
        out = []
        for g in self.param_groups:
            lam = g["lambda"]
            for p in g["params"]:
                # p.grad is ∇L
                gl = p.grad
                if gl is None:
                    raise RuntimeError("closure() must call backward(), but no grad found.")
                out.append((-1.0 / temperature) * gl.detach() - lam * p.detach())
        return out

    @torch.no_grad()
    def step(self, closure):
        """
        Performs ONE MALA iteration. Returns (accepted: bool, log_alpha: float).
        """
        T = self.defaults["temperature"]
        lr = self.defaults["lr"]

        # Effective time step ‘h’ and noise scaling with (optional) group mass
        # We implement MALA with M as scalar per group by reparametrizing proposals per group.
        # For M = I, this reduces to the standard form.
        _zero_grads(self.param_groups)
        loss_cur = closure()  # computes ∇L(theta)
        if _loss_is_bad(loss_cur):
            raise RuntimeError("closure() returned non-finite loss at current state.")
        Lc = float(loss_cur.detach())

        # Cache current params
        current_params = [p.detach().clone() for p in _iter_params(self.param_groups)]

        # Compute grad log posterior at current
        gradlogpi_cur = self._grad_log_posterior(T)

        # Prior quadratic at current
        prior_q_cur = self._accumulate_prior_quadratic(self.param_groups)

        # Propose
        idx = 0
        noises = []
        # Also accumulate q(prop|cur) quadratic efficiently group-wise
        quad_prop_given_cur = 0.0
        for g in self.param_groups:
            mass = g["mass"]
            h = lr * T * mass
            std = math.sqrt(2.0 * h)
            for p in g["params"]:
                eps = torch.randn_like(p)#, generator=self._gen)
                noises.append(eps)
                mean = p + h * gradlogpi_cur[idx]
                prop = mean + std * eps
                p.copy_(prop)  # load proposal in-place

                # accumulate ||prop - mean||^2 / (4h)
                diff = (prop - mean).flatten()
                quad_prop_given_cur += (diff @ diff).item() / (4.0 * h)
                idx += 1

        # Evaluate proposed loss + grads (needed for reverse proposal)
        _zero_grads(self.param_groups)
        loss_prop = closure()
        if _loss_is_bad(loss_prop):
            # Reject bad proposals safely
            # revert parameters
            it = iter(current_params)
            for p in _iter_params(self.param_groups):
                p.copy_(next(it))
            _zero_grads(self.param_groups)
            self.stats.tried += 1
            self.stats.last_log_alpha = float("-inf")
            return False, self.stats.last_log_alpha

        Lp = float(loss_prop.detach())

        # grad log posterior at proposed
        gradlogpi_prop = self._grad_log_posterior(T)
        prior_q_prop = self._accumulate_prior_quadratic(self.param_groups)

        # Build MH terms
        logpi_cur = -(Lc / T) - prior_q_cur
        logpi_prop = -(Lp / T) - prior_q_prop

        # q(cur | prop)
        idx = 0
        quad_cur_given_prop = 0.0
        for g in self.param_groups:
            mass = g["mass"]
            h = lr * T * mass
            for p in g["params"]:
                mean_rev = p + h * gradlogpi_prop[idx]  # mean at proposed
                # current_params are the "x" here
                cur = current_params[idx]
                diff = (cur - mean_rev).flatten()
                quad_cur_given_prop += (diff @ diff).item() / (4.0 * h)
                idx += 1

        logq_prop_given_cur = -quad_prop_given_cur
        logq_cur_given_prop = -quad_cur_given_prop
        log_alpha = (logpi_prop - logpi_cur) + (logq_cur_given_prop - logq_prop_given_cur)

        self.stats.tried += 1
        if not math.isfinite(log_alpha):                    # guard rail
            accept = False
        elif log_alpha >= 0. :                              # accept without thinking
            accept = True
        else:
            accept = (torch.rand(()).item() < math.exp(log_alpha))   # decide stochastically based on MH
        # accept = (math.log(torch.rand(()).item()) < log_alpha) # since torch.rand() can give 0.0 and produce error in math.log, this is unstable

        if accept:
            self.stats.accepted += 1
            # We keep proposed params already loaded.
        else:
            # revert to previous params
            it = iter(current_params)
            for p in _iter_params(self.param_groups):
                p.copy_(next(it))

        # Hygiene: caller's next step starts from a clean grad state.
        _zero_grads(self.param_groups)
        self.stats.last_log_alpha = float(log_alpha)
        return accept, float(log_alpha), Lc


# ============================================================
#                             pCN
# ============================================================

@dataclass
class pCNStats:
    tried: int = 0
    accepted: int = 0
    last_log_alpha: float = float("nan")

class pCNOpt(torch.optim.Optimizer):
    """
    Preconditioned Crank–Nicolson proposals for Gaussian priors (precision lambda_j per group).
    Proposal preserves the prior exactly; MH ratio depends only on the (tempered) likelihood.
    The main advantage of pCN is that the accept rate does not deteriorate trivially with dimension (which is the case for MALA).
    This is because the proposals always remain on the shell of the prior norm, instead of tending away from the shell.
    The proposals of this sampler are not gradient based, and only diffuse through MH acceptance steps.
    Because of this, acceptance rates (and thus the optimal step size beta) very strongly deteriorate for low temperature, 
    but the sampler is robust and efficient at moderate temperatures, even in high dimensions.

    theta' = sqrt(1 - beta^2) * theta + beta * Lambda^{-1/2} * xi,  xi ~ N(0, I)

    CONTRACT for `closure()`:
      * Must return full-batch likelihood loss (NO prior), scalar tensor.
      * No backward() needed; we evaluate under torch.inference_mode().

    Notes:
      * Temperature T appears only in the MH ratio: log alpha = -(L(theta') - L(theta)) / T
      * Choose beta so that acceptance ~ 0.3–0.6 (typical); larger beta explores faster but accepts less.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        beta: float,
        temperature: float,
        priors: Sequence[float],
        # generator: Optional[torch.Generator] = None,
    ):
        assert 0.0 < beta < 1.0, "beta must be in (0,1)"
        param_groups = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                params = [p for p in layer.parameters() if p.requires_grad]
                if params:
                    param_groups.append({"params": params})

        if len(param_groups) == 0:
            params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{"params": params}]

        defaults = {"beta": float(beta), "temperature": float(temperature)}
        super().__init__(param_groups, defaults)

        assert len(priors) == len(self.param_groups), "priors length must match param groups"
        for g, lam in zip(self.param_groups, priors):
            assert lam > 0. , f"encountered flat (or even negative) prior precision {lam}, not allowed for pCN"
            g["lambda"] = float(lam)

        # self._gen = generator if generator is not None else torch.Generator(device=next(_iter_params(self.param_groups)).device)
        self.stats = pCNStats()

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.stats.tried == 0 else self.stats.accepted / self.stats.tried

    @torch.no_grad()
    def step(self, closure):
        beta = self.defaults["beta"]
        T = self.defaults["temperature"]

        # cache current params and current loss
        current_params = [p.detach().clone() for p in _iter_params(self.param_groups)]

        with torch.inference_mode():
            Lc = float(closure().detach())
        if not math.isfinite(Lc):
            raise RuntimeError("closure() returned non-finite loss at current state.")

        # propose prior-preserving move
        root = math.sqrt(max(1.0 - beta * beta, 0.0))
        proposed_params = []
        for g in self.param_groups:
            lam = g["lambda"]
            inv_sqrt_lam = 1.0 / math.sqrt(lam) # requirement lam > 0 already checked at init. if lam==0, prior is flat and pCN proposal would explode
            for p in g["params"]:
                eps = torch.randn_like(p)#, generator=self._gen)
                prop = root * p + beta * inv_sqrt_lam * eps
                proposed_params.append(prop)

        # load proposal
        it = iter(proposed_params)
        for p in _iter_params(self.param_groups):
            p.copy_(next(it))

        # evaluate proposed likelihood
        with torch.inference_mode():
            Lp = float(closure().detach())

        # accept/reject based on MH ratio (tempered) log_alpha
        log_alpha = -(Lp - Lc) / T
        self.stats.tried += 1
        if not math.isfinite(log_alpha):                    # guard rail
            accept = False
        elif log_alpha >= 0. :                              # accept without thinking
            accept = True
        else:
            accept = (torch.rand(()).item() < math.exp(log_alpha))   # decide stochastically based on MH
        # accept = (math.log(torch.rand(()).item()) < log_alpha) and math.isfinite(log_alpha)  # torch.rand can give exactly 0.0, then math.log() throws error -> not long term stable

        if accept:
            self.stats.accepted += 1
        else:
            # revert
            it = iter(current_params)
            for p in _iter_params(self.param_groups):
                p.copy_(next(it))

        self.stats.last_log_alpha = float(log_alpha)
        return accept, float(log_alpha), Lc

