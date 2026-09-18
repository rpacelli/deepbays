"""Dense, deterministic Laplace evidence for a coupled softmax GP.

This is function-space Laplace at fixed Q, not Laplace over network weights.
The covariance derivative includes the implicit derivative of the fitted
mode. All vectors use example-major order; no kernel inverse or jitter is
used. Intended initially for small problems such as P=50, C=10.
"""

from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from scipy.linalg import cho_solve, helmert
from scipy.special import logsumexp, softmax, ndtri
from scipy.stats import qmc
from ..conv_geometry import positive_int
from ._laplace_covariance import (DenseCovariance, SeparableCovariance, apply_blocks,
                                  root_and_factor, reduction_from_factor)


class LaplaceConvergenceError(FloatingPointError):
    """The conditional latent mode did not meet its residual tolerance."""


def contrast_basis(classes):
    classes = positive_int(classes, 'classes')
    if classes < 2:
        raise ValueError('classification requires at least two classes')
    return helmert(classes, full=False).T


def class_labels(y, count, classes):
    y = np.asarray(y)
    if (y.shape != (count,) or not np.all(np.isfinite(y))
            or not np.all(y == np.floor(y)) or np.any(y < 0) or np.any(y >= classes)):
        raise ValueError(f'labels must be integer class indices of shape ({count},) in [0,{classes})')
    return y.astype(np.int64, copy=True)


def softmax_terms(g, y, basis, beta):
    """Summed beta*CE, gradient, local Hessians, and probabilities."""
    g = np.asarray(g).reshape(len(y), basis.shape[1])
    logits = g @ basis.T
    p = softmax(logits, axis=1)
    value = beta * np.sum(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    residual = p.copy()
    residual[np.arange(len(y)), y] -= 1
    gradient = beta * residual @ basis
    projected = p @ basis
    W = beta * (np.einsum('ma,ai,aj->mij', p, basis, basis)
                - np.einsum('mi,mj->mij', projected, projected))
    return float(value), gradient.reshape(-1), W, p


@dataclass
class LaplaceState:
    nll: float
    gradient: np.ndarray
    mode: np.ndarray
    alpha: np.ndarray
    reduction: np.ndarray
    probabilities: np.ndarray
    mode_residual: float
    iterations: int
    objective_history: tuple
    posterior_blocks: np.ndarray
    seconds: float
    _covariance: object = field(repr=False)
    _posterior_covariance: object = field(default=None, repr=False)

    @property
    def posterior_covariance(self):
        """Full latent covariance, materialized only when explicitly accessed."""
        if self._posterior_covariance is None:
            covariance = self._covariance
            kr = covariance.matmul(self.reduction)
            posterior = covariance.dense() - covariance.matmul(kr.T).T
            self._posterior_covariance = (posterior+posterior.T)/2
        return self._posterior_covariance

    @property
    def explicit_gradient(self):
        """Fixed-mode part of the evidence derivative, primarily for checks."""
        return .5*(self.reduction-np.outer(self.alpha, self.alpha))


def fit_laplace(K, y, basis, beta=1., *, mode_tol=1e-10, maxiter=100,
                alpha0=None, max_dense_size=2000, verbose=False):
    """Return -log Z_Laplace and its full Frobenius covariance derivative.

    basis may be the orthonormal contrast basis or an identity matrix for
    independent full-logit checks. beta multiplies summed cross-entropy; it
    does not rescale logits inside softmax. beta=0 is a useful prior limit.
    """
    started = perf_counter()
    basis = np.asarray(basis, dtype=float)
    maxiter = positive_int(maxiter, 'maxiter')
    max_dense_size = positive_int(max_dense_size, 'max_dense_size')
    if basis.ndim != 2 or min(basis.shape) < 1 or not np.all(np.isfinite(basis)):
        raise ValueError('basis must be a finite matrix')
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(mode_tol) or mode_tol <= 0:
        raise ValueError('invalid beta or mode tolerance')
    c = basis.shape[1]
    shape = K.shape if isinstance(K, SeparableCovariance) else np.shape(K)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0 or shape[0] % c:
        raise ValueError('K must be square with dimension P * number of latent coordinates')
    n = shape[0]
    if n > max_dense_size:
        raise ValueError('dense Laplace size limit exceeded; reduce P or increase max_dense_size')
    if verbose:
        print(f'Laplace: dimension={n}, validating covariance and fitting mode', flush=True)
    covariance = K if isinstance(K, SeparableCovariance) else DenseCovariance(K, c)
    if covariance.c != c:
        raise ValueError('covariance and contrast basis dimensions differ')
    P = n//c
    y = class_labels(y, P, basis.shape[0])
    alpha = np.zeros(n) if alpha0 is None else np.asarray(alpha0, dtype=float).copy()
    if alpha.shape != (n,) or not np.all(np.isfinite(alpha)):
        raise ValueError('alpha0 must be finite with shape (P*c,)')
    history = []
    for iteration in range(maxiter + 1):
        g = covariance.matmul(alpha)
        loss, grad, W, p = softmax_terms(g, y, basis, beta)
        objective = loss + .5 * (alpha @ g)
        history.append(float(objective))
        residual = float(np.linalg.norm(alpha + grad, ord=np.inf))
        if verbose:
            print(f'  Newton {iteration}: objective={objective:.9g}, residual={residual:.3g}, elapsed={perf_counter()-started:.1f}s', flush=True)
        if residual <= mode_tol:
            break
        if iteration == maxiter:
            raise LaplaceConvergenceError(f'latent mode residual {residual:.3g} exceeds {mode_tol:.3g}')
        roots, factor = root_and_factor(covariance, W)
        b = np.einsum('mab,mb->ma', W, g.reshape(P, c)).reshape(-1) - grad
        proposal = b - apply_blocks(roots, cho_solve(factor, apply_blocks(roots, covariance.matmul(b)), check_finite=False))
        direction = proposal - alpha
        slope = float(covariance.matmul(alpha + grad) @ direction)
        step = 1.
        for _ in range(40):
            trial = alpha + step * direction
            trial_g = covariance.matmul(trial)
            trial_logits = trial_g.reshape(P, c) @ basis.T
            trial_loss = beta*np.sum(logsumexp(trial_logits, axis=1)-trial_logits[np.arange(P), y])
            trial_objective = trial_loss + .5 * (trial @ trial_g)
            roundoff = 16 * np.finfo(float).eps * (1 + abs(objective))
            if trial_objective <= objective + 1e-4 * step * min(slope, 0.) + roundoff:
                alpha = trial
                break
            step *= .5
        else:
            raise LaplaceConvergenceError('Newton line search failed')

    roots, factor = root_and_factor(covariance, W)
    nll = objective + np.log(np.diag(factor[0])).sum()
    reduction = reduction_from_factor(roots, factor)
    # Only local covariance blocks enter the third-derivative contraction.
    blocks = covariance.posterior_blocks(reduction)
    blocks = (blocks+blocks.transpose(0, 2, 1))/2
    A = np.einsum('ai,mij,bj->mab', basis, blocks, basis, optimize=True)
    v = np.diagonal(A, axis1=1, axis2=2) - 2 * np.einsum('mab,mb->ma', A, p)
    h = (.5 * beta * (p * (v - np.sum(p * v, axis=1, keepdims=True))) @ basis).reshape(-1)
    # d mode = (I - K reduction) dK alpha.
    t = h - reduction @ covariance.matmul(h)
    gradient = .5*reduction
    # Form rank-one terms in small row batches rather than full n-by-n temporaries.
    for start in range(0, n, 64):
        rows = slice(start, min(n, start+64))
        gradient[rows] += (.5*t[rows, None]*alpha[None, :]
                           + .5*alpha[rows, None]*(t-alpha)[None, :])
    if not np.isfinite(nll) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError('non-finite Laplace evidence or gradient')
    seconds = perf_counter()-started
    if verbose:
        print(f'Laplace done: nll={nll:.9g}, Newton steps={iteration}, residual={residual:.3g}, time={seconds:.1f}s', flush=True)
    return LaplaceState(float(nll), gradient, g.reshape(P, c), alpha, reduction,
                        p, residual, iteration, tuple(history), blocks, seconds, covariance)


def gaussian_softmax_probabilities(mean, covariance, basis, samples=4096, seed=0):
    """Integrate softmax under per-example Gaussian marginals using scrambled Sobol.

    This approximates the predictive integral, not the training evidence.
    A power-of-two sample count preserves Sobol balance. No extra likelihood
    temperature is applied to predictive logits.
    """
    return _gaussian_softmax_integrals(mean, covariance, basis, samples, seed, statistics=False)


def gaussian_softmax_statistics(mean, covariance, basis, samples=4096, seed=0):
    """Mean/variance of softmax and class-argmax probabilities under a Gaussian.

    For a true class y, argmax_probabilities[i, y] is the expected correctness
    of one posterior function draw at example i. Averaging these marginals
    gives expected test accuracy without needing joint test covariances.
    Their average is not the distribution of accuracy of a whole function
    draw. Ties use NumPy's first-maximum convention, also used by predict().
    """
    return _gaussian_softmax_integrals(mean, covariance, basis, samples, seed, statistics=True)


def _gaussian_softmax_integrals(mean, covariance, basis, samples, seed, *, statistics):
    samples = positive_int(samples, 'samples')
    if samples & (samples - 1):
        raise ValueError('samples must be a power of two')
    mean, covariance, basis = (np.asarray(x, dtype=float) for x in (mean, covariance, basis))
    if basis.ndim != 2 or min(basis.shape) < 1 or not np.all(np.isfinite(basis)):
        raise ValueError('basis must be a finite matrix')
    c = basis.shape[1]
    if (mean.ndim != 2 or len(mean) == 0 or mean.shape[1] != c
            or covariance.shape != (len(mean), c, c)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))):
        raise ValueError('invalid Gaussian marginal shapes')
    values, vectors = np.linalg.eigh((covariance + covariance.transpose(0, 2, 1)) / 2)
    if values.min() < -1e-9 * max(1., float(np.max(np.abs(values)))):
        raise FloatingPointError('negative predictive covariance beyond roundoff')
    roots = vectors * np.sqrt(np.maximum(values, 0.))[:, None, :]
    uniform = qmc.Sobol(c, scramble=True, seed=seed).random_base2(samples.bit_length() - 1)
    z = ndtri(np.clip(uniform, np.finfo(float).eps, 1 - np.finfo(float).eps))
    result = np.empty((len(mean), basis.shape[0]))
    if statistics:
        variance, winners = np.empty_like(result), np.empty_like(result)
    for i in range(len(mean)):
        logits = (mean[i] + z @ roots[i].T) @ basis.T
        probabilities = softmax(logits, axis=1)
        result[i] = probabilities.mean(axis=0)
        if statistics:
            variance[i] = np.mean((probabilities - result[i])**2, axis=0)
            winners[i] = np.bincount(logits.argmax(axis=1), minlength=basis.shape[0]) / samples
    if statistics:
        return dict(probabilities=result, probability_variance=variance, argmax_probabilities=winners)
    return result
