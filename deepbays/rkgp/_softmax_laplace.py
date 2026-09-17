"""Dense, deterministic Laplace evidence for a coupled softmax GP.

This is function-space Laplace at fixed Q, not Laplace over network weights.
The covariance derivative includes the implicit derivative of the fitted
mode. All vectors use example-major order; no kernel inverse or jitter is
used. Intended initially for small problems such as P=50, C=10.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, helmert
from scipy.special import logsumexp, softmax, ndtri
from scipy.stats import qmc
from ..conv_geometry import positive_int


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


def _root_and_factor(K, W):
    values, vectors = np.linalg.eigh(W)
    tolerance = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(values))))
    if values.min() < -tolerance:
        raise FloatingPointError('softmax Hessian is not positive semidefinite')
    roots = (vectors * np.sqrt(np.maximum(values, 0.))[:, None, :]) @ vectors.transpose(0, 2, 1)
    C = block_diag(*roots)
    B = np.eye(len(K)) + C @ K @ C
    return C, cho_factor((B + B.T) / 2, lower=True, check_finite=False)


@dataclass
class LaplaceState:
    nll: float
    gradient: np.ndarray
    mode: np.ndarray
    alpha: np.ndarray
    reduction: np.ndarray
    posterior_covariance: np.ndarray
    probabilities: np.ndarray
    mode_residual: float
    iterations: int
    objective_history: tuple
    explicit_gradient: np.ndarray


def fit_laplace(K, y, basis, beta=1., *, mode_tol=1e-10, maxiter=100,
                alpha0=None, max_dense_size=2000):
    """Return -log Z_Laplace and its full Frobenius covariance derivative.

    basis may be the orthonormal contrast basis or an identity matrix for
    independent full-logit checks. beta multiplies summed cross-entropy; it
    does not rescale logits inside softmax. beta=0 is a useful prior limit.
    """
    K, basis = np.asarray(K, dtype=float), np.asarray(basis, dtype=float)
    maxiter = positive_int(maxiter, 'maxiter')
    max_dense_size = positive_int(max_dense_size, 'max_dense_size')
    if basis.ndim != 2 or min(basis.shape) < 1 or not np.all(np.isfinite(basis)):
        raise ValueError('basis must be a finite matrix')
    c = basis.shape[1]
    if K.ndim != 2 or K.shape[0] != K.shape[1] or len(K) == 0 or len(K) % c:
        raise ValueError('K must be square with dimension P * number of latent coordinates')
    if len(K) > max_dense_size:
        raise ValueError('dense Laplace size limit exceeded; this backend is for small problems')
    if (not np.all(np.isfinite(K)) or not np.allclose(K, K.T, rtol=1e-10, atol=1e-12)
            or not np.isfinite(beta) or beta < 0 or not np.isfinite(mode_tol) or mode_tol <= 0):
        raise ValueError('invalid covariance, beta, or mode tolerance')
    K = (K + K.T) / 2
    ev = np.linalg.eigvalsh(K)
    if ev[0] < -1e-10 * max(1., float(np.max(np.abs(ev)))):
        raise ValueError('K must be positive semidefinite')
    P = len(K) // c
    y = class_labels(y, P, basis.shape[0])
    alpha = np.zeros(len(K)) if alpha0 is None else np.asarray(alpha0, dtype=float).copy()
    if alpha.shape != (len(K),) or not np.all(np.isfinite(alpha)):
        raise ValueError('alpha0 must be finite with shape (P*c,)')
    history = []
    for iteration in range(maxiter + 1):
        g = K @ alpha
        loss, grad, W, p = softmax_terms(g, y, basis, beta)
        objective = loss + .5 * (alpha @ g)
        history.append(float(objective))
        residual = float(np.linalg.norm(alpha + grad, ord=np.inf))
        if residual <= mode_tol:
            break
        if iteration == maxiter:
            raise LaplaceConvergenceError(f'latent mode residual {residual:.3g} exceeds {mode_tol:.3g}')
        C, factor = _root_and_factor(K, W)
        b = np.einsum('mab,mb->ma', W, g.reshape(P, c)).reshape(-1) - grad
        proposal = b - C @ cho_solve(factor, C @ (K @ b), check_finite=False)
        direction = proposal - alpha
        slope = float((K @ (alpha + grad)) @ direction)
        step = 1.
        for _ in range(40):
            trial = alpha + step * direction
            trial_g = K @ trial
            trial_loss = softmax_terms(trial_g, y, basis, beta)[0]
            trial_objective = trial_loss + .5 * (trial @ trial_g)
            roundoff = 16 * np.finfo(float).eps * (1 + abs(objective))
            if trial_objective <= objective + 1e-4 * step * min(slope, 0.) + roundoff:
                alpha = trial
                break
            step *= .5
        else:
            raise LaplaceConvergenceError('Newton line search failed')

    C, factor = _root_and_factor(K, W)
    reduction = C @ cho_solve(factor, C, check_finite=False)
    reduction = (reduction + reduction.T) / 2
    posterior = K - K @ reduction @ K
    posterior = (posterior + posterior.T) / 2
    nll = objective + np.log(np.diag(factor[0])).sum()

    # c_mode = .5 * d_g logdet(I+K W(g)), holding K fixed.
    # Only the P local blocks of posterior covariance enter the third-
    # derivative contraction; no rank-three global derivative is formed.
    blocks = posterior.reshape(P, c, P, c)[np.arange(P), :, np.arange(P), :]
    A = np.einsum('ai,mij,bj->mab', basis, blocks, basis, optimize=True)
    v = np.diagonal(A, axis1=1, axis2=2) - 2 * np.einsum('mab,mb->ma', A, p)
    h = (.5 * beta * (p * (v - np.sum(p * v, axis=1, keepdims=True))) @ basis).reshape(-1)
    # d mode = (I - K reduction) dK alpha.
    t = h - reduction @ (K @ h)
    explicit = .5 * (reduction - np.outer(alpha, alpha))
    gradient = explicit + .5 * (np.outer(t, alpha) + np.outer(alpha, t))
    if not np.isfinite(nll) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError('non-finite Laplace evidence or gradient')
    return LaplaceState(float(nll), gradient, g.reshape(P, c), alpha, reduction,
                        posterior, p, residual, iteration, tuple(history), explicit)


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
