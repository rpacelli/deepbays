"""Private numerical tools for equal-width matrix Wishart order parameters.

Objectives take packed symmetric log(Q) coordinates. Keeping the callback at
this level lets separable kernels reuse the log-matrix eigensystem directly.
"""

import numpy as np
from scipy.optimize import minimize
from ..conv_geometry import positive_int


class SymmetricCoordinates:
    """Orthonormal coordinates for the Frobenius inner product."""

    def __init__(self, dimension):
        self.d = positive_int(dimension, "dimension")
        self.indices = np.triu_indices(self.d)
        self.scale = np.where(self.indices[0] == self.indices[1], 1., np.sqrt(2.))

    def pack(self, matrix):
        return matrix[self.indices] * self.scale

    def unpack(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != self.scale.shape:
            raise ValueError("invalid number of symmetric matrix coordinates")
        matrix = np.zeros((self.d, self.d))
        matrix[self.indices] = vector / self.scale
        matrix[(self.indices[1], self.indices[0])] = matrix[self.indices]
        return matrix

    def validate_q(self, Q):
        if hasattr(Q, "detach"):
            Q = Q.detach().cpu().numpy()
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 0:
            Q = float(Q) * np.eye(self.d)
        if (Q.shape != (self.d, self.d) or not np.all(np.isfinite(Q))
                or not np.allclose(Q, Q.T, rtol=1e-10, atol=1e-12)):
            raise ValueError(f"Q must be a finite symmetric ({self.d}, {self.d}) matrix, or a positive scalar times identity")
        Q = (Q + Q.T) / 2
        eigenvalues, vectors = np.linalg.eigh(Q)
        if eigenvalues[0] <= 0:
            raise ValueError("Q must be strictly positive definite")
        return Q, eigenvalues, vectors


def matrix_prior(log_eigenvalues, depth):
    """Return 2 I(exp(H)) and its diagonal gradient in H's eigenbasis."""
    gradient = np.expm1(log_eigenvalues / depth)
    return float(depth * gradient.sum() - log_eigenvalues.sum()), gradient


def exp_divided_differences(eigenvalues, normalized=False):
    """Frechet coefficients of exp, evaluated continuously at degeneracies.

    normalized=True returns F_ab / exp((s_a+s_b)/2). This lets a likelihood
    use symmetrically scaled scores without constructing a large dS/dQ.
    """
    distance = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    ratio = np.ones_like(distance)
    np.divide(-np.expm1(-distance), distance, out=ratio, where=distance != 0)
    exponent = distance / 2 if normalized else np.maximum(eigenvalues[:, None], eigenvalues[None, :])
    return np.exp(exponent) * ratio


def minimize_log_matrix(objective, coordinates, depth, Q0=1., maxiter=500,
                        gtol=1e-6, n_restarts=2, random_state=0):
    """L-BFGS, optional small BFGS refinement, and independent final checks.

    Returns the selected scipy result and all attempts. Additional fields on
    the selected result include Q, R, log_eigenvalues, and eigenvectors.
    A small-step success flag alone never marks a result converged.
    """
    maxiter = positive_int(maxiter, "maxiter")
    if isinstance(n_restarts, (bool, np.bool_)) or not isinstance(n_restarts, (int, np.integer)) or n_restarts < 0:
        raise ValueError("n_restarts must be a nonnegative integer")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    _, ev, basis = coordinates.validate_q(Q0)
    start = coordinates.pack((basis * np.log(ev)) @ basis.T)
    try:
        objective(start)
    except np.linalg.LinAlgError as error:
        raise ValueError("training covariance is not positive definite; at T=0 use independent inputs or a positive T") from error

    def safe_objective(x):
        try:
            value, gradient = objective(x)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("non-finite action or gradient")
            return value, gradient
        except (np.linalg.LinAlgError, FloatingPointError):
            return np.inf, np.zeros_like(x)

    def check(result):
        result.fun, result.jac = safe_objective(result.x)
        result.gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf)) if np.isfinite(result.fun) else np.inf
        result.converged = bool(np.isfinite(result.fun) and result.gradient_norm <= gtol)
        return result

    rng = np.random.default_rng(random_state)
    starts = [start]
    for k in range(n_restarts):
        offset = (-1 if k % 2 == 0 else 1) * (1 + k // 2)
        starts.append(start + offset * coordinates.pack(np.eye(coordinates.d))
                      + rng.normal(0, .05, len(start)))
    results = []
    for initial in starts:
        result = check(minimize(safe_objective, initial, jac=True, method="L-BFGS-B",
                                options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-14,
                                         "maxls": 40, "maxcor": 20}))
        results.append(result)
        # Full BFGS uses O(d^4) storage; keep it restricted to small problems.
        if not result.converged and len(start) <= 512 and np.isfinite(result.fun):
            results.append(check(minimize(safe_objective, result.x, jac=True, method="BFGS",
                                           options={"maxiter": maxiter, "gtol": gtol})))
    finite = [r for r in results if np.isfinite(r.fun)]
    if not finite:
        raise RuntimeError("all matrix saddle attempts failed; inspect temperature, priors, and input scale")
    minimum = min(r.fun for r in finite)
    tied = [r for r in finite if r.fun <= minimum + 1e-12 * (1 + abs(minimum))]
    result = min(tied, key=lambda r: (not r.converged, r.gradient_norm))
    ev, basis = np.linalg.eigh(coordinates.unpack(result.x))
    result.log_eigenvalues, result.eigenvectors = ev, basis
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result.Q = (basis * np.exp(ev)) @ basis.T
        result.R = (basis * np.exp(ev / depth)) @ basis.T
    if (not np.all(np.isfinite(result.Q)) or not np.all(np.isfinite(result.R))
            or np.linalg.eigvalsh(result.Q)[0] <= 0):
        result.converged = False
        result.message = "Q is too ill-conditioned to represent as positive definite in float64"
    return result, results
