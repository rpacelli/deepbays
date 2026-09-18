"""Equal-width, multi-output MLPs under the central Equivalent Wishart Ansatz.

K_Q = Q (Kronecker) K_NNGP, with output-major vectorization Y.T.reshape(-1).
The partition-function exponent is -N1*S/2, where

    S = L tr(Q**(1/L)) - logdet(Q) - L*D
        + [logdet(T*I + K_Q) + y.T solve(T*I + K_Q, y)] / N1.

The optimizer uses a single eigensystem of H=log(Q), with analytic Frechet
derivatives. It never forms a PD-square covariance or re-diagonalizes exp(H).
All numerical optimization and kernel preparation use float64 on the CPU.
"""

import warnings
import numpy as np
import torch
from ..conv_geometry import layer_precisions, positive_int
from ..kernels import kernels as scalar_kernels
from ..kernels.conv_kernels import as_numpy
from ._matrix_order_parameter import (SymmetricCoordinates, matrix_prior,
                                      exp_divided_differences, minimize_log_matrix)


class FC_deep_multioutput:
    """Cached central EWA theory for a bias-free MLP with D linear outputs.

    L is the number of hidden layers, all of width N1. The full-rank Wishart
    representation requires N1 >= D; its rate assumes fixed L,D as N1,P grow.
    priors contains L+1 weight precisions, including the readout. Two entries
    abbreviate [first, remaining, ...]. gamma divides each network output.
    T is a nonnegative isotropic observation variance. At T=0 the training
    kernel must have a numerically resolved positive spectrum; no jitter is
    added. ReLU and square use the same central ansatz as the vanilla MLP.

    Targets have shape (P,D); D=1 also accepts (P,). optQ is the final product
    covariance, and optR is its positive L-th root. Default losses average
    over both examples and outputs, while the likelihood sums both axes.
    """

    def __init__(self, L, N1, D, T, priors=(1., 1.), act="erf", gamma=1.,
                 batch_size=128):
        self.L, self.N1, self.D = positive_int(L, "L"), positive_int(N1, "N1"), positive_int(D, "D")
        if self.N1 < self.D:
            raise ValueError("the SPD Wishart representation requires N1 >= D")
        if not np.isfinite(T) or T < 0:
            raise ValueError("T must be finite and nonnegative")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite")
        self.T, self.gamma = float(T), float(gamma)
        self.priors = layer_precisions(priors, self.L)
        self.act = {"quad": "quadratic"}.get(act, act)
        if self.act not in ("erf", "relu", "id", "square", "quadratic"):
            raise ValueError("supported activations are erf, relu, id, square, quadratic, and quad")
        self.kernel = getattr(scalar_kernels, "kernel_" + self.act)
        self.batch_size = positive_int(batch_size, "batch_size")
        self._coordinates = SymmetricCoordinates(self.D)
        self._ready = False
        self._reset_solution()

    def _invalidate_prediction(self):
        self._prediction = None
        self.Ypred = self.predictive_variance = self.predictive_covariance = None

    def _reset_solution(self):
        self.optQ = self.optR = self.result = self.solution_kind = None
        self.converged = False
        self.optimization_results = []
        self._solution_spectrum = self._solution_q_snapshot = None
        self._invalidate_prediction()

    def _require_preprocessed(self):
        if not self._ready:
            raise RuntimeError("call preprocess(X, Y) first")

    def _require_solution(self):
        self._require_preprocessed()
        if self.optQ is None or not self.converged:
            raise RuntimeError("call optimize() and check converged, or explicitly select setIW(), before prediction")

    def _pack(self, matrix):
        return self._coordinates.pack(matrix)

    def _unpack(self, vector):
        return self._coordinates.unpack(vector)

    def _validate_q(self, Q):
        return self._coordinates.validate_q(Q)

    def _inputs(self, X):
        X = as_numpy(X)
        if X.ndim != 2 or min(X.shape) < 1 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite nonempty array of shape (P, N0)")
        return X

    def _targets(self, Y, count):
        Y = as_numpy(Y)
        if Y.shape == (count,) and self.D == 1:
            Y = Y[:, None]
        if Y.shape != (count, self.D) or not np.all(np.isfinite(Y)):
            raise ValueError(f"Y must be finite labels of shape ({count}, {self.D})" + (" or (P,)" if self.D == 1 else ""))
        return Y

    def _activation(self, left, cross, right):
        if self.act == "relu":
            valid = (left > 0) & (right > 0)
            lv, rv = np.where(left > 0, left, 1.), np.where(right > 0, right, 1.)
            bound = np.sqrt(lv * rv)
            return np.where(valid, self.kernel(lv, np.clip(cross, -bound, bound), rv), 0.)
        return self.kernel(left, cross, right)

    def preprocess(self, X, Y):
        """Cache K, its eigensystem, rotated labels, and layer diagonals."""
        self._ready = False
        self._reset_solution()
        X = self._inputs(X).copy()
        labels = self._targets(Y, len(X)).copy()
        self._vector_labels = as_numpy(Y).ndim == 1
        self.X, self.Y = X, labels[:, 0] if self._vector_labels else labels
        self._labels = labels
        self.P, self.N0 = X.shape
        self.corrNorm = 1. / (self.N0 * self.priors[0])
        self._train_diagonals = []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            kernel = (X @ X.T) * self.corrNorm
            for l in range(self.L):
                diagonal = kernel.diagonal().copy()
                self._train_diagonals.append(diagonal)
                kernel = self._activation(diagonal[:, None], kernel, diagonal[None, :]) / self.priors[l + 1]
            kernel = ((kernel + kernel.T) / 2) / self.gamma**2
        if not np.all(np.isfinite(kernel)):
            raise FloatingPointError("non-finite FC kernel; check input scale, activation, and precisions")
        eigenvalues, vectors = np.linalg.eigh(kernel)
        scale = float(np.max(np.abs(eigenvalues)))
        self.kernel_tolerance = 32 * np.finfo(np.float64).eps * self.P * scale
        if eigenvalues[0] < -self.kernel_tolerance:
            raise ValueError("training kernel has negative eigenvalues beyond roundoff")
        negative = eigenvalues < 0
        self.kernel_spectral_correction = float(max(0., -eigenvalues[0]))
        if np.any(negative):
            warnings.warn("roundoff-scale negative kernel eigenvalues were set to zero; no positive eigenvalue floor is applied", RuntimeWarning)
            kernel += (vectors[:, negative] * -eigenvalues[negative]) @ vectors[:, negative].T
            eigenvalues[negative] = 0.
        if self.T == 0 and eigenvalues[0] <= self.kernel_tolerance:
            raise ValueError("T=0 requires a numerically resolved positive definite training kernel; use independent inputs or positive T")
        if 0 < self.T <= self.kernel_tolerance and eigenvalues[0] <= self.kernel_tolerance:
            warnings.warn("temperature is below the kernel's spectral roundoff scale; small kernel modes may be unreliable", RuntimeWarning)
        self.finalKNNGP = kernel
        self._kernel_eigenvalues, self._kernel_vectors = eigenvalues, vectors
        self._rotated_labels = vectors.T @ labels
        with np.errstate(divide="ignore"):
            self._log_kernel_eigenvalues = np.log(eigenvalues)
        self._zero_task_gram = None
        self._zero_boundary = False
        if self.T == 0:
            whitened = self._rotated_labels / np.sqrt(eigenvalues[:, None])
            self._zero_task_gram = whitened.T @ whitened
            self._kernel_logdet = self._log_kernel_eigenvalues.sum()
            self._zero_boundary = self.P >= self.N1 and np.linalg.matrix_rank(whitened) < self.D
        for array in (self.X, self.Y, self._labels, self.finalKNNGP,
                      self._kernel_eigenvalues, self._kernel_vectors,
                      self._rotated_labels, self._log_kernel_eigenvalues,
                      *self._train_diagonals):
            array.setflags(write=False)
        self._ready = True
        return self

    def _spectral_likelihood(self, s, vectors):
        """Likelihood and H-gradient in H's eigenbasis, without dense exp(H)."""
        if self.T == 0:
            task = vectors.T @ self._zero_task_gram @ vectors
            value = (self.D * self._kernel_logdet + self.P * s.sum()
                     + np.dot(np.diag(task), np.exp(-s))) / self.N1
            gradient = (self.P * np.eye(self.D)
                        - exp_divided_differences(-s) * task) / self.N1
            return value, gradient
        log_signal = self._log_kernel_eigenvalues[:, None] + s[None, :]
        log_denominator = np.logaddexp(np.log(self.T), log_signal)
        fraction = np.exp(log_signal - log_denominator)  # k*q/(T+k*q), in [0,1]
        normalized = (self._rotated_labels @ vectors) * np.exp(-.5 * log_denominator)
        score = np.sqrt(fraction) * normalized
        # Scale each column by sqrt(q) before the Gram product. This avoids
        # constructing a possibly huge physical-Q gradient only to shrink it
        # with the Frechet derivative. Repeated eigenvalues remain regular.
        gradient = (np.diag(fraction.sum(axis=0))
                    - exp_divided_differences(s, normalized=True) * (score.T @ score)) / self.N1
        value = (log_denominator.sum() + np.sum(normalized**2)) / self.N1
        return value, gradient

    def _log_action_gradient(self, coordinates):
        self._require_preprocessed()
        s, vectors = np.linalg.eigh(self._unpack(coordinates))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value, gradient = self._spectral_likelihood(s, vectors)
            prior, prior_gradient = matrix_prior(s, self.L)
            value += prior
            gradient = vectors @ (gradient + np.diag(prior_gradient)) @ vectors.T
        packed = self._pack((gradient + gradient.T) / 2)
        if not np.isfinite(value) or not np.all(np.isfinite(packed)):
            raise FloatingPointError("non-finite action or gradient")
        return float(value), packed

    def computeActionGrad(self, Q):
        """Analytic Frobenius gradient with respect to physical SPD Q.

        This diagnostic may be poorly scaled near singular Q; optimization
        uses the fused log-coordinate gradient instead.
        """
        self._require_preprocessed()
        _, q, vectors = self._validate_q(Q)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if self.T == 0:
                task = vectors.T @ self._zero_task_gram @ vectors
                gradient = (np.diag(self.P / q) - task / (q[:, None] * q[None, :])) / self.N1
            else:
                k = self._kernel_eigenvalues[:, None]
                denominator = self.T + k * q[None, :]
                score = (self._rotated_labels @ vectors) / denominator
                gradient = (np.diag(np.sum(k / denominator, axis=0)) - score.T @ (k * score)) / self.N1
            gradient += np.diag(np.expm1(np.log(q) / self.L) / q)
            return vectors @ gradient @ vectors.T

    def effectiveAction(self, Q):
        """Differentiable torch action via small Cholesky blocks (diagnostic).

        Uses O(P*D^3) work and, with autograd, O(P*D^2) storage. The optimizer
        uses the faster fused NumPy path. No eigenvector autodiff is needed.
        """
        self._require_preprocessed()
        Q = torch.as_tensor(Q, dtype=torch.float64)
        if Q.ndim == 0:
            Q = Q * torch.eye(self.D, dtype=Q.dtype, device=Q.device)
        if (tuple(Q.shape) != (self.D, self.D) or not torch.all(torch.isfinite(Q))
                or not torch.allclose(Q, Q.T, rtol=1e-10, atol=1e-12)):
            raise ValueError("Q must be a finite symmetric matrix of output dimension")
        Q = (Q + Q.T) / 2
        eigenvalues = torch.linalg.eigvalsh(Q)
        if eigenvalues[0].item() <= 0:
            raise ValueError("Q must be strictly positive definite")
        logs = torch.log(eigenvalues)
        prior = self.L * torch.expm1(logs / self.L).sum() - logs.sum()
        if self.T == 0:
            factor = torch.linalg.cholesky(Q)
            task = torch.tensor(self._zero_task_gram, dtype=Q.dtype, device=Q.device)
            likelihood = (self.D * self._kernel_logdet + self.P * logs.sum()
                          + torch.trace(torch.cholesky_solve(task, factor)))
        else:
            likelihood = Q.new_zeros(())
            eye = torch.eye(self.D, dtype=Q.dtype, device=Q.device)
            for start in range(0, self.P, self.batch_size):
                block = slice(start, start + self.batch_size)
                k = torch.tensor(self._kernel_eigenvalues[block], dtype=Q.dtype, device=Q.device)
                y = torch.tensor(self._rotated_labels[block], dtype=Q.dtype, device=Q.device)
                factor = torch.linalg.cholesky(k[:, None, None] * Q + self.T * eye)
                solved = torch.cholesky_solve(y[:, :, None], factor)[:, :, 0]
                likelihood = likelihood + 2 * torch.log(factor.diagonal(dim1=-2, dim2=-1)).sum() + (y * solved).sum()
        return prior + likelihood / self.N1

    def optimize(self, Q0=1., maxiter=500, gtol=1e-6, n_restarts=0, random_state=0, *, verbose=False):
        """Minimize in symmetric log(Q) coordinates; inspect result/converged.

        Q0 is a positive scalar times identity or a physical SPD matrix.
        Defaults to one start at identity; additional starts are explicit. For scans use
        the previous optQ as Q0 and n_restarts=0. No global minimum guarantee.
        """
        self._require_preprocessed()
        self._reset_solution()
        if self._zero_boundary:
            raise ValueError("no finite SPD minimizer at T=0 with rank-deficient targets and P >= N1; use positive T")
        result, attempts = minimize_log_matrix(
            self._log_action_gradient, self._coordinates, self.L, Q0=Q0,
            maxiter=maxiter, gtol=gtol, n_restarts=n_restarts, random_state=random_state, verbose=verbose)
        self.result, self.optimization_results = result, attempts
        self.optQ, self.optR = result.Q.copy(), result.R.copy()
        self._solution_spectrum = (result.log_eigenvalues.copy(), result.eigenvectors.copy())
        self._solution_q_snapshot = self.optQ.copy()
        self.converged = bool(result.converged)
        self.solution_kind = "saddle"
        if not self.converged:
            warnings.warn(f"FC multi-output saddle did not converge: log-coordinate gradient norm {result.gradient_norm:.3g}; predictions are disabled", RuntimeWarning)
        return result

    def optimize_smart(self, **kwargs):
        return self.optimize(**kwargs)

    def setIW(self):
        self._require_preprocessed()
        self._reset_solution()
        self.optQ = self.optR = np.eye(self.D)
        self._solution_spectrum = (np.zeros(self.D), np.eye(self.D))
        self._solution_q_snapshot = self.optQ.copy()
        self.converged, self.solution_kind = True, "infinite_width"

    def _prediction_spectrum(self):
        self._require_solution()
        # Preserve the authoritative log spectrum on normal optimizer paths.
        # Also support deliberate manual replacement of optQ after setIW().
        if not np.array_equal(self.optQ, self._solution_q_snapshot):
            Q, q, vectors = self._validate_q(self.optQ)
            self.optQ = Q
            self._solution_spectrum = (np.log(q), vectors)
            self._solution_q_snapshot = Q.copy()
            self.optR = (vectors * q**(1. / self.L)) @ vectors.T
            self._invalidate_prediction()
        return self._solution_spectrum

    def effectiveKernel(self):
        """Explicit dense (K_Q, K_Q+T*I) inspection; allocates O(P^2*D^2).

        Optimization and prediction never call this method. Index a*P+mu
        corresponds to output a and training example mu.
        """
        self._prediction_spectrum()
        kernel = np.kron(self.optQ, self.finalKNNGP)
        covariance = kernel.copy()
        covariance.flat[::self.P * self.D + 1] += self.T
        return kernel, covariance

    def _test_kernels(self, X):
        cross = (X @ self.X.T) * self.corrNorm
        diagonal = np.einsum("ij,ij->i", X, X) * self.corrNorm
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for l, train_diagonal in enumerate(self._train_diagonals):
                cross = self._activation(diagonal[:, None], cross, train_diagonal[None, :]) / self.priors[l + 1]
                diagonal = self._activation(diagonal, diagonal, diagonal) / self.priors[l + 1]
            return cross / self.gamma**2, diagonal / self.gamma**2

    def predict(self, Xtest, batch_size=None, return_cov=False):
        """Mean and cached marginal latent variances, computed in batches.

        return_cov=True additionally returns each test example's D-by-D
        output covariance (not covariances between different test examples).
        Observation noise T is not added to any returned variance.
        """
        s, vectors = self._prediction_spectrum()
        self._invalidate_prediction()
        Xtest = self._inputs(Xtest)
        if Xtest.shape[1] != self.N0:
            raise ValueError("test inputs must match the training input dimension")
        bs = self.batch_size if batch_size is None else positive_int(batch_size, "batch_size")
        q = np.exp(s)
        if self.T == 0:
            weights = np.broadcast_to(1. / self._kernel_eigenvalues[:, None], (self.P, self.D))
        else:
            denominator = np.logaddexp(np.log(self.T), self._log_kernel_eigenvalues[:, None] + s)
            weights = np.exp(s - denominator)  # q/(T+k*q)
        mean_coefficients = weights * (self._rotated_labels @ vectors)
        means, variances, covariances = [], [], []
        for start in range(0, len(Xtest), bs):
            cross, prior = self._test_kernels(Xtest[start:start + bs])
            projection = cross @ self._kernel_vectors
            mean = (projection @ mean_coefficients) @ vectors.T
            modes = q * (prior[:, None] - (projection**2) @ weights)
            tolerance = 1e-9 * max(1., float(np.max(np.abs(prior[:, None] * q))))
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(modes)):
                raise FloatingPointError("non-finite posterior prediction; check kernel conditioning")
            if np.min(modes) < -tolerance:
                raise FloatingPointError("negative posterior variance beyond roundoff; check kernel conditioning")
            modes = np.maximum(modes, 0.)
            means.append(mean)
            variances.append(modes @ (vectors**2).T)
            if return_cov:
                covariances.append(np.einsum("aj,tj,bj->tab", vectors, modes, vectors))
        mean, variance = np.concatenate(means), np.concatenate(variances)
        self.Ptest = len(Xtest)
        self._prediction = (mean, variance)
        self.Ypred = mean[:, 0] if self._vector_labels else mean
        self.predictive_variance = variance[:, 0] if self._vector_labels else variance
        if return_cov:
            self.predictive_covariance = np.concatenate(covariances)
            return self.Ypred, self.predictive_covariance
        return self.Ypred

    def averageLoss(self, Ytest, per_output=False):
        """Expected squared error, squared bias, and latent variance.

        Defaults to three scalar means over examples and outputs. With
        per_output=True, returns three length-D arrays of example averages.
        """
        if self._prediction is None:
            raise RuntimeError("call predict(Xtest) before averageLoss(Ytest)")
        if not np.array_equal(self.optQ, self._solution_q_snapshot):
            self._invalidate_prediction()
            raise RuntimeError("optQ changed; call predict(Xtest) again")
        mean, variance = self._prediction
        bias = (self._targets(Ytest, len(mean)) - mean)**2
        if per_output:
            return (bias + variance).mean(axis=0), bias.mean(axis=0), variance.mean(axis=0)
        return float(np.mean(bias + variance)), float(np.mean(bias)), float(np.mean(variance))
