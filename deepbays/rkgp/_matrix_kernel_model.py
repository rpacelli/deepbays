"""Small dense EWA models sharing covariance and log-SPD optimization code.

These helpers are additive; previous scalar and spectral theory routines
retain their existing implementations and output-major conventions.
"""

import warnings
import numpy as np
from ..conv_geometry import positive_int
from ..kernels.conv_kernels import StackedCNNKernel, as_numpy, image_batch
from ..kernels.conv_diagonal import DiagonalCNNKernel
from ..kernels.multioutput import PatchCovariance, contract_blocks, contract_self
from ._matrix_order_parameter import (SymmetricCoordinates, matrix_prior,
                                      exp_divided_differences, minimize_log_matrix)


class CNNFeatures:
    def __init__(self, L, *, priors, act, gamma, mask, stride, padding,
                 pooling, kernel_backend, batch_size, max_kernel_bytes, kernel_cache=None):
        if pooling not in (None, 'avg'):
            raise ValueError("pooling must be None or 'avg'")
        if kernel_backend not in ('auto', 'reference', 'diagonal'):
            raise ValueError("kernel_backend must be 'auto', 'reference', or 'diagonal'")
        self.L, self.pooling, self.kernel_backend = L, pooling, kernel_backend
        self.kernel_cache = kernel_cache
        self.kwargs = dict(priors=priors, act=act, gamma=gamma, mask=mask,
                           stride=stride, padding=padding, batch_size=batch_size,
                           max_kernel_bytes=max_kernel_bytes)

    def inputs(self, X):
        return image_batch(X)

    def _pool(self, H):
        return H.mean(axis=(-2, -1), keepdims=True) if self.pooling == 'avg' else H

    def prepare(self, X):
        self.builder = StackedCNNKernel(X.shape[-2:], X.shape[1], self.L, **self.kwargs)
        if self.kernel_backend == 'diagonal' or (self.kernel_backend == 'auto' and self.builder.d == 1):
            self.builder = DiagonalCNNKernel(X.shape[-2:], X.shape[1], self.L, **self.kwargs)
        self.patch_shapes = tuple(layer.output_shape for layer in self.builder.layers)
        self.d = 1 if self.pooling == 'avg' else self.builder.d
        self.train = self.builder.prepare(X)
        return self._pool(self._blocks(self.train))

    def _blocks(self, left, right=None, *, diagonal=False):
        if self.kernel_cache is not None:
            return self.kernel_cache.blocks(self.builder, left, right, diagonal=diagonal)
        return self.builder.self_blocks(left) if diagonal else self.builder.cross(left, right)

    def test(self, X):
        prepared = self.builder.prepare(X)
        return self._pool(self._blocks(prepared, self.train)), self._pool(self._blocks(prepared, diagonal=True))


class FCFeatures:
    def __init__(self, L, N1, c, *, priors, act, gamma, batch_size):
        # Reuse the established FC kernel/diagonal cache. Its Gaussian target
        # cache is unused; this object only provides prior kernels.
        from .theory_deep_FC_multioutput import FC_deep_multioutput
        self.provider = FC_deep_multioutput(L, N1, c, 1., priors, act, gamma, batch_size)
        self.d = 1

    def inputs(self, X):
        return self.provider._inputs(X)

    def prepare(self, X):
        self.provider.preprocess(X, np.zeros((len(X), self.provider.D)))
        return self.provider.finalKNNGP[:, :, None, None]

    def covariance(self, Q):
        # preprocess has already checked/corrected the scalar kernel spectrum.
        from ._laplace_covariance import SeparableCovariance
        return SeparableCovariance(self.provider.finalKNNGP, Q)

    def test(self, X):
        if X.shape[1] != self.provider.N0:
            raise ValueError('test input dimension must match training inputs')
        cross, diagonal = self.provider._test_kernels(X)
        return cross[:, :, None, None], diagonal[:, None, None]


class MatrixKernelModel:
    def _initialize(self, L, width, D, c, features, batch_size, max_dense_size):
        self.L = positive_int(L, 'L')
        self.N1 = self.Nc = positive_int(width, 'width')
        self.D, self.c = positive_int(D, 'D'), positive_int(c, 'latent outputs')
        self.features = features
        self.batch_size = positive_int(batch_size, 'batch_size')
        self.max_dense_size = positive_int(max_dense_size, 'max_dense_size')
        self._ready = False
        self._reset_solution()

    def _reset_solution(self, keep_evidence=False):
        self.optQ = self.optR = self.result = self.solution_kind = None
        self.converged = False
        self.optimization_results = []
        self._cached_Q = self._posterior = None
        if not keep_evidence:
            self._warm_alpha = self._last_evidence = None
        self._cached_evidence_signature = None
        self._prediction = None

    def preprocess(self, X, Y):
        self._ready = False
        self._reset_solution()
        X = self.features.inputs(X).copy()
        if len(X) * self.c > self.max_dense_size:
            raise ValueError('dense covariance size limit exceeded; reduce P or explicitly increase max_dense_size')
        labels = self._targets(Y, len(X))
        H = self.features.prepare(X)
        self.operator = PatchCovariance(H, self.c)
        self.d = self.operator.d
        if self.N1 < self.operator.dimension:
            raise ValueError(f'the full-rank Wishart representation requires width >= d*c={self.operator.dimension}')
        self._coordinates = SymmetricCoordinates(self.operator.dimension)
        self.X, self.Y, self.P = X, labels, len(X)
        for array in (self.X, self.Y, self.operator.H):
            array.setflags(write=False)
        self._ready = True
        return self

    def _require_ready(self):
        if not self._ready:
            raise RuntimeError('call preprocess(X, Y) first')

    def _log_action_gradient(self, coordinates):
        self._require_ready()
        s, U = np.linalg.eigh(self._coordinates.unpack(coordinates))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            Q = (U * np.exp(s)) @ U.T
            nll, gradient, _ = self._evidence(Q)
            prior, prior_gradient = matrix_prior(s, self.L)
            dQ = (2. / self.N1) * self.operator.adjoint(gradient)
            dH = U @ (exp_divided_differences(s) * (U.T @ dQ @ U)
                      + np.diag(prior_gradient)) @ U.T
            value = prior + 2 * nll / self.N1
        if not np.isfinite(value) or not np.all(np.isfinite(dH)):
            raise FloatingPointError('non-finite action or log-Q gradient')
        return float(value), self._coordinates.pack((dH + dH.T) / 2)

    def effectiveAction(self, Q):
        """Scalar action, with evidence constants independent of Q omitted."""
        self._require_ready()
        Q, ev, _ = self._coordinates.validate_q(Q)
        return matrix_prior(np.log(ev), self.L)[0] + 2 * self._evidence(Q)[0] / self.N1

    def computeActionGrad(self, Q):
        self._require_ready()
        Q, ev, U = self._coordinates.validate_q(Q)
        gradient = 2 * self.operator.adjoint(self._evidence(Q)[1]) / self.N1
        return (gradient + gradient.T) / 2 + (U * (ev**(1 / self.L - 1) - 1 / ev)) @ U.T

    def optimize(self, Q0=1., maxiter=300, gtol=1e-6, n_restarts=0, random_state=0, *, verbose=False):
        self._require_ready()
        self._reset_solution(keep_evidence=True)
        self._optimizing, self._optimization_verbose = True, bool(verbose)
        try:
            result, attempts = minimize_log_matrix(self._log_action_gradient, self._coordinates,
                                                   self.L, Q0, maxiter, gtol, n_restarts, random_state,
                                                   verbose=verbose)
        finally:
            self._optimizing = self._optimization_verbose = False
        self.result, self.optimization_results = result, attempts
        self.optQ, self.optR = result.Q.copy(), result.R.copy()
        self.converged, self.solution_kind = bool(result.converged), 'saddle'
        if not self.converged:
            warnings.warn(f'matrix saddle did not converge: gradient norm {result.gradient_norm:.3g}; predictions disabled', RuntimeWarning)
        return result

    def setQ(self, Q):
        """Explicitly select a fixed physical covariance for diagnostics."""
        self._require_ready()
        Q, ev, U = self._coordinates.validate_q(Q)
        self._reset_solution(keep_evidence=True)
        self.optQ = Q.copy()
        self.optR = (U * ev**(1 / self.L)) @ U.T
        self.converged, self.solution_kind = True, 'fixed_Q'
        return self

    def setIW(self):
        self.setQ(1.)
        self.solution_kind = 'infinite_width'
        return self

    def _solution_posterior(self):
        self._require_ready()
        if not self.converged or self.optQ is None:
            raise RuntimeError('optimize successfully, or explicitly select setIW()/setQ(), before prediction')
        Q, _, _ = self._coordinates.validate_q(self.optQ)
        signature = self._evidence_signature()
        if (self._cached_Q is None or not np.array_equal(Q, self._cached_Q)
                or signature != self._cached_evidence_signature):
            _, _, self._posterior = self._evidence(Q)
            self._cached_Q = Q.copy()
            self._cached_evidence_signature = signature
        return Q, self._posterior

    def effectiveKernel(self):
        """Dense latent covariance; index mu*c+a denotes example mu, output a."""
        Q, _ = self._solution_posterior()
        return self.operator.dense(Q)

    def _predict_gaussian(self, Xtest, alpha, reduction, Q, batch_size=None):
        Xtest = self.features.inputs(Xtest)
        bs = self.batch_size if batch_size is None else positive_int(batch_size, 'batch_size')
        means, covariances = [], []
        for start in range(0, len(Xtest), bs):
            X = Xtest[start:start + bs]
            H, self_H = self.features.test(X)
            cross = contract_blocks(H, Q, self.c, self.operator.normalization)
            prior = contract_self(self_H, Q, self.c, self.operator.normalization)
            cross = cross.reshape(len(X), self.c, -1)
            mean = np.einsum('man,n->ma', cross, alpha)
            cov = prior - np.einsum('man,nk,mbk->mab', cross, reduction, cross, optimize=True)
            cov = (cov + cov.transpose(0, 2, 1)) / 2
            ev, U = np.linalg.eigh(cov)
            if ev.min() < -1e-9 * max(1., float(np.max(np.abs(prior)))):
                raise FloatingPointError('negative predictive covariance beyond roundoff')
            cov = (U * np.maximum(ev, 0.)[:, None, :]) @ U.transpose(0, 2, 1)
            means.append(mean)
            covariances.append(cov)
        return np.concatenate(means), np.concatenate(covariances)
