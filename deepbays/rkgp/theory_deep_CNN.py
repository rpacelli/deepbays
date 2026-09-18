"""Equal-channel, single-output CNN theory under the stacked Wishart ansatz.

The optimized order parameter optQ is the *product covariance*, a d x d SPD
matrix, where d is the final number of patches. It is not the vector of L
scalar layer parameters used by FC_deep_vanilla. At channel-count speed Nc,

    2 I(Q) = L tr(Q**(1/L)) - log det Q - L d.

We cache all activation kernels before optimization, use Cholesky solves for
the data covariance, and optimize a symmetric S = log(Q). Its matrix
exponential has an analytic Frechet derivative, including repeated
eigenvalues, so no large autograd graph or matrix-root differentiation is
needed by the optimizer. All theory calculations use float64 on the CPU.
"""

import warnings
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from ..conv_geometry import layer_precisions, positive_int
from ._matrix_order_parameter import (SymmetricCoordinates, matrix_prior,
                                      exp_divided_differences, minimize_log_matrix)
from ..kernels.conv_kernels import StackedCNNKernel, as_numpy, image_batch


class CNN_deep:
    """MLP-style preprocess/optimize/predict interface for 2D CNNs.

    Parameters
    ----------
    L : int
        Number of convolutional hidden layers, each followed by activation.
    Nc : int
        Common channel count of all hidden layers; final readout is scalar.
    T : float
        Observation noise variance / sampling temperature, >= 0. At T=0 the
        training kernel must be strictly positive definite; no jitter is added.
    priors : sequence
        L convolution weight precisions followed by the readout precision.
        Two entries abbreviate [first, remaining, ...], as in the MLP theory.
    mask, stride, padding
        Scalars/tuples are shared across layers; lists have one entry per
        layer. A tuple means a spatial (height, width) pair. Padding is zero
        padding: 'valid', 'same', or symmetric integer/spatial-pair padding.
    gamma : float
        Output normalization, matching ConvNet (1 for standard scaling).

    Biases and pooling are not part of this theory. The Wishart rate assumes
    fixed L and d as P,Nc grow. Square/quadratic and ReLU use the central EWA
    just as the vanilla MLP; accuracy for those nonlinearities is an ansatz.
    """

    def __init__(self, L, Nc, T, priors=(1., 1.), act="erf", mask=3,
                 stride=1, padding="valid", gamma=1., batch_size=32,
                 max_kernel_bytes=64 * 1024**2):
        self.L, self.Nc = positive_int(L, "L"), positive_int(Nc, "Nc")
        self.N1 = self.Nc
        if not np.isfinite(T) or T < 0:
            raise ValueError("T must be finite and nonnegative")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite")
        self.T, self.gamma = float(T), float(gamma)
        self.priors = layer_precisions(priors, self.L)
        self.act, self.mask, self.stride, self.padding = act, mask, stride, padding
        self.batch_size, self.max_kernel_bytes = batch_size, max_kernel_bytes
        self.converged = False
        self.optQ = None
        self.optR = None
        self.result = None
        self.optimization_results = []
        self.solution_kind = None
        self._features = None
        self._invalidate_prediction()

    def _invalidate_prediction(self):
        self._factor = None
        self._prediction = None

    def _require_preprocessed(self):
        if self._features is None:
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

    def preprocess(self, X, Y):
        """Cache final patch blocks and symmetric kernel coefficients once."""
        # A failed rebuild must not leave an old solution usable with new data.
        self._features, self.optQ, self.converged = None, None, False
        self.optR, self.result, self.solution_kind = None, None, None
        self.optimization_results = []
        self._invalidate_prediction()
        X = image_batch(X).copy()
        labels = as_numpy(Y)
        if labels.shape not in ((len(X),), (len(X), 1)) or not np.all(np.isfinite(labels)):
            raise ValueError("Y must be finite scalar labels of shape (P,) or (P, 1)")
        self.X, self.Y = X, labels.copy()
        self.y = labels.reshape(-1).copy()
        self.P = len(X)
        self.N0 = int(np.prod(X.shape[1:]))
        self.builder = StackedCNNKernel(X.shape[-2:], X.shape[1], self.L,
                                       mask=self.mask, stride=self.stride, padding=self.padding,
                                       priors=self.priors, act=self.act, gamma=self.gamma,
                                       batch_size=self.batch_size, max_kernel_bytes=self.max_kernel_bytes)
        if self.Nc < self.builder.d:
            raise ValueError(f"the SPD Wishart representation requires Nc >= final patches d={self.builder.d}; increase Nc or reduce the final spatial size")
        self._train = self.builder.prepare(X)
        blocks = self.builder.cross(self._train)
        self.d = self.builder.d
        self.patch_shapes = tuple(layer.output_shape for layer in self.builder.layers)
        self.H = blocks.transpose(2, 3, 0, 1)  # H[i,j,mu,nu], includes readout scaling
        self.H.setflags(write=False)
        self._coordinates = SymmetricCoordinates(self.d)
        self._indices = self._coordinates.indices
        self._coordinate_scale = self._coordinates.scale
        i, j = self._indices
        coefficients = (blocks[:, :, i, j] + blocks[:, :, j, i]) / (2 * self.d)
        coefficients *= self._coordinate_scale
        self._features = np.ascontiguousarray(coefficients.reshape(self.P * self.P, -1))
        self._torch_features = torch.from_numpy(self._features)
        self._torch_y = torch.from_numpy(self.y)
        self.finalKNNGP = self._kernel(np.eye(self.d))
        return self

    def _kernel(self, Q):
        kernel = (self._features @ self._pack(Q)).reshape(self.P, self.P)
        return (kernel + kernel.T) / 2

    def _likelihood(self, Q):
        sigma = self._kernel(Q)
        sigma.flat[::self.P + 1] += self.T
        if not np.all(np.isfinite(sigma)):
            raise FloatingPointError("non-finite data covariance")
        factor = cho_factor(sigma, lower=True, check_finite=False)
        v = cho_solve(factor, self.y, check_finite=False)
        value = (2 * np.log(np.diag(factor[0])).sum() + self.y @ v) / self.Nc
        score = cho_solve(factor, np.eye(self.P), check_finite=False) - np.outer(v, v)
        gradient = self._unpack(self._features.T @ score.reshape(-1)) / self.Nc
        return value, gradient

    def effectiveAction(self, Q):
        """Differentiable torch action of the physical product matrix Q.

        Uses logdet(K+T I), differing from logdet(I+K/T) only by a Q-independent
        constant for T>0. This also agrees with FC_deep_vanilla's convention.
        The additive prior constant is chosen so 2 I(I)=0.
        """
        self._require_preprocessed()
        Q = torch.as_tensor(Q, dtype=torch.float64)
        if Q.ndim == 0:
            Q = Q * torch.eye(self.d, dtype=Q.dtype, device=Q.device)
        if tuple(Q.shape) != (self.d, self.d) or not torch.all(torch.isfinite(Q)) or not torch.allclose(Q, Q.T):
            raise ValueError("Q must be a finite symmetric matrix of final patch dimension")
        Q = (Q + Q.T) / 2
        eigenvalues = torch.linalg.eigvalsh(Q)
        if eigenvalues[0].item() <= 0:
            raise ValueError("Q must be strictly positive definite")
        log_eigenvalues = torch.log(eigenvalues)
        prior = self.L * torch.expm1(log_eigenvalues / self.L).sum() - log_eigenvalues.sum()
        coordinates = Q[self._indices] * torch.as_tensor(self._coordinate_scale, device=Q.device)
        kernel = (self._torch_features.to(Q.device) @ coordinates).reshape(self.P, self.P)
        sigma = (kernel + kernel.T) / 2 + self.T * torch.eye(self.P, dtype=Q.dtype, device=Q.device)
        factor = torch.linalg.cholesky(sigma)
        y = self._torch_y.to(Q.device)
        solve = torch.cholesky_solve(y[:, None], factor)[:, 0]
        return prior + (2 * torch.log(torch.diagonal(factor)).sum() + y @ solve) / self.Nc

    def computeActionGrad(self, Q):
        """Analytic Frobenius gradient with respect to symmetric physical Q."""
        self._require_preprocessed()
        Q, eigenvalues, vectors = self._validate_q(Q)
        _, gradient = self._likelihood(Q)
        return gradient + (vectors * (eigenvalues**(1. / self.L - 1) - 1. / eigenvalues)) @ vectors.T

    def _log_action_gradient(self, coordinates):
        S = self._unpack(coordinates)
        eigenvalues, vectors = np.linalg.eigh(S)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            exp_s = np.exp(eigenvalues)
            Q = (vectors * exp_s) @ vectors.T
            value, gradient_q = self._likelihood(Q)
            prior, prior_gradient = matrix_prior(eigenvalues, self.L)
            value += prior
            frechet = exp_divided_differences(eigenvalues)
            gradient = vectors @ (frechet * (vectors.T @ gradient_q @ vectors)
                                  + np.diag(prior_gradient)) @ vectors.T
        packed = self._pack((gradient + gradient.T) / 2)
        if not np.isfinite(value) or not np.all(np.isfinite(packed)):
            raise FloatingPointError("non-finite action or gradient")
        return float(value), packed

    def optimize(self, Q0=1., maxiter=500, gtol=1e-6, n_restarts=0, random_state=0, *, verbose=False):
        """Minimize in symmetric log(Q) coordinates using analytic gradients.

        One start at Q0 by default; n_restarts requests additional starts.
        Use n_restarts=0 for warm starts along a channel-count/temperature scan.
        L-BFGS with line search is followed, if necessary, by a BFGS refinement
        for <=512 variables. `converged` tests the actual log-coordinate
        gradient, not just the optimizer's small-step/function-change flag.
        `optimization_results` contains all attempts; `result` is the selected
        scipy OptimizeResult, whose x is the packed symmetric log(Q).

        The action is not globally convex; multiple starts are a check, not a
        proof that the global minimum has been found.
        """
        self._require_preprocessed()
        self.converged = False
        self.result, self.solution_kind = None, None
        self.optimization_results = []
        self._invalidate_prediction()
        result, results = minimize_log_matrix(
            self._log_action_gradient, self._coordinates, self.L, Q0=Q0,
            maxiter=maxiter, gtol=gtol, n_restarts=n_restarts, random_state=random_state, verbose=verbose)
        self.optimization_results, self.result = results, result
        self.optQ, self.optR = result.Q, result.R
        self.converged = bool(result.converged)
        self.solution_kind = "saddle"
        if not self.converged:
            warnings.warn(f"CNN saddle did not converge: log-coordinate gradient norm {result.gradient_norm:.3g}; predictions are disabled", RuntimeWarning)
        return result

    def optimize_smart(self, **kwargs):
        """Alias for optimize, which already includes restarts/refinement."""
        return self.optimize(**kwargs)

    def setIW(self):
        self._require_preprocessed()
        self.optQ = np.eye(self.d)
        self.optR = np.eye(self.d)
        self.converged = True
        self.solution_kind = "infinite_width"
        self.result = None
        self.optimization_results = []
        self._invalidate_prediction()

    def effectiveKernel(self):
        self._require_solution()
        kernel = self._kernel(self.optQ)
        return kernel, kernel + self.T * np.eye(self.P)

    def predict(self, Xtest, batch_size=None):
        """Posterior mean; cache latent variances for averageLoss.

        Test-train blocks are contracted and discarded batch by batch. Only
        within-example patch blocks are needed for the test prior variance.
        Output shape matches the shape convention of the training labels.
        """
        self._require_solution()
        Xtest = image_batch(Xtest)
        prepared = self.builder.prepare(Xtest)
        bs = self.builder.batch_size if batch_size is None else positive_int(batch_size, "batch_size")
        _, sigma = self.effectiveKernel()
        factor = cho_factor(sigma, lower=True, check_finite=False)
        self._factor = factor
        self.Ptest = len(Xtest)
        means, variances, prior_variances = [], [], []
        cross_kernels, products = [], []
        for start in range(0, len(Xtest), bs):
            subset = self.builder.subset(prepared, slice(start, start + bs))
            blocks = self.builder.cross(subset, self._train)
            cross = np.einsum("abij,ij->ab", blocks, self.optQ, optimize=True) / self.d
            self_blocks = self.builder.self_blocks(subset)
            prior = np.einsum("aij,ij->a", self_blocks, self.optQ, optimize=True) / self.d
            product = cho_solve(factor, cross.T, check_finite=False).T
            variance = prior - np.sum(product * cross, axis=1)
            tolerance = 1e-9 * max(1., float(np.max(np.abs(prior))))
            if np.min(variance) < -tolerance:
                raise FloatingPointError("negative posterior variance beyond roundoff; check kernel conditioning")
            means.append(product @ self.y)
            variances.append(np.maximum(variance, 0.))
            prior_variances.append(prior)
            cross_kernels.append(cross)
            products.append(product)
        mean, variance = np.concatenate(means), np.concatenate(variances)
        self.Ypred = mean[:, None] if self.Y.ndim == 2 else mean
        self.predictive_variance = variance
        self.rK0L = np.concatenate(prior_variances)
        self.rK0XL = np.concatenate(cross_kernels)
        self.K0_invK = np.concatenate(products)
        self._prediction = (mean, variance)
        return self.Ypred

    def averageLoss(self, Ytest):
        """Expected squared error, squared bias, and latent posterior variance.

        As in the MLP implementation, observation noise T is not added to the
        latent function variance. Column labels are flattened to avoid silent
        (Ptest, Ptest) broadcasting.
        """
        if self._prediction is None:
            raise RuntimeError("call predict(Xtest) before averageLoss(Ytest)")
        mean, variance = self._prediction
        labels = as_numpy(Ytest)
        if labels.shape not in ((len(mean),), (len(mean), 1)) or not np.all(np.isfinite(labels)):
            raise ValueError("test labels must have shape (Ptest,) or (Ptest, 1)")
        bias = (labels.reshape(-1) - mean)**2
        return float(np.mean(bias + variance)), float(np.mean(bias)), float(np.mean(variance))


# Match the existing package's CONV architecture naming as well.
CONV_deep = CNN_deep
