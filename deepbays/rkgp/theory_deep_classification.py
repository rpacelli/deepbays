"""Multi-class FC/CNN EWA with deterministic function-space Laplace inference.

For C=D classes, c=C-1 orthonormal contrasts remove the unobserved common
logit. Q has dimension d*c (or c for FC and globally pooled CNNs). The
Gibbs likelihood is exp(-beta * summed_cross_entropy), and the action is
2I(Q) - (2/N) log Z_Laplace(Q). The rate function is the central EWA rate;
Laplace introduces a separate, explicitly approximate likelihood integral.
"""

from ._cnn_spatial_rate import CNNSpatialRate
import numpy as np
from ..kernels.conv_kernels import as_numpy
from ._matrix_kernel_model import MatrixKernelModel, FCFeatures, CNNFeatures
from ._softmax_laplace import (contrast_basis, class_labels, fit_laplace,
                               gaussian_softmax_probabilities, gaussian_softmax_statistics)


class SoftmaxMatrixModel(MatrixKernelModel):
    def _classification_settings(self, D, beta, mode_tol, mode_maxiter, verbose=False):
        self.verbose = bool(verbose)
        self.basis = contrast_basis(D)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if not np.isfinite(mode_tol) or mode_tol <= 0:
            raise ValueError('mode_tol must be finite and positive')
        from ..conv_geometry import positive_int
        self.beta, self.mode_tol = float(beta), float(mode_tol)
        self.mode_maxiter = positive_int(mode_maxiter, 'mode_maxiter')

    def _targets(self, Y, count):
        return class_labels(as_numpy(Y), count, self.D)

    def _evidence(self, Q):
        Q = np.asarray(Q, dtype=float)
        Q = (Q+Q.T)/2
        signature = self._evidence_signature()
        cached = self._last_evidence
        if cached is not None and cached[1] == signature and np.array_equal(cached[0], Q):
            state = cached[2]
            return state.nll, state.gradient, state
        # Retain only a small mode warm start while replacing the dense state.
        self._last_evidence = self._posterior = self._cached_Q = None
        cached = None
        covariance = (self.features.covariance(Q) if isinstance(self.features, FCFeatures)
                      else self.operator.dense(Q))
        optimizing = getattr(self, '_optimizing', False)
        state = fit_laplace(covariance, self.Y, self.basis, self.beta,
                            mode_tol=self.mode_tol, maxiter=self.mode_maxiter,
                            alpha0=self._warm_alpha, max_dense_size=self.max_dense_size,
                            verbose=self.verbose and not optimizing)
        self._warm_alpha = state.alpha.copy()
        self._last_evidence = (np.array(Q, copy=True), signature, state)
        if getattr(self, '_optimization_verbose', False):
            print(f'    Laplace: Newton steps={state.iterations}, residual={state.mode_residual:.3g}, time={state.seconds:.1f}s', flush=True)
        return state.nll, state.gradient, state

    def _evidence_signature(self):
        return (self.beta, self.mode_tol, self.mode_maxiter, tuple(self.basis.ravel()))

    @property
    def laplace_state(self):
        """Fitted mode, covariance, evidence, and residual at the selected Q."""
        return self._solution_posterior()[1]

    def predict_latent(self, Xtest, batch_size=None, *, logits=False):
        """Return per-example Gaussian mean/covariance in contrast coordinates.

        logits=True lifts them to the zero-sum C-logit representation. It
        does not restore the irrelevant common-logit prior variance.
        """
        Q, posterior = self._solution_posterior()
        mean, cov = self._predict_gaussian(Xtest, posterior.alpha, posterior.reduction, Q, batch_size)
        if logits:
            return mean @ self.basis.T, np.einsum('ai,mij,bj->mab', self.basis, cov, self.basis)
        return mean, cov

    def predict_proba(self, Xtest, *, samples=4096, seed=0, batch_size=None):
        """Posterior mean softmax probabilities, using scrambled Sobol quadrature."""
        mean, cov = self.predict_latent(Xtest, batch_size)
        return gaussian_softmax_probabilities(mean, cov, self.basis, samples, seed)

    def predict(self, Xtest, **kwargs):
        """Class indices from the posterior mean probabilities."""
        return self.predict_proba(Xtest, **kwargs).argmax(axis=1)

    def predict_statistics(self, Xtest, *, samples=4096, seed=0, batch_size=None):
        """Posterior softmax moments, argmax probabilities and Gaussian marginals.

        All arrays have one row per test example. The mean probabilities and
        their variances have D columns; argmax_probabilities gives the chance
        each class wins in a posterior draw (not a categorical-label draw).
        latent_mean/covariance use the D-1 contrast coordinates. These are
        conditional Laplace predictions at the selected EWA/IW Q.
        """
        mean, cov = self.predict_latent(Xtest, batch_size)
        result = gaussian_softmax_statistics(mean, cov, self.basis, samples, seed)
        result.update(latent_mean=mean, latent_covariance=cov)
        return result

    def metrics(self, Xtest, Ytest, **kwargs):
        """Accuracy, predictive NLL, and Brier score of mean probabilities."""
        p = self.predict_proba(Xtest, **kwargs)
        y = self._targets(Ytest, len(p))
        return dict(accuracy=float(np.mean(p.argmax(1) == y)),
                    predictive_nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny)).mean()),
                    brier=float(np.sum((p - np.eye(self.D)[y])**2, axis=1).mean()))


class FC_deep_classifier(SoftmaxMatrixModel):
    """Equal-width MLP classification; D classes, Q of dimension D-1.

    beta=1 is the ordinary categorical likelihood. beta=1/T reproduces the
    Gibbs convention with summed CE; beta=0 selects the no-data limit.
    No observation-noise diagonal is added. The existing FC regression
    class and its fast spectral optimizer are unaffected.
    """

    def __init__(self, L, N1, D, beta=1., priors=(1., 1.), act='erf', gamma=1.,
                 batch_size=32, *, mode_tol=1e-10, mode_maxiter=100, max_dense_size=2000, verbose=False):
        self._classification_settings(D, beta, mode_tol, mode_maxiter, verbose)
        features = FCFeatures(L, N1, D - 1, priors=priors, act=act, gamma=gamma, batch_size=batch_size)
        self.gamma, self.act = gamma, act
        self._initialize(L, N1, D, D - 1, features, batch_size, max_dense_size)


class CNN_deep_classifier(CNNSpatialRate, SoftmaxMatrixModel):
    """Equal-channel CNN classification, with flattening or global-average readout.

    Width must be at least d*(D-1), with fixed d,D in the EWA limit. For
    pooling='avg', d=1 in Q even when the pre-pooling grid has several patches.
    All inference currently uses the dense deterministic reference backend.
    Optional kernel_cache is a deepbays.kernels.cnn_cache.CNNKernelCache shared
    across widths; it caches prior patch blocks, never fitted Q or posteriors.
    rate_correction=True opts into a scalar spatial correction for one final
    pre-pooling patch. correction_weighting='label_free' is the default;
    'iw_dual' uses fixed IW Laplace-mode duals in the D-1 contrast space.
    The Laplace likelihood approximation itself is unchanged.
    """

    def __init__(self, L, Nc, D, beta=1., priors=(1., 1.), act='erf', mask=3,
                 stride=1, padding='valid', gamma=1., batch_size=32,
                 max_kernel_bytes=64 * 1024**2, *, pooling=None,
                 kernel_backend='auto', mode_tol=1e-10, mode_maxiter=100,
                 max_dense_size=2000, kernel_cache=None, verbose=False,
                 rate_correction=False, correction_weighting='label_free'):
        self._classification_settings(D, beta, mode_tol, mode_maxiter, verbose)
        features = CNNFeatures(L, priors=priors, act=act, gamma=gamma, mask=mask,
                               stride=stride, padding=padding, pooling=pooling,
                               kernel_backend=kernel_backend, batch_size=batch_size,
                               max_kernel_bytes=max_kernel_bytes, kernel_cache=kernel_cache)
        self.gamma, self.act, self.pooling = gamma, act, pooling
        self._initialize(L, Nc, D, D - 1, features, batch_size, max_dense_size)
        self._init_rate_correction(rate_correction, correction_weighting)
