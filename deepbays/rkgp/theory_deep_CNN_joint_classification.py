"""Multi-class crossentropy with a joint CNN rate and function-space Laplace."""

import numpy as np
from ..conv_geometry import positive_int
from ..kernels.conv_kernels import as_numpy
from ._cnn_joint_model import JointCNNModel
from ._softmax_laplace import (contrast_basis, class_labels, fit_laplace,
    gaussian_softmax_probabilities, gaussian_softmax_statistics)


class CNN_deep_joint_classifier(JointCNNModel):
    """D-class joint Laplace-EWA in D-1 orthonormal contrast latents.

    Logits are (g/sqrt(2), -g/sqrt(2)), as in CNN_deep_classifier(D=2).
    beta multiplies summed CE. This is not a single-logit Bernoulli convention.
    The spatial prior is exact for linear networks; erf uses the explicitly
    experimental fixed-IW path-gain closure. See CNN_deep_joint for geometry,
    solver, and memory options. The CE evidence remains a Laplace approximation.
    rank_policy='leading_rate' permits the deterministic covariance-tilt saddle
    below the empirical innovation rank threshold; it does not integrate the
    finite-width innovation posterior. The default is 'full_rank'.
    """

    def __init__(self, L, Nc, D=2, beta=1., priors=(1., 1.), act='erf', mask=3,
                 stride=1, padding='valid', gamma=1., batch_size=16,
                 max_kernel_bytes=128*1024**2, *, pooling=None, closure='auto',
                 parameterization='innovation', kernel_backend='auto',
                 max_dense_size=2000, max_joint_coordinates=20000,
                 mode_tol=1e-10, mode_maxiter=100, rank_policy='full_rank'):
        self.D = positive_int(D, 'D')
        if self.D < 2:
            raise ValueError("classification requires D >= 2")
        self.c = self.D-1
        self.basis = contrast_basis(self.D)
        self.basis.setflags(write=False)
        self.beta, self.mode_tol = float(beta), float(mode_tol)
        self.mode_maxiter = positive_int(mode_maxiter, 'mode_maxiter')
        self._evidence_signature()
        self._initialize(L, Nc, priors=priors, act=act, mask=mask, stride=stride,
            padding=padding, gamma=gamma, pooling=pooling, closure=closure,
            parameterization=parameterization, batch_size=batch_size,
            max_kernel_bytes=max_kernel_bytes, kernel_backend=kernel_backend,
            max_dense_size=max_dense_size, max_joint_coordinates=max_joint_coordinates,
            rank_policy=rank_policy)

    def _targets(self, Y, count):
        return class_labels(as_numpy(Y), count, self.D)

    def _evidence_signature(self):
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not np.isfinite(self.mode_tol) or self.mode_tol <= 0:
            raise ValueError("mode_tol must be finite and positive")
        positive_int(self.mode_maxiter, 'mode_maxiter')
        return (self.beta, self.mode_tol, self.mode_maxiter, tuple(self.basis.ravel()))

    def _evidence(self, K):
        self._evidence_signature()
        state = fit_laplace(K, self.Y, self.basis, beta=self.beta, mode_tol=self.mode_tol,
            maxiter=self.mode_maxiter, alpha0=self._warm_alpha, max_dense_size=self.max_dense_size)
        self._warm_alpha = state.alpha.copy()
        return state.nll, state.gradient, state

    @property
    def laplace_state(self):
        return self._solution_posterior()

    def predict_latent(self, Xtest, batch_size=None, *, logits=False):
        mean, covariance = self._predict_gaussian(Xtest, batch_size)
        if logits:
            return mean @ self.basis.T, np.einsum('ai,mij,bj->mab', self.basis, covariance, self.basis)
        return mean, covariance

    def predict_proba(self, Xtest, *, samples=4096, seed=0, batch_size=None):
        mean, covariance = self.predict_latent(Xtest, batch_size)
        return gaussian_softmax_probabilities(mean, covariance, self.basis, samples, seed)

    def predict(self, Xtest, **kwargs):
        return self.predict_proba(Xtest, **kwargs).argmax(axis=1)

    def predict_statistics(self, Xtest, *, samples=4096, seed=0, batch_size=None):
        mean, covariance = self.predict_latent(Xtest, batch_size)
        result = gaussian_softmax_statistics(mean, covariance, self.basis, samples, seed)
        result.update(latent_mean=mean, latent_covariance=covariance)
        return result

    def metrics(self, Xtest, Ytest, **kwargs):
        """Accuracy, predictive NLL and Brier score of mean probabilities."""
        p = self.predict_proba(Xtest, **kwargs)
        y = self._targets(Ytest, len(p))
        return dict(accuracy=float(np.mean(p.argmax(1) == y)),
                    predictive_nll=float(-np.log(np.maximum(
                        p[np.arange(len(y)), y], np.finfo(float).tiny)).mean()),
                    brier=float(np.sum((p-np.eye(self.D)[y])**2, axis=1).mean()))
