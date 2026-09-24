"""Scalar Gaussian-output CNNs with a full joint spatial-rate saddle."""

from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ..kernels.conv_kernels import as_numpy
from ._cnn_joint_model import JointCNNModel


class CNN_deep_joint(JointCNNModel):
    """Joint-rate reference theory, with independent Gram/innovation solvers.

    closure='auto' uses the supported exact spatial rate for act='id', and
    the experimental fixed-IW path-gain EWA closure for act='erf'. Only these
    activations and one output are currently supported. For either closure,
    parameterization='gram' and 'innovation' minimize the same action.

    Nc is a positive integer or a length-L sequence of channel widths.
    max_kernel_bytes caps a dense training feature cache and separately
    targets block workspace; it is not a process memory limit. Auto selects
    streaming when that cache would exceed the budget. The P-by-P evidence
    matrices are controlled by max_dense_size.
    """

    def __init__(self, L, Nc, T, priors=(1., 1.), act='erf', mask=3,
                 stride=1, padding='valid', gamma=1., batch_size=16,
                 max_kernel_bytes=128*1024**2, *, pooling=None, closure='auto',
                 parameterization='innovation', kernel_backend='auto',
                 max_dense_size=2000, max_joint_coordinates=20000,
                 rank_policy='full_rank'):
        self.D, self.c = 1, 1
        self.T = float(T)
        self._evidence_signature()
        self._initialize(L, Nc, priors=priors, act=act, mask=mask, stride=stride,
            padding=padding, gamma=gamma, pooling=pooling, closure=closure,
            parameterization=parameterization, batch_size=batch_size,
            max_kernel_bytes=max_kernel_bytes, kernel_backend=kernel_backend,
            max_dense_size=max_dense_size, max_joint_coordinates=max_joint_coordinates,
            rank_policy=rank_policy)

    def _targets(self, Y, count):
        Y = as_numpy(Y)
        if Y.shape == (count, 1):
            Y = Y[:, 0]
        if Y.shape != (count,) or not np.all(np.isfinite(Y)):
            raise ValueError(f"Y must be finite with shape ({count},) or ({count},1)")
        return Y.copy()

    def _evidence_signature(self):
        if not np.isfinite(self.T) or self.T < 0:
            raise ValueError("T must be finite and nonnegative")
        return (self.T,)

    def _evidence(self, K):
        self._evidence_signature()
        C = (K+K.T)/2 + self.T*np.eye(len(K))
        try:
            factor = cho_factor(C, lower=True, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Gaussian training covariance is not positive definite; "
                "at T=0 use independent inputs or positive T") from error
        y = self.Y.reshape(-1)
        alpha = cho_solve(factor, y, check_finite=False)
        inverse = cho_solve(factor, np.eye(len(K)), check_finite=False)
        nll = np.log(np.diag(factor[0])).sum()+.5*y @ alpha
        gradient = .5*(inverse-np.outer(alpha, alpha))
        return float(nll), gradient, SimpleNamespace(alpha=alpha, reduction=inverse)

    def predict(self, Xtest, batch_size=None, return_cov=False):
        """Conditional mean and optional per-example latent covariance."""
        mean, covariance = self._predict_gaussian(Xtest, batch_size)
        self.Ypred, self.predictive_covariance = mean, covariance
        self.predictive_variance = np.diagonal(covariance, axis1=1, axis2=2).copy()
        self._prediction = (mean.copy(), self.predictive_variance.copy(), self._evidence_signature())
        return (mean, covariance) if return_cov else mean

    def averageLoss(self, Ytest, per_output=False):
        if self._prediction is None or self._prediction[2] != self._evidence_signature():
            raise RuntimeError("call predict at the current state and temperature first")
        mean, variance, _ = self._prediction
        bias = (self._targets(Ytest, len(mean)).reshape(-1, self.c)-mean)**2
        axis = 0 if per_output else None
        result = (bias+variance).mean(axis=axis), bias.mean(axis=axis), variance.mean(axis=axis)
        return result if per_output else tuple(float(value) for value in result)
