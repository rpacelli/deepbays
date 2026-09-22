"""Dense reference EWA for multi-output CNNs with a Gaussian likelihood."""

from ._cnn_spatial_rate import CNNSpatialRate
from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ..kernels.conv_kernels import as_numpy
from ._matrix_kernel_model import MatrixKernelModel, CNNFeatures


class CNN_deep_multioutput(CNNSpatialRate, MatrixKernelModel):
    """Equal-channel CNNs, square loss, and a full d*D order parameter.

    This small-system
    reference uses example-major vectors Y.reshape(-1), unlike the older
    FC output-major convention. With pooling='avg', Q has dimension D and
    the fixed globally averaged hidden features define the scalar kernel.
    rate_correction=False preserves the original rate. With True, a shared
    scalar multiplies the rate, using correction_weighting='label_free'
    (default) or 'iw_dual'. Only a single pre-pooling patch is supported;
    Q remains a full output covariance. See rate_correction_info for details.
    The uncorrected action is 2I(Q) + [logdet(K_Q+T I)+y.T solve(K_Q+T I,y)]/Nc.
    """

    def __init__(self, L, Nc, D, T, priors=(1., 1.), act='erf', mask=3,
                 stride=1, padding='valid', gamma=1., batch_size=32,
                 max_kernel_bytes=64 * 1024**2, *, pooling=None,
                 kernel_backend='auto', max_dense_size=2000,
                 rate_correction=False, correction_weighting='label_free'):
        if not np.isfinite(T) or T < 0:
            raise ValueError('T must be finite and nonnegative')
        self.T, self.gamma, self.act = float(T), gamma, act
        self.pooling = pooling
        features = CNNFeatures(L, priors=priors, act=act, gamma=gamma, mask=mask,
                               stride=stride, padding=padding, pooling=pooling,
                               kernel_backend=kernel_backend, batch_size=batch_size,
                               max_kernel_bytes=max_kernel_bytes)
        self._initialize(L, Nc, D, D, features, batch_size, max_dense_size)
        self._init_rate_correction(rate_correction, correction_weighting)

    def _targets(self, Y, count):
        Y = as_numpy(Y)
        if self.D == 1 and Y.shape == (count,):
            Y = Y[:, None]
        if Y.shape != (count, self.D) or not np.all(np.isfinite(Y)):
            raise ValueError(f'Y must be finite with shape ({count},{self.D})')
        return Y.copy()

    def _evidence(self, Q):
        K = self.operator.dense(Q)
        covariance = (K + K.T) / 2
        covariance.flat[::len(K) + 1] += self.T
        factor = cho_factor(covariance, lower=True, check_finite=False)
        y = self.Y.reshape(-1)
        alpha = cho_solve(factor, y, check_finite=False)
        inverse = cho_solve(factor, np.eye(len(K)), check_finite=False)
        nll = np.log(np.diag(factor[0])).sum() + .5 * y @ alpha
        gradient = .5 * (inverse - np.outer(alpha, alpha))
        return float(nll), gradient, SimpleNamespace(alpha=alpha, reduction=inverse)

    def _evidence_signature(self):
        return (self.T,)

    def predict(self, Xtest, batch_size=None, return_cov=False):
        Q, posterior = self._solution_posterior()
        mean, cov = self._predict_gaussian(Xtest, posterior.alpha, posterior.reduction, Q, batch_size)
        self.Ypred, self.predictive_covariance = mean, cov
        self.predictive_variance = np.diagonal(cov, axis1=1, axis2=2).copy()
        # Validation symmetrizes Q, which can change roundoff-level asymmetry
        # in an optimizer result. Snapshot the actual public state for the
        # stale-prediction check, not that separately normalized copy.
        self._prediction = (mean, self.predictive_variance, self.optQ.copy())
        return (mean, cov) if return_cov else mean

    def averageLoss(self, Ytest, per_output=False):
        if self._prediction is None or not np.array_equal(self.optQ, self._prediction[2]):
            raise RuntimeError('call predict() at the current Q first')
        mean, variance, _ = self._prediction
        bias = (self._targets(Ytest, len(mean)) - mean)**2
        axis = 0 if per_output else None
        return (bias + variance).mean(axis=axis), bias.mean(axis=axis), variance.mean(axis=axis)
