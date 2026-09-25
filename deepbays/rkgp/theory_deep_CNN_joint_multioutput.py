"""Full output-coupled path/spatial EWA with a Gaussian likelihood."""

import numpy as np
from ..conv_geometry import positive_int
from ..kernels.conv_kernels import as_numpy
from .theory_deep_CNN_joint import CNN_deep_joint


class CNN_deep_joint_multioutput(CNN_deep_joint):
    """Joint CNN saddle for D Gaussian outputs and example-major targets.

    Each state matrix has dimension D times its retained spatial/path rank.
    Predictions have shape (Ptest,D), with per-example (D,D) covariance.
    rank_policy='leading_rate' explicitly permits SPD covariance-tilt saddles
    when a finite-width empirical innovation would be singular. This remains
    a leading-rate approximation, not finite-width integration or an ELBO.
    The conservative default 'full_rank' requires Nc[l] >= D*r[l].

    Features, pooling and prior normalization match CNN_deep_joint; both
    Gram and innovation solvers, unequal widths, and streaming are supported.
    Identity and the frozen-IW erf closure are supported by default. The explicit
    allow_experimental_relu=True override attempts literal ReLU G/Z gains
    without a lifted-kernel PSD guarantee; all numerical checks remain active.
    """

    def __init__(self, L, Nc, D, T, priors=(1., 1.), act='erf', mask=3,
                 stride=1, padding='valid', gamma=1., batch_size=16,
                 max_kernel_bytes=128*1024**2, *, pooling=None, closure='auto',
                 parameterization='innovation', kernel_backend='auto',
                 max_dense_size=2000, max_joint_coordinates=20000,
                 rank_policy='full_rank', allow_experimental_relu=False):
        self.D = self.c = positive_int(D, 'D')
        self.T = float(T)
        self._evidence_signature()
        self._initialize(L, Nc, priors=priors, act=act, mask=mask, stride=stride,
            padding=padding, gamma=gamma, pooling=pooling, closure=closure,
            parameterization=parameterization, batch_size=batch_size,
            max_kernel_bytes=max_kernel_bytes, kernel_backend=kernel_backend,
            max_dense_size=max_dense_size, max_joint_coordinates=max_joint_coordinates,
            rank_policy=rank_policy, allow_experimental_relu=allow_experimental_relu)

    def _targets(self, Y, count):
        Y = as_numpy(Y)
        if self.D == 1 and Y.shape == (count,):
            Y = Y[:, None]
        if Y.shape != (count, self.D) or not np.all(np.isfinite(Y)):
            raise ValueError(f'Y must be finite with shape ({count},{self.D})')
        return Y.copy()
