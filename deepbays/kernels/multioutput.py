"""Output-coupled covariance operations; vectors use example-major ordering.

H[mu,nu,i,j] includes the readout precision and gamma, but not 1/d.
Unlike the scalar CNN coefficients, off-diagonal patch blocks are NOT
symmetrized separately. Only H[mu,nu,i,j] = H[nu,mu,j,i] is assumed.
"""

import numpy as np
from ..conv_geometry import positive_int


def contract_blocks(H, Q, outputs, normalization):
    """Return K[mu,a,nu,b] as a dense example-major matrix."""
    d = H.shape[-1]
    q = np.asarray(Q).reshape(d, outputs, d, outputs)
    return (np.einsum('mnij,iajb->manb', H, q, optimize=True)
            .reshape(H.shape[0] * outputs, H.shape[1] * outputs) / normalization)


def contract_self(H, Q, outputs, normalization):
    d = H.shape[-1]
    return np.einsum('mij,iajb->mab', H, Q.reshape(d, outputs, d, outputs),
                     optimize=True) / normalization


class PatchCovariance:
    """Cached patch kernel, with dense, matrix-product, and adjoint operations.

    No PD-by-PD covariance or derivative tensors are retained. The optional
    torch adapter is imported lazily, keeping legacy installations usable.
    """

    def __init__(self, H, outputs, normalization=None):
        self.H = np.asarray(H, dtype=np.float64)
        if (self.H.ndim != 4 or self.H.shape[0] != self.H.shape[1]
                or self.H.shape[2] != self.H.shape[3] or min(self.H.shape) < 1
                or not np.all(np.isfinite(self.H))):
            raise ValueError('H must be finite with shape (P,P,d,d)')
        if not np.allclose(self.H, self.H.transpose(1, 0, 3, 2), rtol=1e-10, atol=1e-12):
            raise ValueError('H must have joint example/patch symmetry')
        self.P, _, self.d, _ = self.H.shape
        self.c = positive_int(outputs, 'outputs')
        self.normalization = self.d if normalization is None else float(normalization)
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError('normalization must be positive')
        self.dimension = self.d * self.c
        self.shape = (self.P * self.c,) * 2

    def dense(self, Q):
        return contract_blocks(self.H, Q, self.c, self.normalization)

    def matmul(self, Q, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        vector = rhs.ndim == 1
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError('rhs must have shape (P*c,) or (P*c,k)')
        v = rhs.reshape(self.P, self.c, -1)
        q = np.asarray(Q).reshape(self.d, self.c, self.d, self.c)
        result = np.zeros_like(v)
        for i in range(self.d):
            for j in range(self.d):
                h_v = (self.H[:, :, i, j] @ v.reshape(self.P, -1)).reshape(v.shape)
                result += np.einsum('mbr,ab->mar', h_v, q[i, :, j, :])
        result = result.reshape(self.shape[0], -1) / self.normalization
        return result[:, 0] if vector else result

    def diagonal_blocks(self, Q):
        return contract_self(self.H[np.arange(self.P), np.arange(self.P)],
                             Q, self.c, self.normalization)

    def adjoint(self, score):
        """Frobenius adjoint: <score, dK> = <adjoint(score), dQ>."""
        score = np.asarray(score).reshape(self.P, self.c, self.P, self.c)
        gradient = np.einsum('mnij,manb->iajb', self.H, score, optimize=True)
        return gradient.reshape(self.dimension, self.dimension) / self.normalization

    def bilinear_adjoint(self, left, right):
        """Adjoint for sum_s left_s.T K right_s without a dense score."""
        left = np.asarray(left).reshape(self.P, self.c, -1)
        right = np.asarray(right).reshape(self.P, self.c, -1)
        gradient = np.empty((self.d, self.c, self.d, self.c))
        for i in range(self.d):
            for j in range(self.d):
                hr = (self.H[:, :, i, j] @ right.reshape(self.P, -1)).reshape(right.shape)
                gradient[i, :, j, :] = np.einsum('mas,mbs->ab', left, hr)
        return gradient.reshape(self.dimension, self.dimension) / self.normalization

    def as_linear_operator(self, Q):
        from ._torch_patch_operator import TorchPatchOperator
        import torch
        # Copy read-only NumPy caches to avoid PyTorch's unsafe-view warning.
        H = torch.tensor(self.H, dtype=torch.float64)
        Q = Q if torch.is_tensor(Q) else torch.tensor(Q, dtype=torch.float64)
        return TorchPatchOperator(H.to(Q), Q, normalization=self.normalization)
