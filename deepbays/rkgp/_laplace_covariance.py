"""Exact covariance operations used by the dense Laplace solver.

The FC path carries K0 and Q separately. The Newton system is still dense,
but multiplying by K and checking its positivity need not be dense operations.
"""
import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg.lapack import get_lapack_funcs


def block_congruence(matrix, roots):
    """C matrix C, where C consists of symmetric per-example blocks."""
    p, c, _ = roots.shape
    blocks = matrix.reshape(p, c, p, c).transpose(0, 2, 1, 3)
    result = (roots[:, None] @ blocks) @ roots[None, :]
    return result.transpose(0, 2, 1, 3).reshape(p*c, p*c)


def apply_blocks(roots, vector):
    p, c, _ = roots.shape
    return (roots @ vector.reshape(p, c, -1)).reshape(vector.shape)


class DenseCovariance:
    def __init__(self, matrix, c):
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
                or not len(matrix) or len(matrix) % c
                or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('K must be a finite symmetric matrix of dimension P*c')
        self.matrix = (matrix + matrix.T) / 2
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if eigenvalues[0] < -1e-10 * max(1., float(np.max(np.abs(eigenvalues)))):
            raise ValueError('K must be positive semidefinite')
        self.shape = self.matrix.shape
        self.p, self.c = len(matrix)//c, c

    def matmul(self, rhs):
        return self.matrix @ rhs

    def congruence(self, roots):
        return block_congruence(self.matrix, roots)

    def dense(self):
        return self.matrix

    def posterior_blocks(self, reduction):
        p, c = self.p, self.c
        product = self.matmul(reduction).reshape(p, c, p, c)
        kernel = self.matrix.reshape(p, c, p, c)
        correction = np.einsum('manb,nbmd->mad', product, kernel, optimize=True)
        return kernel[np.arange(p), :, np.arange(p), :] - correction


class SeparableCovariance:
    """Private FC covariance K0 tensor Q; K0 has already been PSD-validated.

    Constructed only from the immutable kernel cache of FCFeatures. Checking
    Q here is cheap; arbitrary user-supplied dense kernels use DenseCovariance.
    """
    def __init__(self, validated_scalar, Q):
        self.scalar = validated_scalar
        self.Q = np.asarray(Q, dtype=float).copy()
        if (self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]
                or not np.all(np.isfinite(self.Q))
                or not np.allclose(self.Q, self.Q.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('Q must be finite and symmetric')
        self.Q = (self.Q + self.Q.T)/2
        if np.linalg.eigvalsh(self.Q)[0] <= 0:
            raise ValueError('Q must be positive definite')
        self.p, self.c = len(self.scalar), len(self.Q)
        self.shape = (self.p*self.c,)*2

    def matmul(self, rhs):
        v = rhs.reshape(self.p, self.c, -1)
        left = (self.scalar @ v.reshape(self.p, -1)).reshape(v.shape)
        return (self.Q @ left).reshape(rhs.shape)

    def congruence(self, roots):
        blocks = (roots @ self.Q)[:, None] @ roots[None, :]
        blocks *= self.scalar[:, :, None, None]
        return blocks.transpose(0, 2, 1, 3).reshape(self.shape)

    def dense(self):
        return np.kron(self.scalar, self.Q)

    def posterior_blocks(self, reduction):
        product = self.matmul(reduction).reshape(self.p, self.c, self.p, self.c)
        weighted = np.einsum('manb,nm->mab', product, self.scalar, optimize=True)
        return self.scalar.diagonal()[:, None, None]*self.Q - weighted @ self.Q


def root_and_factor(covariance, W):
    values, vectors = np.linalg.eigh(W)
    tolerance = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
    if values.min() < -tolerance:
        raise FloatingPointError('softmax Hessian is not positive semidefinite')
    roots = (vectors*np.sqrt(np.maximum(values, 0.))[:, None, :]) @ vectors.transpose(0, 2, 1)
    B = covariance.congruence(roots)
    B.flat[::len(B)+1] += 1.
    B = (B+B.T)/2
    return roots, cho_factor(B, lower=True, overwrite_a=True, check_finite=False)


def reduction_from_factor(roots, factor):
    """C B^-1 C, exploiting the Cholesky factor and block structure of C.

    POTRI exploits symmetry when obtaining the full inverse of B. This is
    the same reduction previously formed with a many-RHS Cholesky solve;
    the prior covariance is never inverted. Consumes the Cholesky factor.
    """
    inverse, info = get_lapack_funcs('potri', (factor[0],))(
        factor[0], lower=int(factor[1]), overwrite_c=True)
    if info:
        raise np.linalg.LinAlgError(f'Cholesky inverse failed (POTRI info={info})')
    # Mirror one triangle without allocating global triangular index arrays.
    for i in range(len(inverse)):
        inverse[:i, i] = inverse[i, :i]
    reduction = block_congruence(inverse, roots)
    return (reduction+reduction.T)/2
