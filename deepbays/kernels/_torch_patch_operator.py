"""Optional linear_operator adapter. Not imported by legacy deepbays paths."""

import torch
from linear_operator.operators import LinearOperator


class TorchPatchOperator(LinearOperator):
    """Symmetric covariance with fixed H and differentiable full SPD Q.

    Use a complete symmetric Q (e.g. matrix_exp of symmetric coordinates).
    Off-diagonal H_ij and Q_ij individually need not be symmetric or PSD.
    This initial adapter supports unbatched operators and batched RHS columns.
    """

    def __init__(self, H, Q, normalization=1.):
        super().__init__(H, Q, normalization=normalization)
        self.H, self.Q, self.normalization = H, Q, normalization
        self.P, _, self.d, _ = H.shape
        self.c = Q.shape[0] // self.d

    def _size(self):
        return torch.Size((self.P * self.c, self.P * self.c))

    def _transpose_nonbatch(self):
        return self

    def _matmul(self, rhs):
        v = rhs.reshape(self.P, self.c, -1)
        q = self.Q.reshape(self.d, self.c, self.d, self.c)
        terms = []
        for i in range(self.d):
            for j in range(self.d):
                hv = (self.H[:, :, i, j] @ v.reshape(self.P, -1)).reshape(v.shape)
                terms.append(torch.einsum('mbr,ab->mar', hv, q[i, :, j, :]))
        return torch.stack(terms).sum(0).reshape(rhs.shape) / self.normalization

    def _diagonal(self):
        k = torch.arange(self.P, device=self.H.device)
        q = self.Q.reshape(self.d, self.c, self.d, self.c)
        return torch.einsum('mij,iaja->ma', self.H[k, k], q).reshape(-1) / self.normalization

    def _get_indices(self, row_index, col_index, *batch_indices):
        if batch_indices:
            raise NotImplementedError('batched patch operators are not supported')
        h = self.H[row_index // self.c, col_index // self.c]
        q = self.Q.reshape(self.d, self.c, self.d, self.c)
        # Advanced indexing places the selected output dimensions first.
        selected = q.permute(1, 3, 0, 2)[row_index % self.c, col_index % self.c]
        return (h * selected).sum((-2, -1)) / self.normalization
