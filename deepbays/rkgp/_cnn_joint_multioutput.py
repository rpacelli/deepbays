"""Output-coupled joint kernels, sharing the scalar frozen spatial features.

Q is path-major/output-minor; likelihood vectors are example-major. H already
contains every pooling and precision normalization: no extra 1/r or 1/c.
"""

import numpy as np
from ..kernels.multioutput import contract_blocks, contract_self
from ._cnn_joint_rate import RepeatedMap, OutputGatherMap, sym


class MultioutputJointFeatures:
    def __init__(self, spatial, outputs):
        self.spatial, self.c = spatial, outputs
        self.ranks = tuple(r*outputs for r in spatial.ranks)
        self.maps = tuple(RepeatedMap(b.m, b.upper*outputs) if isinstance(b, RepeatedMap)
                          else OutputGatherMap(b, outputs) for b in spatial.maps)

    def __getattr__(self, name):
        # Geometry, immutable H cache and preparation stay in spatial units.
        return getattr(object.__getattribute__(self, 'spatial'), name)

    @property
    def info(self):
        return dict(self.spatial.info, ranks=self.ranks, spatial_ranks=self.spatial.ranks,
                    outputs=self.c)

    def _slice(self, section):
        return slice(section.start*self.c, section.stop*self.c)

    def kernel(self, Q):
        f = self.spatial
        if f.H is not None:
            return sym(contract_blocks(f.H, Q, self.c, 1.))
        K = np.empty((f.n*self.c, f.n*self.c))
        for a, b in f._pairs(f.n, f.n, symmetric=True):
            H = f.block(f._subset(f.prepared, a), f._subset(f.prepared, b))
            ia, ib = self._slice(a), self._slice(b)
            K[ia, ib] = contract_blocks(H, Q, self.c, 1.)
            if a.start != b.start:
                K[ib, ia] = K[ia, ib].T
        return sym(K)

    def adjoint(self, G):
        f, c = self.spatial, self.c
        def block(H, score):
            return np.einsum('mnij,manb->iajb', H,
                score.reshape(H.shape[0], c, H.shape[1], c), optimize=True).reshape(
                    self.ranks[0], self.ranks[0])
        if f.H is not None:
            return sym(block(f.H, G))
        gradient = np.zeros((self.ranks[0], self.ranks[0]))
        for a, b in f._pairs(f.n, f.n, symmetric=True):
            H = f.block(f._subset(f.prepared, a), f._subset(f.prepared, b))
            # The opposite example block contributes the matrix transpose.
            # Final symmetrization turns 2*block into block + block.T.
            factor = 1 if a.start == b.start else 2
            gradient += factor*block(H, G[self._slice(a), self._slice(b)])
        return sym(gradient)

    def cross(self, prepared, Q):
        f, count = self.spatial, len(prepared[0])
        K = np.empty((count*self.c, f.n*self.c))
        for a, b in f._pairs(count, f.n):
            H = f.block(f._subset(prepared, a), f._subset(f.prepared, b))
            K[self._slice(a), self._slice(b)] = contract_blocks(H, Q, self.c, 1.)
        return K

    def diagonal(self, prepared, Q):
        f, count = self.spatial, len(prepared[0])
        blocks = np.empty((count, self.c, self.c))
        for start in range(0, count, f.batch_size):
            a = slice(start, start+f.batch_size)
            H = f.block(f._subset(prepared, a), paired=True)
            blocks[a] = contract_self(H, Q, self.c, 1.)
        return (blocks+blocks.transpose(0, 2, 1))/2
