"""Exact spatial-diagonal recursion for CNNs ending in one patch.

The matching-offset convolution closes on spatial diagonals when there is
no pooling. This is the same representation used by public cnn-gp and by
Neural Tangents' diagonal_spatial mode; all architecture conventions and
activation formulas here are inherited from deepbays' reference builder.
"""

import numpy as np
from .conv_kernels import StackedCNNKernel


class DiagonalCNNKernel(StackedCNNKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.d != 1:
            raise ValueError('spatial-diagonal recursion requires exactly one final patch')
        # The inherited conservative batch limit is safe, although not needed
        # for the smaller spatial-diagonal intermediates.

    def _cross_chunk(self, left, right):
        xp, xv = left
        yp, yv = right
        covariance = np.einsum('mif,nif->mni', xp, yp, optimize=True)
        for l in range(len(self.layers)):
            if l:
                index = self.maps[l - 1]
                gathered = covariance[..., np.maximum(index, 0)] * (index >= 0)
                covariance = gathered.sum(-1) / (self.layers[l].area * self.priors[l])
            covariance = self._activation(xv[l][:, None, :], covariance, yv[l][None, :, :])
        return covariance[..., None] * self.output_scale

    def self_blocks(self, prepared):
        variance = prepared[1][-1]
        return self._activation(variance, variance, variance)[..., None] * self.output_scale
