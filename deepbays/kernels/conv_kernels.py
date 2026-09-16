"""Cached stacked NNGP kernels for bias-free two-dimensional CNNs.

The activation formulas are reused from kernels.py. Convolution averages
covariances at *matching filter offsets*, as required by weight sharing.
Arrays use (left example, right example, left patch, right patch) ordering.
"""

import numpy as np
from . import kernels as scalar_kernels
from ..conv_geometry import convolution_geometry, layer_precisions, positive_int


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def image_batch(value):
    value = as_numpy(value)
    if value.ndim == 3:
        value = value[:, None, :, :]
    if value.ndim != 4 or min(value.shape) < 1 or not np.all(np.isfinite(value)):
        raise ValueError("images must be finite arrays of shape (P, C, H, W), or (P, H, W) for one channel")
    return value


class StackedCNNKernel:
    """Build the final activation blocks, including readout precision/gamma.

    ``batch_size`` bounds both example axes of intermediate kernels. The
    memory budget further reduces it when early layers have many patches;
    it is an intermediate-block target, not a bound on the final result or
    on all NumPy temporaries. Final train blocks require 8 * P**2 * d**2 bytes.
    """

    def __init__(self, image_shape, input_channels, L, mask=3, stride=1,
                 padding="valid", priors=None, act="erf", gamma=1.,
                 batch_size=32, max_kernel_bytes=64 * 1024**2):
        self.layers = convolution_geometry(image_shape, L, mask, stride, padding)
        self.input_channels = positive_int(input_channels, "input_channels")
        self.priors = layer_precisions(priors, L)
        self.act = {"quad": "quadratic"}.get(act, act)
        if self.act not in ("erf", "relu", "id", "square", "quadratic"):
            raise ValueError("CNN kernels support erf, relu, id, square, quadratic, and quad")
        self.kernel = getattr(scalar_kernels, "kernel_" + self.act)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite")
        self.output_scale = 1. / (self.priors[-1] * float(gamma)**2)
        self.d = self.layers[-1].patches
        self.maps = [layer.indices() for layer in self.layers[1:]]
        budget = positive_int(max_kernel_bytes, "max_kernel_bytes")
        largest = max(layer.patches for layer in self.layers)
        self.batch_size = min(positive_int(batch_size, "batch_size"),
                              max(1, int(np.sqrt(budget / (8 * largest**2)))))

    def _activation(self, left_var, cross, right_var):
        if self.act == "relu":
            # Existing ReLU formula needs protection at zero variance and at
            # correlations +/-1, including roundoff on covariance diagonals.
            valid = (left_var > 0) & (right_var > 0)
            lv, rv = np.where(left_var > 0, left_var, 1.), np.where(right_var > 0, right_var, 1.)
            bound = np.sqrt(lv * rv)
            result = self.kernel(lv, np.clip(cross, -bound, bound), rv)
            return np.where(valid, result, 0.)
        return self.kernel(left_var, cross, right_var)

    def prepare(self, X):
        """Extract first-layer patches and compute preactivation variances."""
        X = image_batch(X)
        if X.shape[1:] != (self.input_channels,) + self.layers[0].input_shape:
            raise ValueError("image channels and spatial dimensions must match the training architecture")
        first = self.layers[0]
        left, right, top, bottom = first.padding
        padded = np.pad(X, ((0, 0), (0, 0), (top, bottom), (left, right)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, first.kernel_size, axis=(-2, -1))
        windows = windows[:, :, ::first.stride[0], ::first.stride[1], :, :]
        patches = np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5)).reshape(len(X), first.patches, -1)
        # A full-image/1x1 window can still be a read-only view even after
        # ascontiguousarray, so normalization must allocate its own array.
        patches = patches / np.sqrt(self.input_channels * first.area * self.priors[0])
        variance = np.einsum("aif,aif->ai", patches, patches)
        variances = [variance]
        for layer_index, index in enumerate(self.maps, 1):
            post = self._activation(variance, variance, variance)
            gathered = post[:, np.maximum(index, 0)] * (index >= 0)[None, :, :]
            variance = gathered.sum(axis=-1) / (self.layers[layer_index].area * self.priors[layer_index])
            variances.append(variance)
        return patches, tuple(variances)

    def _convolve(self, covariance, layer_index):
        layer = self.layers[layer_index]
        result = np.zeros(covariance.shape[:-2] + (layer.patches, layer.patches), dtype=np.float64)
        for column in self.maps[layer_index - 1].T:
            out = np.flatnonzero(column >= 0)
            source = column[out]
            result[..., out[:, None], out[None, :]] += covariance[..., source[:, None], source[None, :]]
        return result / (layer.area * self.priors[layer_index])

    def _cross_chunk(self, left, right):
        xp, xv = left
        yp, yv = right
        covariance = (xp.reshape(-1, xp.shape[-1]) @ yp.reshape(-1, yp.shape[-1]).T)
        covariance = covariance.reshape(len(xp), xp.shape[1], len(yp), yp.shape[1]).transpose(0, 2, 1, 3)
        for l in range(len(self.layers)):
            if l:
                covariance = self._convolve(covariance, l)
            covariance = self._activation(xv[l][:, None, :, None], covariance, yv[l][None, :, None, :])
        return covariance * self.output_scale

    @staticmethod
    def subset(prepared, index):
        return prepared[0][index], tuple(v[index] for v in prepared[1])

    def cross(self, left, right=None):
        """Final blocks for two prepared batches; exploit symmetry for training."""
        symmetric = right is None
        right = left if symmetric else right
        n, m = len(left[0]), len(right[0])
        blocks = np.empty((n, m, self.d, self.d), dtype=np.float64)
        bs = self.batch_size
        for i in range(0, n, bs):
            a = slice(i, min(i + bs, n))
            for j in range(i if symmetric else 0, m, bs):
                b = slice(j, min(j + bs, m))
                value = self._cross_chunk(self.subset(left, a), self.subset(right, b))
                blocks[a, b] = value
                if symmetric and i != j:
                    blocks[b, a] = value.transpose(1, 0, 3, 2)
        if not np.all(np.isfinite(blocks)):
            raise FloatingPointError("non-finite CNN kernel; check input scale, activation, and precisions")
        return blocks

    def self_blocks(self, prepared):
        """Within-example patch blocks only, without a full test-test matrix."""
        patches, variances = prepared
        result = np.empty((len(patches), self.d, self.d))
        for start in range(0, len(patches), self.batch_size):
            s = slice(start, start + self.batch_size)
            covariance = np.einsum("aif,ajf->aij", patches[s], patches[s], optimize=True)
            for l in range(len(self.layers)):
                if l:
                    covariance = self._convolve(covariance, l)
                variance = variances[l][s]
                covariance = self._activation(variance[:, :, None], covariance, variance[:, None, :])
            result[s] = covariance * self.output_scale
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("non-finite CNN self kernel")
        return result


def stacked_cnn_kernel(X, L, X2=None, **kwargs):
    """Convenience function returning H[i,j,mu,nu], with row-major patches.

    Includes the readout precision and gamma**-2, but not the final 1/d or Q
    contraction. For repeated train/test calculations use StackedCNNKernel.
    """
    X = image_batch(X)
    builder = StackedCNNKernel(X.shape[-2:], X.shape[1], L, **kwargs)
    left = builder.prepare(X)
    right = None if X2 is None else builder.prepare(X2)
    return builder.cross(left, right).transpose(2, 3, 0, 1)
