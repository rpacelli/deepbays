"""Convolution conventions shared by the finite network and stacked kernels.

Spatial pairs are tuples, e.g. ``mask=(3, 5)``. Lists specify layers, e.g.
``mask=[(3, 5), 1, 1]``. Patches are ordered by row, then column, exactly as
flattening the last two dimensions of a PyTorch NCHW tensor.
"""

from dataclasses import dataclass
from numbers import Integral
import numpy as np


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def spatial_pair(value, name, allow_zero=False):
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = (value, value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an integer or a (height, width) tuple")
    minimum = 0 if allow_zero else 1
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < minimum for x in value):
        raise ValueError(f"invalid {name}: {value}")
    return tuple(int(x) for x in value)


def per_layer(value, L, name):
    if isinstance(value, list):
        if len(value) != L:
            raise ValueError(f"{name} must have {L} layer entries; use a tuple for a spatial pair")
        return value
    return [value] * L


def layer_precisions(values, L):
    """L convolution precisions followed by the readout precision.

    As in FC_deep_vanilla, two entries abbreviate [first, remaining, ...].
    """
    values = np.ones(L + 1) if values is None else np.asarray(values, dtype=float)
    if values.shape == (2,) and L > 1:
        values = np.r_[values[0], np.repeat(values[1], L)]
    if values.shape != (L + 1,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"precisions must contain {L + 1} positive finite values (or the two-entry shorthand)")
    return tuple(values)


@dataclass(frozen=True)
class ConvLayerGeometry:
    input_shape: tuple
    output_shape: tuple
    kernel_size: tuple
    stride: tuple
    padding: tuple  # left, right, top, bottom, as in torch.nn.ZeroPad2d

    @property
    def area(self):
        return self.kernel_size[0] * self.kernel_size[1]

    @property
    def patches(self):
        return self.output_shape[0] * self.output_shape[1]

    def indices(self):
        """(output patch, filter offset) -> input patch; -1 denotes padding."""
        h, w = self.input_shape
        oh, ow = self.output_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        left, _, top, _ = self.padding
        rows = np.repeat(np.arange(oh), ow)[:, None] * sh + np.repeat(np.arange(kh), kw) - top
        cols = np.tile(np.arange(ow), oh)[:, None] * sw + np.tile(np.arange(kw), kh) - left
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        return np.where(valid, rows * w + cols, -1)


def convolution_geometry(image_shape, L, mask=3, stride=1, padding="valid"):
    """Resolve all spatial shapes and zero padding without constructing a net.

    'same' gives ceil(input / stride) patches and puts any odd extra padding on
    the right/bottom. Integer or (height, width) padding is symmetric. 'valid'
    is ordinary unpadded convolution, including PyTorch's floor at the edge.
    """
    L = positive_int(L, "L")
    shape = spatial_pair(tuple(image_shape), "image_shape")
    layers = []
    for k, s, p in zip(per_layer(mask, L, "mask"), per_layer(stride, L, "stride"), per_layer(padding, L, "padding")):
        k, s = spatial_pair(k, "mask"), spatial_pair(s, "stride")
        if isinstance(p, str):
            if p == "valid":
                pad = (0, 0, 0, 0)
            elif p == "same":
                out = tuple((a + b - 1) // b for a, b in zip(shape, s))
                total = tuple(max((o - 1) * b + c - a, 0) for a, b, c, o in zip(shape, s, k, out))
                ph, pw = total
                pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            else:
                raise ValueError("padding must be 'valid', 'same', an integer, or a spatial tuple")
        else:
            ph, pw = spatial_pair(p, "padding", allow_zero=True)
            pad = (pw, pw, ph, ph)
        left, right, top, bottom = pad
        out = ((shape[0] + top + bottom - k[0]) // s[0] + 1,
               (shape[1] + left + right - k[1]) // s[1] + 1)
        if min(out) < 1:
            raise ValueError(f"convolution {len(layers) + 1}: mask {k} does not fit image {shape} with padding {pad}")
        layers.append(ConvLayerGeometry(shape, out, k, s, pad))
        shape = out
    return tuple(layers)
