"""Fixed IW feature operators for scalar-output joint CNN saddles.

linear_physical is the exact linear spatial hierarchy on its support.
iw_path_gain is an additional nonlinear EWA closure for erf/id. ReLU requires
an explicit experimental override and has no lifted-kernel PSD guarantee.
It never evaluates activation derivatives or adapts gains at the saddle.
"""

from math import prod
import numpy as np
from ..conv_geometry import positive_int
from ..kernels.conv_kernels import StackedCNNKernel
from ._cnn_joint_rate import GatherMap, RepeatedMap, JointRate, sym


class JointCNNFeatures:
    def __init__(self, X, *, L, widths, priors, act, gamma, mask, stride, padding,
                 pooling, closure, batch_size, max_kernel_bytes,
                 max_joint_coordinates, kernel_backend, outputs=1, rank_policy='full_rank',
                 allow_experimental_relu=False):
        if pooling not in (None, 'avg'):
            raise ValueError("pooling must be None or 'avg'")
        if not isinstance(allow_experimental_relu, bool):
            raise ValueError('allow_experimental_relu must be Boolean')
        if act == 'relu' and not allow_experimental_relu:
            raise ValueError("ReLU path EWA requires allow_experimental_relu=True; "
                             "the default joint closures support only 'id' and 'erf'")
        if act not in ('id', 'erf', 'relu'):
            raise ValueError("joint CNN theories support 'id', 'erf', and opt-in experimental 'relu'")
        self.allow_experimental_relu = allow_experimental_relu
        if closure == 'auto':
            closure = 'linear_physical' if act == 'id' else 'iw_path_gain'
        if closure not in ('linear_physical', 'iw_path_gain'):
            raise ValueError("closure must be 'auto', 'linear_physical', or 'iw_path_gain'")
        if closure == 'linear_physical' and act != 'id':
            raise ValueError("linear_physical requires act='id'; use iw_path_gain for nonlinear activations")
        if kernel_backend not in ('auto', 'dense', 'stream'):
            raise ValueError("kernel_backend must be 'auto', 'dense', or 'stream'")
        self.budget = positive_int(max_kernel_bytes, 'max_kernel_bytes')
        self.coordinate_limit = positive_int(max_joint_coordinates, 'max_joint_coordinates')
        self.outputs = positive_int(outputs, 'outputs')
        self.rank_policy = JointRate.validate_rank_policy(rank_policy)
        self.closure, self.pooling = closure, pooling
        self.builder = StackedCNNKernel(X.shape[-2:], X.shape[1], L, priors=priors,
            act=act, gamma=gamma, mask=mask, stride=stride, padding=padding,
            batch_size=batch_size, max_kernel_bytes=self.budget)
        self.widths = widths
        self.ranks, self.maps, self.support_spectra = None, None, None
        if closure == 'linear_physical':
            self._physical_geometry()
            raw_dimension = self.ranks[0]
        else:
            self._path_geometry()
            raw_dimension = len(self.visited)
        self.coordinates = sum((r*self.outputs)*(r*self.outputs+1)//2 for r in self.ranks)
        largest = max(layer.patches for layer in self.builder.layers)
        # Conservative array-workspace target, not a process RSS guarantee.
        self.pair_workspace_bytes = 8 * 10 * (raw_dimension**2 + largest**2 + self.outputs**2-1)
        if self.pair_workspace_bytes > self.budget:
            raise MemoryError(f"one joint feature pair needs an estimated "
                              f"{self.pair_workspace_bytes/2**20:.2f} MiB workspace; "
                              "increase max_kernel_bytes or use a smaller geometry")
        self.batch_size = min(self.builder.batch_size,
                              max(1, int(np.sqrt(self.budget/self.pair_workspace_bytes))))
        self.prepared = self.builder.prepare(X)
        for array in (self.prepared[0], *self.prepared[1]):
            array.setflags(write=False)
        self.n = len(X)
        self.dense_bytes = 8 * self.n**2 * self.ranks[0]**2
        if kernel_backend == 'dense' and self.dense_bytes > self.budget:
            raise MemoryError(f"dense joint features need {self.dense_bytes/2**20:.2f} MiB; "
                              "use kernel_backend='stream' or increase max_kernel_bytes")
        self.backend = ('dense' if self.dense_bytes <= self.budget else 'stream'
                        ) if kernel_backend == 'auto' else kernel_backend
        self.H = None
        if self.backend == 'dense':
            r = self.ranks[0]
            self.H = np.empty((self.n, self.n, r, r))
            for a, b in self._pairs(self.n, self.n, symmetric=True):
                block = self.block(self._subset(self.prepared, a), self._subset(self.prepared, b))
                self.H[a, b] = block
                if a.start != b.start:
                    self.H[b, a] = block.transpose(1, 0, 3, 2)
            self.H.setflags(write=False)

    def _check_ranks(self, ranks):
        dimensions = [r*self.outputs for r in ranks]
        for i, (r, n) in enumerate(zip(dimensions, self.widths)):
            if r > n and self.rank_policy == 'full_rank':
                raise ValueError(f"layer {i+1}: width {n} < joint support/path rank {r}; "
                                 "use rank_policy='leading_rate' to explicitly continue "
                                 "the SPD covariance-tilt saddle below the empirical rank threshold")
        count = sum(r*(r+1)//2 for r in dimensions)
        if count > self.coordinate_limit:
            raise ValueError(f"joint state needs {count} coordinates, exceeding "
                             f"max_joint_coordinates={self.coordinate_limit}")

    def _readout(self):
        p, scale = self.builder.d, self.builder.output_scale
        if self.pooling == 'avg':
            return np.ones((p, 1))*np.sqrt(scale)/p
        return np.eye(p)*np.sqrt(scale/p)

    def _physical_geometry(self):
        layers, L = self.builder.layers, len(self.builder.layers)
        self._check_ranks([0]*(L-1) + [1 if self.pooling == 'avg' else self.builder.d])
        factors, maps, spectra = [None]*L, [None]*(L-1), [None]*L
        factors[-1] = self._readout()
        spectra[-1] = np.linalg.svd(factors[-1], compute_uv=False)
        for l in range(L-2, -1, -1):
            index = self.builder.maps[l]
            upper, lower, m = factors[l+1].shape[1], layers[l].patches, index.shape[1]
            raw_bytes = 8*m*upper*lower
            if raw_bytes > self.budget:
                raise MemoryError(f"physical gather workspace needs {raw_bytes/2**20:.2f} MiB; "
                                  "increase max_kernel_bytes or reduce spatial size")
            raw = np.zeros((m, upper, lower))
            for t in range(m):
                good = np.flatnonzero(index[:, t] >= 0)
                # A single offset is injective for positive convolution strides.
                raw[t, :, index[good, t]] = factors[l+1][good]
            raw /= np.sqrt(layers[l+1].area * self.builder.priors[l+1])
            _, s, Vt = np.linalg.svd(raw.reshape(m*upper, lower), full_matrices=False)
            threshold = 32*np.finfo(float).eps * max(m*upper, lower)*s[0]
            active = s > threshold
            if not np.any(active):
                raise ValueError("convolution/readout has empty spatial support")
            factors[l] = Vt[active].T * s[active]
            inverse = Vt[active].T / s[active]
            maps[l] = GatherMap(raw @ inverse)
            spectra[l] = s
        self.ranks = tuple(F.shape[1] for F in factors)
        self._check_ranks(self.ranks)
        self.maps, self.factors, self.support_spectra = tuple(maps), tuple(factors), tuple(spectra)

    def _path_geometry(self):
        layers, L, p = self.builder.layers, len(self.builder.layers), self.builder.d
        ranks = [1 if self.pooling == 'avg' else p]
        for layer in reversed(layers[1:]):
            ranks.insert(0, layer.area*ranks[0])
        self._check_ranks(ranks)
        raw = ranks[0] * (p if self.pooling == 'avg' else 1)
        largest = max(layer.patches for layer in layers)
        if 8*10*(raw**2+largest**2) > self.budget:
            raise MemoryError("one IW path feature pair exceeds max_kernel_bytes; "
                              "increase the budget or reduce spatial size")
        self.ranks = tuple(ranks)
        self.maps = tuple(RepeatedMap(layers[l+1].area, ranks[l+1]) for l in range(L-1))
        # Ordering t_2, ..., t_L, final position, last index fastest.
        visited = np.arange(p)[:, None]
        for l in range(L-2, -1, -1):
            index = self.builder.maps[l]
            rows = []
            for t in range(index.shape[1]):
                parent = visited[:, 0]
                lower = np.where(parent >= 0, index[np.maximum(parent, 0), t], -1)
                rows.append(np.column_stack((lower, visited)))
            visited = np.concatenate(rows)
        self.visited = np.maximum(visited, 0)
        self.valid_paths = np.all(visited >= 0, axis=1)
        self.path_scale = self.builder.output_scale / prod(
            layer.area*self.builder.priors[l] for l, layer in enumerate(layers) if l)
        self.path_scale /= p**2 if self.pooling == 'avg' else p

    @staticmethod
    def _subset(prepared, section):
        return StackedCNNKernel.subset(prepared, section)

    def _pairs(self, n, m, symmetric=False):
        bs = self.batch_size
        for i in range(0, n, bs):
            a = slice(i, min(i+bs, n))
            for j in range(i if symmetric else 0, m, bs):
                yield a, slice(j, min(j+bs, m))

    def block(self, left, right=None, *, paired=False):
        """Features (n,m,r,r), or within-example features (n,r,r)."""
        right = left if right is None else right
        xp, xv = left
        yp, yv = right
        if self.closure == 'linear_physical':
            F = self.factors[0]
            a, b = np.einsum('aif,ij->ajf', xp, F), np.einsum('aif,ij->ajf', yp, F)
            return np.einsum('aif,ajf->aij', a, b) if paired else np.einsum('aif,bjf->abij', a, b)
        covariance = (np.einsum('aif,ajf->aij', xp, yp) if paired else
                      np.einsum('aif,bjf->abij', xp, yp))
        lifted = None
        for l in range(len(self.builder.layers)):
            if l:
                covariance = self.builder._convolve(covariance, l)
            lv = xv[l][:, :, None] if paired else xv[l][:, None, :, None]
            rv = yv[l][:, None, :] if paired else yv[l][None, :, None, :]
            if l == 0:
                covariance = self.builder._activation(lv, covariance, rv)
                ids = self.visited[:, 0]
                lifted = covariance[..., ids[:, None], ids[None, :]].copy()
            elif self.builder.act == 'erf':
                scale = 1 / np.sqrt((1+2*lv)*(1+2*rv))
                rho = np.clip(2*scale*covariance, -1., 1.)
                ratio = np.ones_like(rho)
                np.divide(np.arcsin(rho), rho, out=ratio, where=rho != 0)
                gain = 4/np.pi * scale * ratio
                ids = self.visited[:, l]
                lifted *= gain[..., ids[:, None], ids[None, :]]
                covariance = self.builder._activation(lv, covariance, rv)
            elif self.builder.act == 'relu':
                # Deliberately use the uncentered mean-kernel ratio G/Z, not
                # a derivative, centered kernel, or silently clipped PSD gain.
                post = self.builder._activation(lv, covariance, rv)
                if np.any((covariance == 0) & (post != 0)):
                    raise FloatingPointError(
                        'experimental ReLU path gain G/Z is undefined: '
                        'zero preactivation covariance with nonzero ReLU covariance')
                gain = np.zeros_like(covariance)
                np.divide(post, covariance, out=gain, where=covariance != 0)
                ids = self.visited[:, l]
                lifted *= gain[..., ids[:, None], ids[None, :]]
                covariance = post
            # For id, the activation and the gain are both trivial.
        lifted *= self.valid_paths[:, None] * self.valid_paths[None, :]
        if self.pooling == 'avg':
            p, r = self.builder.d, self.ranks[0]
            lifted = lifted.reshape(lifted.shape[:-2]+(r, p, r, p)).sum(axis=(-3, -1))
        lifted *= self.path_scale
        if not np.all(np.isfinite(lifted)):
            raise FloatingPointError("non-finite joint IW feature kernel")
        return lifted

    def kernel(self, Q):
        if self.H is not None:
            return sym(np.einsum('abij,ij->ab', self.H, Q, optimize=True))
        K = np.empty((self.n, self.n))
        for a, b in self._pairs(self.n, self.n, symmetric=True):
            H = self.block(self._subset(self.prepared, a), self._subset(self.prepared, b))
            K[a, b] = np.einsum('abij,ij->ab', H, Q, optimize=True)
            if a.start != b.start:
                K[b, a] = K[a, b].T
        return sym(K)

    def adjoint(self, G):
        if self.H is not None:
            return sym(np.einsum('abij,ab->ij', self.H, G, optimize=True))
        result = np.zeros((self.ranks[0], self.ranks[0]))
        for a, b in self._pairs(self.n, self.n, symmetric=True):
            H = self.block(self._subset(self.prepared, a), self._subset(self.prepared, b))
            factor = 1 if a.start == b.start else 2
            result += factor*np.einsum('abij,ab->ij', H, G[a, b], optimize=True)
        return sym(result)

    def cross(self, prepared, Q):
        K = np.empty((len(prepared[0]), self.n))
        for a, b in self._pairs(len(prepared[0]), self.n):
            H = self.block(self._subset(prepared, a), self._subset(self.prepared, b))
            K[a, b] = np.einsum('abij,ij->ab', H, Q, optimize=True)
        return K

    def diagonal(self, prepared, Q):
        diagonal = np.empty(len(prepared[0]))
        for start in range(0, len(diagonal), self.batch_size):
            a = slice(start, start+self.batch_size)
            H = self.block(self._subset(prepared, a), paired=True)
            diagonal[a] = np.einsum('aij,ij->a', H, Q, optimize=True)
        return diagonal

    @property
    def info(self):
        return dict(closure=self.closure, ranks=self.ranks, coordinates=self.coordinates,
                    allow_experimental_relu=self.allow_experimental_relu,
                    experimental_relu=(self.builder.act == 'relu'),
                    path_gain_psd_guaranteed=(self.builder.act in ('id', 'erf')),
                    kernel_backend=self.backend, dense_feature_bytes=self.dense_bytes,
                    pair_workspace_estimate_bytes=self.pair_workspace_bytes,
                    batch_size=self.batch_size, pooling=self.pooling,
                    physical_patch_counts=tuple(l.patches for l in self.builder.layers),
                    raw_path_count=(len(self.visited) if self.closure == 'iw_path_gain' else None))
