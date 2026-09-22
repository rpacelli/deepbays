"""Spatial-only EWA width correction from deterministic training IW kernels.

Equal hidden widths and one *unpooled* final patch are required. At each
layer eta = tr(M**2)/tr(M)**2 is averaged as a ratio of Gaussian moments.
The rate is multiplied by L/sum(eta); its tail shape is left unchanged.
No activation derivatives, nonlinear four-point terms, or sampled networks
enter this approximation. See docs/cnn_spatial_rate.md for the projection.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve


WEIGHTINGS = ('label_free', 'iw_dual')


def spatial_path_weights(builder):
    if builder.d != 1:
        raise ValueError('spatial rate correction requires exactly one final spatial patch (before pooling)')
    weights = [None] * len(builder.layers)
    weights[-1] = np.ones(1)
    for layer in range(len(weights) - 1, 0, -1):
        geometry = builder.layers[layer]
        index = geometry.indices()
        previous = np.zeros(builder.layers[layer - 1].patches)
        for offset in range(geometry.area):
            valid = index[:, offset] >= 0
            np.add.at(previous, index[valid, offset], weights[layer][valid] / geometry.area)
        weights[layer - 1] = previous
    return weights


def _projected_eta(covariance, weights):
    # covariance[a,b,i,j] = (A.T G_ij A)[a,b]. Only its symmetric
    # output part enters (z.T A.T G_ij A z)**2.
    symmetric = (covariance + covariance.swapaxes(0, 1)) / 2
    trace = np.einsum('aaij->ij', symmetric)
    second = trace**2 + 2 * np.einsum('abij,abij->ij', symmetric, symmetric)
    mean = np.einsum('abii,i->ab', symmetric, weights)
    denominator = np.trace(mean)**2 + 2 * np.sum(mean**2)
    numerator = np.einsum('i,j,ij->', weights, weights, second)
    return numerator, denominator


def spatial_rate_statistics(builder, prepared, probe=None):
    """Stream full spatial blocks, even for a diagonal final-kernel backend.

    probe=None averages v~N(0,I_P). A (P,m) probe averages v=A z,
    z~N(0,I_m). Ratios are E[numerator]/E[denominator], not E[ratio].
    Batch size respects the builder's intermediate-block memory target.
    Projected matrices additionally require sum_l 8*m*m*p_l*p_l bytes.
    """
    started = perf_counter()
    weights = spatial_path_weights(builder)
    patches, variances = prepared
    count, spatial, channels = patches.shape
    if probe is not None:
        probe = np.asarray(probe, dtype=np.float64)
        if probe.ndim != 2 or probe.shape[0] != count or not np.isfinite(probe).all():
            raise ValueError('IW direction matrix must be finite with shape (P, m)')
        scale = np.max(np.abs(probe), initial=0.)
        if scale == 0:
            raise ValueError('IW direction matrix has no nonzero direction')
        probe = probe / scale
        projected = [np.zeros((probe.shape[1], probe.shape[1], len(w), len(w))) for w in weights]
    else:
        trace = [np.zeros((len(w), len(w))) for w in weights]
        numerator, denominator, trace_mean = np.zeros((3, len(weights)))
    bs = builder.batch_size
    for left in range(0, count, bs):
        a = slice(left, min(left + bs, count))
        for right in range(0, count, bs):
            b = slice(right, min(right + bs, count))
            covariance = patches[a].reshape(-1, channels) @ patches[b].reshape(-1, channels).T
            covariance = covariance.reshape(len(patches[a]), spatial, len(patches[b]), spatial).transpose(0, 2, 1, 3)
            for layer, w in enumerate(weights):
                if layer:
                    covariance = builder._convolve(covariance, layer)
                covariance = builder._activation(variances[layer][a][:, None, :, None], covariance,
                                                 variances[layer][b][None, :, None, :])
                if not np.isfinite(covariance).all():
                    raise FloatingPointError('non-finite spatial IW kernel in rate correction')
                if probe is not None:
                    projected[layer] += np.einsum('am,abij,bn->mnij', probe[a], covariance, probe[b], optimize=True)
                else:
                    symmetric = (covariance + covariance.swapaxes(-1, -2)) / 2
                    numerator[layer] += 2 * np.einsum('abij,abij,i,j->', symmetric, symmetric, w, w, optimize=True)
                    mean = np.einsum('abii,i->ab', covariance, w)
                    denominator[layer] += 2 * np.sum(mean**2)
                    if left == right:
                        trace[layer] += np.einsum('aaij->ij', covariance)
                        trace_mean[layer] += np.trace(mean)
    if probe is not None:
        moments = [_projected_eta(c, w) for c, w in zip(projected, weights)]
    else:
        moments = [(numerator[l] + np.einsum('ij,ij,i,j->', t, t, w, w),
                    denominator[l] + trace_mean[l]**2)
                   for l, (t, w) in enumerate(zip(trace, weights))]
    eta = []
    for (num, den), w in zip(moments, weights):
        if not np.isfinite(num + den) or den <= 0:
            raise ValueError('spatial rate correction requires nonzero IW variance in every projected layer')
        value, lower = float(num / den), 1. / np.count_nonzero(w)
        if not lower - 1e-8 <= value <= 1 + 1e-8:
            raise FloatingPointError('spatial effective rank violates its positive-covariance bounds')
        eta.append(float(np.clip(value, lower, 1.)))
    return dict(eta=eta, effective_depth=float(sum(eta)), multiplier=len(eta) / sum(eta),
                spatial_patches=[len(w) for w in weights],
                active_patches=[int(np.count_nonzero(w)) for w in weights],
                seconds=perf_counter() - started)


class CNNSpatialRate:
    """Shared opt-in correction for scalar, matrix, and softmax CNN theory."""
    @staticmethod
    def _validate_rate_settings(enabled, weighting):
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError('rate_correction must be Boolean')
        if weighting not in WEIGHTINGS:
            raise ValueError(f'correction_weighting must be one of {WEIGHTINGS}')

    def _init_rate_correction(self, enabled, weighting):
        self._validate_rate_settings(enabled, weighting)
        self.rate_correction, self.correction_weighting = bool(enabled), weighting
        self._clear_rate_correction()

    def _clear_rate_correction(self):
        self._spatial_rate_cache = {}
        self._spatial_rate_ready = False
        self.rate_multiplier = 1.
        self.rate_correction_info = dict(enabled=False, multiplier=1.)

    def _spatial_inputs(self):
        if hasattr(self, 'features'):
            return self.features.builder, self.features.train, self.operator.H[:, :, 0, 0]
        return self.builder, self._train, self.finalKNNGP

    def _dual_signature(self):
        return self._evidence_signature() if hasattr(self, '_evidence_signature') else (self.T,)

    def _iw_directions(self, kernel):
        if hasattr(self, 'beta'):
            # Laplace mode coefficients at fixed IW Q in identifiable contrast
            # coordinates. No corrected-Q or test prediction is used here.
            state = self._evidence(np.eye(self._coordinates.d))[2]
            return state.alpha.reshape(self.P, self.c)
        factor = cho_factor(kernel + self.T * np.eye(self.P), lower=True)
        return cho_solve(factor, self.Y.reshape(self.P, -1))

    def _rate_statistics(self, weighting):
        builder, prepared, kernel = self._spatial_inputs()
        spatial_path_weights(builder)  # Reject pooled multi-patch geometry too.
        key = (weighting, self._dual_signature() if weighting == 'iw_dual' else None)
        if key not in self._spatial_rate_cache:
            probe = self._iw_directions(kernel) if weighting == 'iw_dual' else None
            if probe is not None and not np.any(probe):
                info = deepcopy(self._rate_statistics('label_free'))
                info.update(weighting=weighting, fallback='label_free: zero IW duals')
            else:
                info = spatial_rate_statistics(builder, prepared, probe)
                info.update(weighting=weighting, fallback=None)
            info.update(enabled=True, version=1,
                        direction_source=('isotropic training space' if weighting == 'label_free' else
                                          'IW softmax mode contrasts' if hasattr(self, 'beta') else 'IW Gaussian mean duals'))
            self._spatial_rate_cache[key] = info
        return deepcopy(self._spatial_rate_cache[key])

    def _prepare_rate_correction(self):
        self._spatial_rate_ready = True
        if self.rate_correction:
            info = self._rate_statistics(self.correction_weighting)
            self.rate_correction_info, self.rate_multiplier = info, info['multiplier']
        self._rate_signature = self._dual_signature() if self.correction_weighting == 'iw_dual' else None

    def _rate_scale(self):
        if (self.rate_correction and self.correction_weighting == 'iw_dual'
                and self._rate_signature != self._dual_signature()):
            raise RuntimeError('likelihood settings changed; call set_rate_correction() or preprocess() to rebuild IW directions')
        return self.rate_multiplier

    def set_rate_correction(self, enabled=True, *, weighting=None):
        """Switch the rate at fixed training data, invalidating any fitted Q.

        Coefficients are cached per preprocessing and likelihood settings,
        independent of Nc. Call again after changing T/beta for IW-dual
        weighting. Changing the training data always requires preprocess().
        """
        weighting = self.correction_weighting if weighting is None else weighting
        self._validate_rate_settings(enabled, weighting)
        if not self._spatial_rate_ready:
            raise RuntimeError('call preprocess(X, Y) before switching the rate')
        # Complete the calculation before changing the selected approximation.
        info = self._rate_statistics(weighting) if enabled else dict(enabled=False, multiplier=1.)
        self.rate_correction, self.correction_weighting = bool(enabled), weighting
        self.rate_correction_info, self.rate_multiplier = info, info['multiplier']
        self._rate_signature = self._dual_signature() if weighting == 'iw_dual' else None
        if hasattr(self, '_reset_solution'):
            self._reset_solution(keep_evidence=True)
        else:
            self.optQ = self.optR = self.result = self.solution_kind = None
            self.converged = False
            self.optimization_results = []
            self._invalidate_prediction()
        return self
