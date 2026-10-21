"""Full-path FC kernel projections under power or root tilts of the prior.

Only the target is defined here. Ordinary Pyro HMC/NUTS, adaptation, storage,
and diagnostics are shared with posterior sampling. EWA never enters U(z).
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn

from ..NN import FCNet
from .posteriors import WhitenedNetwork, covariance_root
from .mcmc import SamplerConfig, sample_posterior


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class PowerTilt:
    """Multiply the prior by Q**kappa. Negative powers target the left tail.

    KernelTailTarget permits negative powers only for identity networks, where
    integrability is known exactly: kappa > -N/2. A ReLU zero atom makes every
    negative power improper; the EWA bound alone is not a nonlinear guarantee.
    """
    kappa: float

    def __post_init__(self):
        if not np.isfinite(self.kappa):
            raise ValueError('kappa must be finite')
        object.__setattr__(self, 'kappa', float(self.kappa))

    @classmethod
    def for_mean(cls, mean, *, width, layers):
        """Match the central-EWA mean, exactly the Id mean; a guide for nonlinear nets."""
        width, layers = _integer(width, 'width'), _integer(layers, 'layers')
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError('target mean must be finite and positive')
        return cls(width / 2 * np.expm1(np.log(mean) / layers))

    def log_weight(self, q, *, width, layers):
        if self.kappa == 0:
            return q * 0.
        return self.kappa * (q.log() if torch.is_tensor(q) else np.log(q))


@dataclass(frozen=True)
class RootTilt:
    """Multiply the prior by exp(N*theta*Q**(1/L)).

    Negative theta targets the left tail and is always normalizable because
    the weight is bounded by one. Positive theta has a target-dependent bound.
    For L>1 the force can be stiff near Q=0; integrability is not a mixing claim.
    """
    theta: float

    def __post_init__(self):
        if not np.isfinite(self.theta):
            raise ValueError('theta must be finite')
        object.__setattr__(self, 'theta', float(self.theta))

    def log_weight(self, q, *, width, layers):
        if self.theta == 0:
            return q * 0.
        if torch.is_tensor(q):
            # A flat inactive ReLU region has Q=0 and zero force. Avoid 0*inf
            # in autograd without clipping or modifying any positive Q.
            safe = torch.where(q > 0, q, torch.ones_like(q))
            root = torch.where(q > 0, safe.pow(1 / layers), torch.zeros_like(q))
        else:
            root = np.asarray(q)**(1 / layers)
        return width * self.theta * root


def _kernel_map(k, activation):
    """Gaussian activation second moment, including zero-variance rows.

    These closed forms use clipped correlations only for roundoff. They avoid
    the divisions by zero in the legacy generic ReLU kernel helper.
    """
    if activation == 'id':
        return k.copy()
    if activation == 'erf':
        variances = np.maximum(np.diag(k), 0.)
        scale = np.sqrt(np.outer(1 + 2*variances, 1 + 2*variances))
        return 2 / np.pi * np.arcsin(np.clip(2*k / scale, -1, 1))
    if activation == 'relu':
        variances = np.maximum(np.diag(k), 0.)
        scale = np.sqrt(np.outer(variances, variances))
        correlation = np.divide(k, scale, out=np.zeros_like(k), where=scale > 0)
        correlation = np.clip(correlation, -1, 1)
        angle = np.arccos(correlation)
        result = scale * (np.sqrt(np.maximum(0, 1-correlation**2))
                          + (np.pi-angle)*correlation) / (2*np.pi)
        np.fill_diagonal(result, variances / 2)
        return result
    raise ValueError("activation must be 'id', 'relu', or 'erf'")


class _FeaturePath(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        path = []
        for index, module in enumerate(self.features):
            x = module(x)
            if index % 3 == 2:  # FCNet.Features: Linear, fan-in normalization, activation.
                path.append(x)
        return tuple(path)


class KernelTailTarget:
    """Biased prior of Q = ||bar_f @ phi_L||² / (N * D), D=bar_f.T K_inf bar_f.

    X is prepared (P,input_dimension) data. No preprocessing is performed.
    All L hidden weight matrices are sampled jointly; there is no readout.
    Id/ReLU/Erf, a common width, zero biases, and positive layer precisions
    are supported. The NNGP denominator is fixed, not a finite-width mean fit.
    """

    def __init__(self, X, bar_f, *, width, layers, activation='relu', tilt=None,
                 precisions=1., device='cpu', dtype=torch.float64):
        self.width, self.layers = _integer(width, 'width'), _integer(layers, 'layers')
        if activation not in ('id', 'relu', 'erf'):
            raise ValueError("activation must be 'id', 'relu', or 'erf'")
        self.activation = activation
        self.tilt = PowerTilt(0.) if tilt is None else tilt
        if not isinstance(self.tilt, (PowerTilt, RootTilt)):
            raise TypeError('tilt must be PowerTilt or RootTilt')
        if isinstance(self.tilt, PowerTilt) and self.tilt.kappa < 0:
            if activation != 'id':
                raise ValueError('negative power tilts are supported only for identity; '
                                 'use RootTilt(theta<0) for nonlinear left tails')
            if self.tilt.kappa <= -self.width / 2:
                raise ValueError('identity power tilt requires kappa > -N/2')
        X, f = _array(X), _array(bar_f)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError('X must be a finite nonempty (P,input_dimension) matrix')
        if f.shape != (len(X),) or not np.isfinite(f).all() or not np.any(f):
            raise ValueError('bar_f must be a finite nonzero vector of length P')
        precision = np.asarray(precisions, dtype=float)
        if precision.ndim == 0:
            precision = np.repeat(precision, self.layers)
        if precision.shape != (self.layers,) or not np.isfinite(precision).all() or np.any(precision <= 0):
            raise ValueError('precisions must be positive: one scalar or one per hidden layer')
        self.precisions = precision.tolist()
        self.P, self.input_dimension = X.shape
        self.input_source, self.covariance_roundoff_tolerance = 'raw data: XX.T/input_dimension', None
        factor = X / math.sqrt(self.input_dimension)
        kernel = factor @ factor.T
        for lam in self.precisions:
            kernel = _kernel_map(kernel / lam, activation)
        self.denominator = float(f @ kernel @ f)
        if activation == 'id':
            self.denominator = float(np.sum((factor.T @ f)**2) / np.prod(precision))
        if not np.isfinite(self.denominator) or self.denominator <= 0:
            raise ValueError('the reference kernel projection must be strictly positive')
        if activation == 'id':
            self.root_theta_bound = self.layers / 2
        elif activation == 'erf':
            self.root_theta_bound = math.inf  # Q is bounded for erf.
        else:
            # Q <= c * product_l(||z_l||_F^2 / N). Together with AM-GM,
            # this bounds the root bias by a fraction of the Gaussian energy.
            positive_f, negative_f = np.maximum(f, 0), np.minimum(f, 0)
            directional_norm_squared = max(positive_f @ positive_f, negative_f @ negative_f)
            bound_coefficient = (directional_norm_squared * np.linalg.norm(factor, 2)**2
                                 / (self.denominator * np.prod(precision)))
            self.root_theta_bound = self.layers / (2 * bound_coefficient**(1/self.layers))
        if isinstance(self.tilt, RootTilt) and self.tilt.theta >= self.root_theta_bound:
            raise ValueError(f'root theta must be below {self.root_theta_bound:g}; '
                             'this is a sufficient (possibly conservative) ReLU bound')
        # Construct the same FC model without consuming the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            features = FCNet(self.input_dimension, self.width, self.layers, act=activation,
                             bias=False, precisions=self.precisions + [1.]).Features()
        self.model = WhitenedNetwork(_FeaturePath(features), priors=self.precisions,
                                     device=device, dtype=dtype)
        self.dimension, self.device, self.dtype = self.model.dimension, self.model.device, self.model.dtype
        self.inputs = torch.tensor(X, device=self.device, dtype=dtype)
        self.bar_f = torch.tensor(f, device=self.device, dtype=dtype)
        self.reference_kernel = kernel
        self.reference_kernel.setflags(write=False)
        self.input_sha256 = hashlib.sha256(X.tobytes()).hexdigest()
        self.direction_sha256 = hashlib.sha256(f.tobytes()).hexdigest()
        self._compiled = None

    @classmethod
    def from_factor(cls, factor, bar_f, **kwargs):
        """Use a nonzero-rank factor B with C=B B.T; includes singular covariances."""
        factor = _array(factor)
        if factor.ndim != 2 or min(factor.shape) < 1:
            raise ValueError('factor must have shape (P,rank), rank>=1')
        target = cls(factor * math.sqrt(factor.shape[1]), bar_f, **kwargs)
        target.input_source = 'supplied covariance factor'
        return target

    @classmethod
    def from_covariance(cls, covariance, bar_f, **kwargs):
        """PSD covariance, using the existing support factorization; no jitter.

        Eigenvalues at spectral roundoff are removed by covariance_root.
        The tolerance is recorded. Use from_factor to supply your own factor.
        """
        factor, tolerance = covariance_root(_array(covariance))
        target = cls.from_factor(factor, bar_f, **kwargs)
        target.input_source = 'covariance support factor'
        target.covariance_roundoff_tolerance = tolerance
        return target

    def q(self, z):
        """Last-layer projection, normalized by the fixed NNGP reference."""
        last_features = self.model(z, self.inputs)[-1]
        projection = self.bar_f @ last_features
        return projection.square().sum() / (self.width * self.denominator)

    def _potential(self, z):
        gaussian_energy = .5 * z.square().sum()
        log_bias = self.tilt.log_weight(self.q(z), width=self.width, layers=self.layers)
        return gaussian_energy - log_bias

    def potential(self, z):
        return self._potential(z) if self._compiled is None else self._compiled(z)

    def compile(self, **kwargs):
        self._compiled = torch.compile(self._potential, **kwargs)
        return self

    def observables(self, z):
        """All layer diagnostics in one forward pass; no weight history required."""
        path = self.model(z, self.inputs)
        # Prefix diagnostics use the same final-layer reference D as Q;
        # they are not individually normalized to have layer-wise mean one.
        normalization = self.width * self.denominator
        projections = torch.stack([
            (self.bar_f @ features).square().sum() / normalization for features in path])
        signed_projections = torch.stack([
            (self.bar_f @ features.mean(1)) / math.sqrt(normalization) for features in path])
        traces = torch.stack([features.square().sum() / self.width for features in path])
        layer_energies = torch.stack([
            .5 * weights.square().sum() for weights in z.split(self.model._sizes)])
        return dict(Q=projections[-1], prefix_projection=projections,
                    signed_projection=signed_projections, kernel_trace=traces,
                    gaussian_energy=layer_energies)

    def initial_states(self, chains=4, seed=0, chain_offset=0):
        """Independent prior starts, retried only if the target or gradient is invalid.

        These are initialization draws, not importance proposals. Warmup must
        remove their influence. Supply initial= for deliberate dispersed starts.
        """
        _integer(chains, 'chains')
        for name, value in (('seed', seed), ('chain_offset', chain_offset)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        states = []
        for chain in range(chain_offset, chain_offset+chains):
            initial_seed = int(np.random.SeedSequence([seed, chain, 0x7157]).generate_state(1, dtype=np.uint64)[0])
            generator = torch.Generator(device=self.device).manual_seed(initial_seed)
            for _ in range(100):
                z = torch.randn(self.dimension, generator=generator, device=self.device,
                                dtype=self.dtype, requires_grad=True)
                u = self._potential(z)
                if torch.isfinite(u) and torch.isfinite(torch.autograd.grad(u, z)[0]).all():
                    states.append(z.detach())
                    break
            else:
                raise ValueError('Could not draw a finite initial state; supply initial explicitly')
        return torch.stack(states)

    def metadata(self):
        return dict(kind='tilted_kernel_prior', P=self.P, N=self.width, L=self.layers,
                    activation=self.activation, precisions=self.precisions,
                    tilt=dict(kind=type(self.tilt).__name__, **asdict(self.tilt)),
                    dimension=self.dimension, denominator=self.denominator,
                    denominator_convention='fixed NNGP projection',
                    root_theta_sufficient_bound=self.root_theta_bound if math.isfinite(self.root_theta_bound) else None,
                    input_source=self.input_source, input_dimension=self.input_dimension,
                    covariance_roundoff_tolerance=self.covariance_roundoff_tolerance,
                    input_sha256=self.input_sha256, direction_sha256=self.direction_sha256,
                    compiled=self._compiled is not None, model=self.model.metadata())


def sample_tilted_prior(target, config=None, *, initial=None, output_dir=None,
                        buffer_size=16, on_chain_end=None):
    """Standard Pyro HMC by default; stream Q and layer diagnostics, not weights."""
    if not isinstance(target, KernelTailTarget):
        raise TypeError('target must be KernelTailTarget')
    config = SamplerConfig(sampler='hmc', trajectory_length=1.5, target_accept=.8) if config is None else config
    if not isinstance(config, SamplerConfig):
        raise TypeError('config must be SamplerConfig')
    if initial is None:
        initial = target.initial_states(config.chains, config.seed, config.chain_offset)
    return sample_posterior(target, config, initial=initial, observables=target.observables,
                            save_weights=False, output_dir=output_dir, buffer_size=buffer_size,
                            on_chain_end=on_chain_end)
