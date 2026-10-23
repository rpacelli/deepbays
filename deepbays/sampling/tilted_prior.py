"""Full-path FC kernel projections under fixed scalar biases of the prior.

Only the target is defined here. Ordinary Pyro HMC/NUTS, adaptation, storage,
and diagnostics are shared with posterior sampling. EWA shapes the initial
broad umbrella; normalization is measured separately from network samples.
"""
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class UmbrellaTilt:
    """Bounded scalar umbrella, conditioned on Q>0.

    Enable either or both tails with left/right. Each branch and its first
    derivative vanish at Q=1; smooth coordinate caps keep the bias bounded.
    The reporting endpoints lower/upper are not hard support boundaries.
    Calibration learns left_scale/left_offset and/or slope/offset; optional
    curvature is a fixed right-branch correction, zero by default.
    Use calibrate_umbrella, then freeze the bias for fresh production chains.
    """
    slope: float = 0.
    offset: float = -1.
    upper: float = 5.
    guard: float = .5
    transition: float = .25
    curvature: float = 0.
    lower: float = .2
    left_guard: float = .1
    left_scale: float = 1.
    left_offset: float = -1.
    left: bool = True
    right: bool = True

    def __post_init__(self):
        for name in ('left', 'right'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError('left/right must be boolean tail switches')
            object.__setattr__(self, name, bool(getattr(self, name)))
        if not (self.left or self.right):
            raise ValueError('enable at least one umbrella tail; use PowerTilt(0) for the prior')
        for name, value in asdict(self).items():
            if name not in ('left', 'right'):
                if not np.isfinite(value):
                    raise ValueError('umbrella coefficients must be finite')
                object.__setattr__(self, name, float(value))
        if (self.upper <= 1 or self.guard <= 0 or self.transition <= 0
                or not 0 < self.left_guard < self.lower < 1 or self.left_scale < 0):
            raise ValueError('require upper>1, positive guard/transition, '
                             '0<left_guard<lower<1 and left_scale>=0')

    def log_weight(self, q, *, width, layers):
        if not torch.is_tensor(q):
            q = np.asarray(q, dtype=float)
        value = q*0.
        if self.left:
            value = value+self._left_log_weight(q, width=width, layers=layers)
        if self.right:
            value = value+self._right_log_weight(q, width=width, layers=layers)
        return value

    def _right_log_weight(self, q, *, width, layers):
        tensor = torch.is_tensor(q)
        if not tensor:
            q = np.asarray(q, dtype=float)
        positive = (lambda x: x.clamp_min(0)) if tensor else (lambda x: np.maximum(x, 0))
        expm1, log1p = (torch.expm1, torch.log1p) if tensor else (np.expm1, np.log1p)
        excess = positive(q-self.upper)
        saturated = self.upper-self.guard*expm1(-excess/self.guard)
        coordinate = torch.where(q > self.upper, saturated, q) if tensor else np.where(q > self.upper, saturated, q)
        r = positive(coordinate-1)
        log_root = log1p(r)/layers
        rate = width*layers/2*(expm1(log_root)-log_root)
        hinge = r+self.transition*expm1(-r/self.transition)
        gate = r.square()/(r.square()+self.transition**2) if tensor else r*r/(r*r+self.transition**2)
        raw = rate+self.slope*hinge+self.offset*gate
        if self.curvature != 0:
            # A unit-height bump fills a poorly visited connecting region.
            # Its value and derivative vanish at both reporting endpoints.
            t = (r/(self.upper-1)).clamp_max(1) if tensor else np.minimum(r/(self.upper-1), 1)
            raw = raw+self.curvature*16*t*t*(1-t)*(1-t)
        return (torch.logaddexp(raw, torch.zeros_like(raw)) if tensor
                else np.logaddexp(raw, 0.))-math.log(2.)

    def _left_log_weight(self, q, *, width, layers):
        tensor = torch.is_tensor(q)
        if not tensor:
            q = np.asarray(q, dtype=float)
        below = (q-self.lower).clamp_max(0) if tensor else np.minimum(q-self.lower, 0)
        expm1, log = (torch.expm1, torch.log) if tensor else (np.expm1, np.log)
        capped = self.lower+self.left_guard*expm1(below/self.left_guard)
        coordinate = torch.where(q < self.lower, capped, q) if tensor else np.where(q < self.lower, capped, q)
        coordinate = coordinate.clamp_max(1) if tensor else np.minimum(coordinate, 1)
        y = log(coordinate)/layers
        rate = width*layers/2*(expm1(y)-y)
        distance = 1-coordinate
        gate = distance**2/(distance**2+self.transition**2)
        raw = self.left_scale*rate+self.left_offset*gate
        left = ((torch.logaddexp(raw, torch.zeros_like(raw)) if tensor
                 else np.logaddexp(raw, 0.))-math.log(2.))
        return left


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
        if not isinstance(self.tilt, (PowerTilt, RootTilt, UmbrellaTilt)):
            raise TypeError('tilt must be PowerTilt, RootTilt or UmbrellaTilt')
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
        self.mean_activation_overlaps = []
        for lam in self.precisions:
            if activation == 'relu':
                mean = np.sqrt(np.maximum(np.diag(kernel)/lam, 0.))
                self.mean_activation_overlaps.append(float(f@mean/(np.linalg.norm(f)*np.linalg.norm(mean))))
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
        q = self.q(z)
        energy = gaussian_energy-self.tilt.log_weight(q, width=self.width, layers=self.layers)
        return torch.where(q > 0, energy, torch.full_like(energy, torch.inf)) if isinstance(self.tilt, UmbrellaTilt) else energy

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
        values = dict(Q=projections[-1], prefix_projection=projections,
                      signed_projection=signed_projections, kernel_trace=traces,
                      gaussian_energy=layer_energies)
        if isinstance(self.tilt, UmbrellaTilt):
            values['log_bias'] = self.tilt.log_weight(values['Q'], width=self.width, layers=self.layers)
            values['inverse_bias'] = torch.exp(-values['log_bias'])
        return values

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
                    support='Q>0' if isinstance(self.tilt, UmbrellaTilt) else 'Q>=0',
                    nngp_mean_activation_overlaps=self.mean_activation_overlaps,
                    root_theta_sufficient_bound=self.root_theta_bound if math.isfinite(self.root_theta_bound) else None,
                    input_source=self.input_source, input_dimension=self.input_dimension,
                    covariance_roundoff_tolerance=self.covariance_roundoff_tolerance,
                    input_sha256=self.input_sha256, direction_sha256=self.direction_sha256,
                    compiled=self._compiled is not None, model=self.model.metadata())


def sample_tilted_prior(target, config=None, *, initial=None, output_dir=None,
                        buffer_size=16, on_chain_end=None, on_chain_state=None):
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
                            on_chain_end=on_chain_end, on_chain_state=on_chain_state)


def fit_umbrella_tilt(batches, tilt, *, width, layers, core=(.8, 1.2),
                      bulk_fraction=.3, damping=.7):
    """Fit the enabled tails' coefficients to discarded-pilot occupancy.

    batches is a sequence of (Q samples, frozen bias) pairs; bias=None denotes
    independent positive prior samples. A weighted central count anchors each
    batch, so no EWA normalizer or fitted partition function is needed. Each
    enabled side receives an equal share of the requested non-core mass,
    approximately flat in Q; the core retains the prior shape. Only observed
    bins are fitted. Disabled branches and their coefficients remain unchanged.

    Least squares is regularized and bounded. Importance-weight concentration
    on the last usable batch further damps a large update. This is a calibration
    heuristic, not a convergence or unvisited-support certificate. Production
    must use a frozen bias and fresh warmup, excluding every input batch here.
    """
    from scipy.optimize import least_squares
    from scipy.special import logsumexp
    if not isinstance(tilt, UmbrellaTilt):
        raise TypeError('tilt must be UmbrellaTilt')
    width, layers = _integer(width, 'width'), _integer(layers, 'layers')
    lo, hi = map(float, core)
    if not (tilt.lower < lo < 1 < hi < tilt.upper and 0 < bulk_fraction < 1
            and 0 < damping <= 1):
        raise ValueError('need lower < core[0] < 1 < core[1] < upper, and valid fractions')
    edges = np.r_[np.linspace(tilt.lower, lo, 9)[:-1] if tilt.left else [],
                  np.linspace(lo, hi, 9),
                  np.linspace(hi, tilt.upper, 13)[1:] if tilt.right else []]
    centers, widths = (edges[:-1]+edges[1:])/2, np.diff(edges)
    counts, denominators, skipped, used = np.zeros(len(centers)), [], [], []
    for index, (samples, bias) in enumerate(batches):
        q = np.asarray(samples, dtype=float).reshape(-1)
        if q.size < 32 or not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError('each calibration batch needs >=32 finite positive Q draws')
        if bias is not None and not isinstance(bias, UmbrellaTilt):
            raise TypeError('calibration bias must be a frozen umbrella or None for prior draws')
        anchor = (q >= lo) & (q <= hi)
        if anchor.sum() < 10:
            skipped.append(index)
            continue
        log_anchor = (np.zeros(anchor.sum()) if bias is None else
                      -bias.log_weight(q[anchor], width=width, layers=layers))
        log_b = np.zeros_like(centers) if bias is None else bias.log_weight(centers, width=width, layers=layers)
        # E sum_core exp(-B_j) = M_j * P_prior(core) / Z_j.
        denominators.append(logsumexp(log_anchor)+log_b)
        counts += np.histogram(q, edges)[0]
        used.append((q, bias))
    keep = counts >= 5
    report = dict(edges=edges.tolist(), counts=counts.tolist(), populated=keep.tolist(),
                  skipped_batches=skipped, requested_bulk_fraction=bulk_fraction,
                  core=[lo, hi], previous=asdict(tilt))
    if (not used or keep.sum() < 8
            or tilt.left and np.sum(keep & (centers < 1)) < 3
            or tilt.right and np.sum(keep & (centers > 1)) < 3):
        return tilt, dict(report, fitted=False, reason='insufficient observed bins or core visits',
                          fitted_bias=asdict(tilt))
    log_ratio = np.full(len(centers), np.nan)
    log_ratio[keep] = np.log(counts[keep]/widths[keep])-logsumexp(denominators, axis=0)[keep]
    # Preserve the prior shape in the core; request flat allocations on each side.
    tail_mass = (1-bulk_fraction)/(int(tilt.left)+int(tilt.right))
    log_allocation = np.where(centers < lo, np.log(tail_mass/(lo-tilt.lower)),
                             np.log(tail_mass/(tilt.upper-hi)))
    middle = (centers >= lo) & (centers <= hi)
    log_allocation[middle] = np.log(bulk_fraction)+log_ratio[middle]
    names = (('left_scale', 'left_offset') if tilt.left else ()) + (('slope', 'offset') if tilt.right else ())
    initial = np.array([getattr(tilt, name) for name in names])
    limits = dict(left_scale=.6, left_offset=6., slope=max(2., .08*width), offset=6.)
    steps = np.array([limits[name] for name in names])
    lower, upper = initial-steps, initial+steps
    if tilt.left:
        lower[0] = max(0., lower[0])
    def construct(parameters):
        return replace(tilt, **dict(zip(names, parameters[:len(names)])))
    confidence = np.sqrt(np.minimum(counts[keep], 50)/50)
    def residual(parameters):
        predicted = construct(parameters).log_weight(centers[keep], width=width, layers=layers)
        mismatch = predicted+log_ratio[keep]-log_allocation[keep]-parameters[-1]
        return np.r_[confidence*mismatch, .2*(parameters[:-1]-initial)/steps]
    fit = least_squares(residual, np.r_[initial, -np.log(bulk_fraction)],
                        bounds=(np.r_[lower, -np.inf], np.r_[upper, np.inf]), max_nfev=200)
    if not fit.success:
        return tilt, dict(report, fitted=False, reason='least-squares fit did not converge',
                          fitted_bias=asdict(tilt))
    latest, _ = used[-1]
    old_b = tilt.log_weight(latest, width=width, layers=layers)
    fraction = damping
    for _ in range(9):
        learned = construct(initial+fraction*(fit.x[:-1]-initial))
        change = learned.log_weight(latest, width=width, layers=layers)-old_b
        normalized = np.exp(change-logsumexp(change))
        kish = float(1/np.sum(normalized**2))
        if kish >= .25*latest.size:
            break
        if _ < 8:
            fraction *= .5
    return learned, dict(report, fitted=True, reason=None, damping_used=float(fraction),
        update_weight_kish_ess=kish, fitted_bias=asdict(learned),
        log_prior_to_core=[float(x) if np.isfinite(x) else None for x in log_ratio],
        log_allocation=[float(x) if np.isfinite(x) else None for x in log_allocation],
        observed_fit_range=[float(centers[keep].min()), float(centers[keep].max())])


def calibrate_umbrella(target, *, tilt=None, prior_q=None, prior_draws=2000,
                       rounds=3, config=None, core=(.8, 1.2), bulk_fraction=.3,
                       output_dir=None):
    """Learn and validate an umbrella bias in a fixed sampling budget.

    Returns (immutable_bias, report). The input target is not mutated. A fit
    to iid prior samples initializes the EWA coefficients; discarded HMC
    batches refine them. Every batch uses a fresh kernel and warmup. The last
    batch evaluates the returned bias without updating it afterward. Final
    states may initialize the next calibration batch, but production should
    use new chains/warmup and exclude all calibration draws.

    config controls each pilot (default: 2 chains, 150 draws, 80 warmup).
    rounds is a hard number of pilot batches, not a convergence requirement.
    coverage_ok reports observed coarse-bin coverage, separately from chain
    diagnostics; neither certifies unseen modes or arbitrary quenched directions.
    An optional output_dir saves prior draws, each native pilot and calibration.json.
    """
    import copy
    import json
    from pathlib import Path
    import time
    from .diagnostics import diagnostics_and_mcse
    started = time.perf_counter()
    if not isinstance(target, KernelTailTarget):
        raise TypeError('target must be KernelTailTarget')
    rounds, prior_draws = _integer(rounds, 'rounds'), _integer(prior_draws, 'prior_draws')
    config = (SamplerConfig(sampler='hmc', chains=2, draws=150, warmup=80,
              trajectory_length=1.5, target_accept=.9) if config is None else config)
    if not isinstance(config, SamplerConfig) or config.chains < 2 or config.draws < 16 or config.warmup < 1:
        raise ValueError('calibration needs >=2 chains, >=16 draws and fresh positive warmup')
    bias = (target.tilt if isinstance(target.tilt, UmbrellaTilt) else UmbrellaTilt()) if tilt is None else tilt
    if not isinstance(bias, UmbrellaTilt):
        raise TypeError('tilt must be UmbrellaTilt')
    if prior_q is None:
        generator = torch.Generator(device=target.device).manual_seed(config.seed+811)
        with torch.no_grad():
            prior_q = np.array([float(target.q(torch.randn(target.dimension,
                generator=generator, device=target.device, dtype=target.dtype))) for _ in range(prior_draws)])
    prior_q = np.asarray(prior_q, dtype=float)
    if prior_q.ndim != 1 or not np.isfinite(prior_q).all() or np.any(prior_q < 0):
        raise ValueError('prior_q must be finite nonnegative iid prior draws')
    batches = [(prior_q[prior_q > 0], None)]
    bias, initial_fit = fit_umbrella_tilt(batches, bias, width=target.width,
        layers=target.layers, core=core, bulk_fraction=bulk_fraction)
    folder = Path(output_dir) if output_dir is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=False)
        np.save(folder/'prior_Q.npy', prior_q, allow_pickle=False)
    report = dict(initial_fit=initial_fit, stages=[], prior_draws=len(prior_q),
                  zero_prior_draws=int(np.sum(prior_q == 0)), source_target=target.metadata(),
                  core=list(core), bulk_fraction=bulk_fraction, support='Q>0')
    initial = None
    for epoch in range(rounds):
        pilot = copy.copy(target)
        pilot.tilt, pilot._compiled = bias, None  # Compiled closures can capture the original bias.
        states = []
        pilot_config = replace(config, seed=config.seed+1009*(epoch+1))
        result = sample_tilted_prior(pilot, pilot_config, initial=initial,
            output_dir=None if folder is None else folder/f'pilot_{epoch:02d}',
            on_chain_state=lambda _, z: states.append(z))
        initial = torch.stack(states)
        q = result.get_samples('Q')
        batches.append((q, bias))
        diagnostics, _ = diagnostics_and_mcse({name: result.get_samples(name) for name in
            ('Q', 'signed_projection', 'gaussian_energy')}, chain_info=result.chain_info,
            min_chains=2, min_ess=100)
        coarse_edges = np.r_[np.linspace(bias.lower, core[0], 5)[:-1] if bias.left else [],
                              np.linspace(core[0], core[1], 3),
                              np.linspace(core[1], bias.upper, 7)[1:] if bias.right else []]
        counts = np.histogram(q, coarse_edges)[0]
        anchor_per_chain = np.sum((q >= core[0]) & (q <= core[1]), axis=1)
        coverage = bool(np.all(counts >= 5) and np.all(anchor_per_chain >= 10))
        stage = dict(bias=asdict(bias), config=asdict(pilot_config), diagnostics=diagnostics,
            mean_Q=float(q.mean()), q_range=[float(q.min()), float(q.max())],
            coarse_edges=coarse_edges.tolist(), coarse_counts=counts.tolist(),
            core_hits_per_chain=anchor_per_chain.tolist(), coverage_ok=coverage,
            outside_reporting_fraction=float(np.mean((q < coarse_edges[0]) | (q > coarse_edges[-1]))),
            chain_info=result.chain_info)
        if epoch < rounds-1:
            bias, stage['update'] = fit_umbrella_tilt(batches, bias, width=target.width,
                layers=target.layers, core=core, bulk_fraction=bulk_fraction)
        report['stages'].append(stage)
        # Until the last pilot, the next candidate has not yet been sampled.
        # Keep partial on-disk reports unambiguous after an interrupted run.
        complete = epoch == rounds-1
        report.update(completed=complete, candidate_bias=asdict(bias),
                      frozen_bias=asdict(bias) if complete else None,
                      coverage_ok=coverage if complete else None,
                      elapsed_seconds=time.perf_counter()-started)
        if folder is not None:
            temporary = folder/'calibration.json.tmp'
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            temporary.replace(folder/'calibration.json')
    report['warning'] = (None if report['coverage_ok'] else
                        'Final pilot has incomplete coarse coverage or core visits; inspect before production')
    if folder is not None:
        (folder/'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return bias, report
