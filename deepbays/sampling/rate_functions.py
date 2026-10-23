"""Density rates with supplied or measured normalization, and central-EWA guides.

The central EWA takes Q=prod_l(chi2_N/N). It is exact for the zero-bias
identity network. Nonlinear EWA normalizers are an approximation; measured
inverse-bias normalizers instead use the frozen-target production draws.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, loggamma, logsumexp

from .tilted_prior import PowerTilt, RootTilt, UmbrellaTilt, _integer


def _log_moment(width, layers, power):
    a = width / 2
    return layers * (power*np.log(2/width) + gammaln(a+power) - gammaln(a))


def _left_root_log_normalizer(width, layers, theta, rtol):
    """Laplace transform of the EWA root via a one-dimensional Mellin integral.

    For a=N/2, theta<0 and 0<c<L*a, integrate along s=c+it:
      Z = (1/2pi i) integral Gamma(s) [Gamma(a-s/L)/Gamma(a)]**L
                                 * (2|theta|)**(-s) ds.
    Choose c at the real saddle to reduce oscillation; scale out its log value
    so very small Z never underflows. This avoids an alternating moment series,
    which suffers cancellation and need not converge for strong negative bias.
    SciPy/QUADPACK controls the numerical integration error, not EWA model error.
    """
    shape = width / 2
    if layers == 1:
        return float(-shape * np.log1p(-2*theta))
    if rtol < 50*np.finfo(float).eps:
        raise ValueError('negative-root quadrature needs rtol >= 50*machine epsilon')
    log_strength = np.log(2.) + np.log(abs(theta))
    upper = layers * shape
    margin = np.finfo(float).eps * max(1., upper) * 8
    saddle = brentq(lambda s: digamma(s) - digamma(shape-s/layers) - log_strength,
                    margin, upper-margin, xtol=1e-13)

    def log_integrand(s):
        return (loggamma(s) + layers*(loggamma(shape-s/layers) - gammaln(shape))
                - s*log_strength)

    log_scale = log_integrand(saddle)
    integral = quad(lambda t: np.exp(log_integrand(saddle+1j*t) - log_scale).real,
                    0, np.inf, epsabs=0., epsrel=rtol, limit=500, full_output=1)
    value, error = integral[:2]
    if len(integral) != 3 or not np.isfinite(value) or value <= 0 or error > rtol*value:
        raise ValueError('negative-root normalizer quadrature did not converge; '
                         'relax rtol or reduce the bias')
    return float(log_scale + np.log(value/np.pi))


def central_ewa_log_normalizer(tilt, *, width, layers, rtol=1e-12, max_terms=100000):
    """log E_EWA[exp(b(Q))], including all constants.

    A power requires kappa>-N/2. For theta>0, sum a positive series with an
    explicit relative remainder bound; theta<L/2 is required. For theta<0,
    use a scaled one-dimensional Mellin integral (an analytic formula at L=1).
    rtol controls truncation/integration error, not EWA model error. Numerical
    failure raises an error. max_terms applies only to the positive series.
    """
    width, layers = _integer(width, 'width'), _integer(layers, 'layers')
    if isinstance(tilt, PowerTilt):
        if tilt.kappa <= -width / 2:
            raise ValueError('central-EWA power normalizer requires kappa > -N/2')
        return float(_log_moment(width, layers, tilt.kappa))
    if not isinstance(tilt, RootTilt):
        raise TypeError('tilt must be PowerTilt or RootTilt')
    theta = tilt.theta
    if theta >= layers/2:
        raise ValueError('central-EWA root normalizer requires theta < layers/2')
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must lie in (0,1)')
    max_terms = _integer(max_terms, 'max_terms')
    if theta == 0:
        return 0.
    if theta < 0:
        return _left_root_log_normalizer(width, layers, theta, rtol)
    total = -np.inf
    for start in range(0, max_terms, 256):
        k = np.arange(start, min(start+256, max_terms), dtype=float)
        terms = k*np.log(width*theta) - gammaln(k+1) + _log_moment(width, layers, k/layers)
        total = np.logaddexp(total, logsumexp(terms))
        last = k[-1]
        # Concavity bounds each subsequent ratio, and hence the omitted mass.
        ratio = max(2*theta/layers, 2*theta*(width/2+last/layers)/(last+1))
        if ratio < 1 and terms[-1]+np.log(ratio)-np.log1p(-ratio)-total <= np.log(rtol):
            return float(total)
    raise ValueError('root series did not converge; increase max_terms or reduce theta')


def central_ewa_rate(q, *, width, layers, P=None):
    """Leading large-deviation density rate with speed P (defaults to N)."""
    width, layers = _integer(width, 'width'), _integer(layers, 'layers')
    P = width if P is None else _integer(P, 'P')
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('q must be finite and positive')
    y = np.log(q)/layers
    return width/P * layers/2 * (np.expm1(y)-y)


def _check_kde_points(points, bandwidth):
    points = np.atleast_1d(np.asarray(points, dtype=float))
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all() or np.any(points <= 0):
        raise ValueError('points must be a nonempty positive finite vector')
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive')
    return points


def _log_q_kernels(q, points, bandwidth):
    """Per-draw Gaussian kernels, scaled per evaluation point to avoid underflow.

    The last axis indexes evaluation points; all earlier axes index samples.
    A zero-Q atom contributes zero to the continuous density but still counts
    in sample averages. Callers validate that at least one draw is positive.
    """
    log_q = np.full_like(q, -np.inf)
    np.log(q, out=log_q, where=q > 0)
    distance = (log_q[..., None] - np.log(points)) / bandwidth
    log_kernel = -.5*distance**2 - np.log(bandwidth*np.sqrt(2*np.pi))
    sample_axes = tuple(range(q.ndim))
    log_scale = log_kernel.max(axis=sample_axes)
    kernels = np.exp(log_kernel - log_scale)
    return kernels, log_scale, np.abs(distance) <= 2


def estimate_prior_density_rate(q, points, *, P, bandwidth=.08):
    """Log-Q KDE and pointwise Monte Carlo errors from independent prior draws.

    q is a 1D array of iid draws, including any Q=0 atom. The reported density
    is the continuous part at positive points, normalized by *all* draws.
    bandwidth is the Gaussian standard deviation in log Q, not a KDE factor.

    In addition to the unshifted rate, return a rate shifted to zero at the
    highest density on the supplied grid. Its delta-method MCSE includes the
    covariance with this shared modal estimate. The selected grid mode is
    treated as fixed; these are first-order, pointwise errors, not simultaneous
    confidence bands. Smoothing bias and grid/mode-selection effects are omitted.
    """
    q = np.asarray(q, dtype=float)
    P = _integer(P, 'P')
    points = _check_kde_points(points, bandwidth)
    if (q.ndim != 1 or q.size < 2 or not np.isfinite(q).all()
            or np.any(q < 0) or not np.any(q > 0)):
        raise ValueError('q needs at least two finite nonnegative iid draws, some positive')
    kernels, log_scale, nearby = _log_q_kernels(q, points, bandwidth)
    mean_kernel = kernels.mean(axis=0)
    log_density = np.log(mean_kernel) + log_scale - np.log(points)

    # d log(mean kernel) = d(mean kernel) / mean kernel. Keeping the
    # per-draw contributions also gives the covariance with the modal value.
    relative_contributions = kernels / mean_kernel
    relative_mcse = relative_contributions.std(axis=0, ddof=1) / np.sqrt(q.size)
    mode_index = int(np.argmax(log_density))
    modal_shift = float(log_density[mode_index] / P)
    shifted_contributions = (relative_contributions[:, mode_index, None]
                             - relative_contributions) / P
    shifted_mcse = shifted_contributions.std(axis=0, ddof=1) / np.sqrt(q.size)
    return dict(points=points, log_density=log_density, rate=-log_density/P,
                mcse_rate=relative_mcse/P, relative_mcse=relative_mcse,
                shifted_rate=-log_density/P + modal_shift,
                mcse_shifted_rate=shifted_mcse, modal_shift=modal_shift,
                mcse_modal_shift=float(relative_mcse[mode_index]/P),
                mode_point=float(points[mode_index]), hits=nearby.sum(axis=0),
                contribution_ess=kernels.sum(axis=0)**2/(kernels**2).sum(axis=0),
                zero_count=int(np.sum(q == 0)), bandwidth=float(bandwidth),
                uncertainty='Pointwise iid delta-method MCSE, conditional on bandwidth '
                            'and selected grid mode; excludes smoothing bias and mode selection')


def _check_chains(q):
    q = np.asarray(q, dtype=float)
    if (q.ndim != 2 or q.shape[0] < 2 or q.shape[1] < 8
            or not np.isfinite(q).all() or np.any(q < 0)):
        raise ValueError('q needs (chains>=2, draws>=8) finite nonnegative samples')
    return q


def _chain_mean_mcse(values):
    """Error of a per-draw statistic, retaining chain/time axes and rejections."""
    import arviz as az
    posterior = az.from_dict(posterior={'statistic': values}).posterior
    arviz_se = np.asarray(az.mcse(posterior, method='mean')['statistic'])
    arviz_se = np.where(np.isfinite(arviz_se), arviz_se, np.inf)
    batches = min(20, values.shape[1]//4)
    batch_means = np.stack([
        part.mean(axis=1) for part in np.array_split(values, batches, axis=1)], axis=1)
    batch_se = np.sqrt(np.sum(batch_means.var(axis=1, ddof=1)/batches, axis=0))/values.shape[0]
    between_se = values.mean(axis=1).std(axis=0, ddof=1)/np.sqrt(values.shape[0])
    return np.maximum.reduce([arviz_se, batch_se, between_se])


def estimate_histogram_rate(q, edges, *, P, tilt=None, width=None, layers=None,
                            log_normalizer=None):
    """Unsmoothed bin-average prior density and pointwise log-density MCSE.

    Accept iid draws (one axis) or independent chains (chain, retained draw).
    Repetitions and draws outside edges remain in every denominator. tilt=None
    denotes unbiased prior draws, including a possible Q=0 atom; positive bins
    exclude that atom. Supplied EWA log_normalizer is optional. Otherwise the
    bias is removed by paired inverse-weight self-normalization, conditional
    on positive support for a biased target. Empty bins return NaN/inf.

    Errors include numerator/normalizer covariance and, for chains, serial and
    between-chain variation. They exclude finite-bin bias, missed support and
    any supplied normalizer's approximation error. Kish counts describe weight
    concentration, not MCMC effective sample size. No modal offset is fitted.
    """
    q = np.asarray(q, dtype=float)
    iid = q.ndim == 1
    if iid:
        if q.size < 2 or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError('iid q needs >=2 finite nonnegative observations')
        q = q[None, :]
    else:
        q = _check_chains(q)
    P = _integer(P, 'P')
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError('edges must be finite, increasing and nonnegative')
    if tilt is None:
        log_w = np.zeros_like(q)
        if log_normalizer is not None and log_normalizer != 0:
            raise ValueError('unbiased prior draws have log_normalizer=0')
        log_normalizer = 0.
    else:
        if not isinstance(tilt, (PowerTilt, RootTilt, UmbrellaTilt)):
            raise TypeError('tilt must be a supported prior bias')
        width, layers = _integer(width, 'width'), _integer(layers, 'layers')
        if np.any(q <= 0):
            raise ValueError('biased histogram estimates require positive-support samples')
        log_w = -tilt.log_weight(q, width=width, layers=layers)
    if not np.isfinite(log_w).all() or (log_normalizer is not None and not np.isfinite(log_normalizer)):
        raise ValueError('inverse log weights and supplied log normalizer must be finite')
    def mcse(values):
        return (values[0].std(axis=0, ddof=1)/np.sqrt(q.size) if iid
                else _chain_mean_mcse(values))
    scale = float(log_w.max())
    weights = np.exp(log_w-scale)
    mean_w = weights.mean()
    relative_w = weights/mean_w
    measured = log_normalizer is None
    if measured:
        log_normalizer = -scale-np.log(mean_w)
    constant_bias = (tilt is None or isinstance(tilt, PowerTilt) and tilt.kappa == 0
                     or isinstance(tilt, RootTilt) and tilt.theta == 0)
    unresolved = measured and np.ptp(weights) == 0 and not constant_bias
    normalizer_se = (float(mcse(relative_w)) if measured and not unresolved else
                     np.inf if unresolved else 0.)
    bins = len(edges)-1
    density, error = np.full(bins, np.nan), np.full(bins, np.inf)
    hits, kish = np.zeros((q.shape[0], bins), dtype=int), np.zeros(bins)
    for start in range(0, bins, 8):
        stop = min(start+8, bins)
        inside = (q[..., None] >= edges[start:stop]) & (q[..., None] < edges[start+1:stop+1])
        if stop == bins:
            inside[..., -1] |= q == edges[-1]
        inside &= q[..., None] > 0
        hits[:, start:stop] = inside.sum(axis=1)
        occupied = inside.any(axis=(0, 1))
        if not occupied.any():
            continue
        indices = np.arange(start, stop)[occupied]
        # Scale each bin independently: a rare tail density may be exp(-1000)
        # even though its biased histogram is well populated.
        log_a = np.where(inside[..., occupied], log_w[..., None], -np.inf)
        bin_scale = log_a.max(axis=(0, 1))
        a = np.exp(log_a-bin_scale)
        mean_a = a.mean(axis=(0, 1))
        density[indices] = log_normalizer+bin_scale+np.log(mean_a)-np.log(np.diff(edges)[indices])
        influence = a/mean_a
        if measured:
            influence -= relative_w[..., None]
        if not unresolved:
            error[indices] = mcse(influence)
        kish[indices] = a.sum(axis=(0, 1))**2/np.square(a).sum(axis=(0, 1))
    return dict(points=(edges[:-1]+edges[1:])/2, edges=edges, log_density=density,
        rate=-density/P, relative_mcse=error, mcse_rate=error/P, hits_per_chain=hits,
        contribution_ess=kish, log_normalizer=float(log_normalizer),
        mcse_log_normalizer=normalizer_se, weight_kish_ess=float(weights.sum()**2/np.square(weights).sum()),
        max_weight_fraction=float(weights.max()/weights.sum()),
        per_chain_log_normalizer=-scale-np.log(weights.mean(axis=1)) if measured else None,
        normalizer_source='inverse weights; conditional on Q>0' if measured else 'supplied or unbiased',
        normalizer_warning='constant observed weights; support unresolved' if unresolved else None,
        zero_count_per_chain=(q == 0).sum(axis=1),
        outside_count_per_chain=((q < edges[0]) | (q > edges[-1])).sum(axis=1),
        uncertainty='Pointwise log-density MCSE; measured normalizer covariance included; '
                    'finite-bin bias, unseen support and supplied-normalizer error excluded')


def estimate_density_rate(q, points, *, tilt, width, layers, P, log_normalizer=None,
                          bandwidth=.08):
    """KDE of log Q, then remove bias and the log-coordinate Jacobian.

    q has (independent chains, retained draws) axes, including rejections and
    any Q=0 atom. Zeros contribute zero to the positive density, but remain in
    its normalization and time series. Use estimate_left_probability at x=0
    to estimate the atom separately. At least one positive draw is required.
    Returns the unshifted rate -log(rho_Q)/P, not a survival-probability rate.
    bandwidth is the Gaussian standard deviation in log Q: at Q=x, one
    bandwidth spans [x*exp(-bandwidth), x*exp(bandwidth)]. It does not select
    the evaluation range. Finite-bandwidth smoothing affects rho_tilt.
    With a supplied log_normalizer, MCSE excludes its uncertainty; an EWA value
    introduces model error. With log_normalizer=None, estimate C=1/E_b[exp(-b)]
    from the same draws. This requires a target conditioned on Q>0 and strictly
    positive samples; it does not recover a discarded zero atom. Its MCSE
    includes covariance between the kernel average and inverse-weight average.
    Both modes account for autocorrelation and between-chain variability,
    while excluding missed modes and smoothing bias. No bins are normalized
    separately. Weight concentration diagnostics do not correct for MCMC
    dependence and cannot detect missing support.
    """
    q = _check_chains(q)
    points = _check_kde_points(points, bandwidth)
    width, layers, P = _integer(width, 'width'), _integer(layers, 'layers'), _integer(P, 'P')
    if not np.any(q > 0):
        raise ValueError('no positive samples to estimate a continuous density')
    if not isinstance(tilt, (PowerTilt, RootTilt, UmbrellaTilt)):
        raise TypeError('tilt must be PowerTilt, RootTilt or UmbrellaTilt')
    if log_normalizer is not None and not np.isfinite(log_normalizer):
        raise ValueError('log_normalizer must be finite')
    if isinstance(tilt, PowerTilt) and tilt.kappa != 0 and np.any(q == 0):
        raise ValueError('Q=0 samples are incompatible with a nonzero power tilt')
    kernels, log_scale, nearby = _log_q_kernels(q, points, bandwidth)
    mean_kernel = kernels.mean(axis=(0, 1))

    normalization = {}
    if log_normalizer is None:
        if np.any(q <= 0):
            raise ValueError('inverse-weight normalization requires strictly positive Q samples '
                             'from a target conditioned on Q>0')
        log_weights = -tilt.log_weight(q, width=width, layers=layers)
        if not np.isfinite(log_weights).all():
            raise ValueError('inverse log weights must be finite')
        scale = float(np.max(log_weights))
        weights = np.exp(log_weights-scale)
        mean_weight = weights.mean()
        relative_weights = weights/mean_weight
        log_normalizer = float(-scale-np.log(mean_weight))
        # d log(mean K / mean w) = d mean K / mean K - d mean w / mean w.
        # Keeping the paired statistic includes their shared normalization noise.
        influence = kernels/mean_kernel - relative_weights[..., None]
        relative_mcse = _chain_mean_mcse(influence)
        globally_constant = ((isinstance(tilt, PowerTilt) and tilt.kappa == 0)
                             or (isinstance(tilt, RootTilt) and tilt.theta == 0))
        normalizer_warning = None
        if weights.var() > 0:
            normalizer_mcse = float(_chain_mean_mcse(relative_weights))
        elif globally_constant:
            normalizer_mcse = 0.
        else:
            # A flat observed part of a nonconstant bias cannot establish the
            # normalization of unvisited regions, however large its Kish ESS.
            normalizer_mcse = np.inf
            relative_mcse = np.full_like(relative_mcse, np.inf)
            normalizer_warning = 'constant observed weights; normalization support is unresolved'
        normalization = dict(
            mcse_log_normalizer=normalizer_mcse,
            weight_kish_ess=float(weights.sum()**2/np.sum(weights**2)),
            max_weight_fraction=float(weights.max()/weights.sum()),
            per_chain_log_normalizer=-scale-np.log(weights.mean(axis=1)),
            normalizer_warning=normalizer_warning,
            normalizer_source='inverse weights; conditional on Q>0')
        uncertainty = ('Pointwise MCSE includes estimated normalizer covariance; '
                       'excludes smoothing bias and missed modes/support')
    else:
        # The ESS of Q alone is not the ESS of a local density estimate.
        relative_mcse = _chain_mean_mcse(kernels)/mean_kernel
        uncertainty = 'MCSE excludes EWA error, smoothing bias and missed modes'

    # rho_0,Q(x) = Z_b * exp(-b(x)) * rho_b,logQ(log x) / x.
    # The bias at the evaluation point is fixed; the normalizer may be measured.
    log_density = (np.log(mean_kernel)+log_scale+log_normalizer
                   - tilt.log_weight(points, width=width, layers=layers)-np.log(points))
    return dict(points=points, log_density=log_density, rate=-log_density/P,
                mcse_rate=relative_mcse/P, relative_mcse=relative_mcse,
                hits_per_chain=nearby.sum(axis=1),
                zero_count_per_chain=(q == 0).sum(axis=1),
                # Kish concentration of kernel weights, not an autocorrelation ESS.
                contribution_ess=kernels.sum(axis=(0, 1))**2/(kernels**2).sum(axis=(0, 1)),
                bandwidth=float(bandwidth), log_normalizer=float(log_normalizer),
                uncertainty=uncertainty, **normalization)


def estimate_left_probability(q, thresholds, *, tilt, width, layers, log_normalizer):
    """Estimate Pr_0(Q <= x) = Z_b E_b[exp(-b(Q)) 1{Q <= x}].

    This direct reweighting uses no KDE or modal offset and includes a zero
    atom. Supply x=0 to isolate that atom. Return log probabilities and their
    pointwise delta-method MCSE to avoid numerical underflow in rare tails.
    No observed events give log_probability=-inf and an unresolved (infinite)
    error, never a certificate of zero probability. EWA normalizers introduce
    model error and can even yield probabilities >1; results are not clipped.
    """
    q = _check_chains(q)
    width, layers = _integer(width, 'width'), _integer(layers, 'layers')
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    if (thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)):
        raise ValueError('thresholds must be a nonempty finite nonnegative vector')
    if not isinstance(tilt, (PowerTilt, RootTilt)) or not np.isfinite(log_normalizer):
        raise ValueError('supply a power/root tilt and finite log_normalizer')
    if isinstance(tilt, PowerTilt) and tilt.kappa != 0 and np.any(q == 0):
        raise ValueError('Q=0 samples are incompatible with a nonzero power tilt')
    events = q[..., None] <= thresholds
    log_weights = -tilt.log_weight(q, width=width, layers=layers)
    log_values = np.where(events, log_weights[..., None], -np.inf)
    has_events = events.any(axis=(0, 1))
    log_scale = np.where(has_events, log_values.max(axis=(0, 1)), 0.)
    values = np.exp(log_values - log_scale)
    mean = values.mean(axis=(0, 1))
    log_probability = np.full(thresholds.shape, -np.inf)
    relative_mcse = np.full(thresholds.shape, np.inf)
    # Constant indicators cannot establish the unobserved complement's mass.
    # Avoid asking ArviZ to diagnose an all-zero or all-constant statistic.
    variable = values.var(axis=(0, 1)) > 0
    if np.any(variable):
        relative_mcse[variable] = _chain_mean_mcse(values[..., variable])/mean[variable]
    log_probability[has_events] = (np.log(mean[has_events]) + log_scale[has_events]
                                   + log_normalizer)
    contribution_ess = np.zeros(thresholds.shape)
    contribution_ess[has_events] = (values[..., has_events].sum(axis=(0, 1))**2
                                   / (values[..., has_events]**2).sum(axis=(0, 1)))
    hits = events.sum(axis=1)
    return dict(thresholds=thresholds, log_probability=log_probability,
                mcse_log_probability=relative_mcse, hits_per_chain=hits,
                contribution_ess=contribution_ess,
                local_support_pass=(np.all(hits >= 10, axis=0) & (contribution_ess >= 100)
                                    & np.isfinite(relative_mcse)),
                log_normalizer=float(log_normalizer),
                uncertainty='Pointwise MCSE; excludes normalizer error and missed modes')
