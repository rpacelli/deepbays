"""Prior-tail rate and diagnostics plots using Pyro HMC and EWA normalization.

Run with no arguments for P=N=16, L=2, Erf/ReLU, and three power biases.
This is a few-minute CPU demonstration, not a production convergence claim.
Outputs include rate_function.png and sampling_diagnostics.png. The latter
shows retained Q histories, inverse-bias log weights and two kinds of ESS.

The workflow is separated into sampling and analysis. To change the smoothing
width without rerunning HMC:
    python examples/sample_tilted_prior.py replot runs/my_run --output runs/my_replot --bandwidth .06

For the left tail, use --tilt root --means .8 .5 .3. Negative powers are also
supported with --activations id; nonlinear negative powers are not certified.

See docs/tilted_prior_sampling.md for the target, rate convention and errors.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, k0e
import torch

from deepbays.sampling import (
    KernelTailTarget, PowerTilt, RootTilt, SamplerConfig, SamplingResult,
    sample_tilted_prior, diagnostics_and_mcse, central_ewa_log_normalizer,
    central_ewa_rate, estimate_density_rate, estimate_prior_density_rate,
    estimate_left_probability,
)


def json_safe(value):
    """Convert NumPy values and nonfinite diagnostics to portable JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def draw_prior(target, count, seed):
    """Independent Gaussian weights: no MCMC and no rejection/conditioning."""
    generator = torch.Generator(device=target.device).manual_seed(seed)
    q = np.empty(count)
    with torch.no_grad():
        for index in range(count):
            weights = torch.randn(target.dimension, generator=generator,
                                  device=target.device, dtype=target.dtype)
            q[index] = float(target.q(weights))
    return q


def choose_tilt(mean, prior, kind):
    """Choose a useful bias strength; the nonlinear mean need not match mean."""
    if kind == 'power':
        return PowerTilt.for_mean(mean, width=prior.width, layers=prior.layers)
    if kind == 'root':
        # This matches the large-N EWA saddle, not the exact finite-N mean.
        # Respect the actual target's conservative integrability bound, too.
        theta = prior.layers / 2 * (1 - mean**(-1 / prior.layers))
        return RootTilt(min(theta, .8 * prior.root_theta_bound))
    raise ValueError('unknown tilt kind')


def finite_ewa_log_density(q, width, layers):
    """Exact central-EWA density for one or two gamma factors."""
    shape = width / 2
    if layers == 1:
        return (shape*np.log(shape) - gammaln(shape)
                + (shape-1)*np.log(q) - shape*q)
    if layers == 2:
        argument = 2 * shape * np.sqrt(q)
        # k0e(x) = exp(x) K_0(x); the scaled function avoids tail underflow.
        return (np.log(2) + 2*shape*np.log(shape) - 2*gammaln(shape)
                + (shape-1)*np.log(q) + np.log(k0e(argument)) - argument)
    raise ValueError('closed finite-N curve supports L=1 or 2')


def analyze_prior(q, *, P, bandwidth):
    """Find a modal reference and show only a sufficiently populated range."""
    positive = q[q > 0]
    if not positive.size:
        raise ValueError('no positive prior Q draws')
    grid = np.geomspace(max(positive.min()*.8, 1e-5), positive.max()*1.1, 180)
    rate = estimate_prior_density_rate(q, grid, P=P, bandwidth=bandwidth)
    # Gaussian KDE is positive even beyond the observations. Do not interpret
    # that mathematical extension as measured tail density. Keep >=20 local
    # draws within two log-bandwidths; errors near this cutoff can be large.
    rate['plot_mask'] = rate['hits'] >= 20
    return rate


def tilt_from_metadata(metadata):
    """Reconstruct the actual saved bias, including any root-strength cap."""
    if metadata['kind'] == 'PowerTilt':
        return PowerTilt(metadata['kappa'])
    if metadata['kind'] == 'RootTilt':
        return RootTilt(metadata['theta'])
    raise ValueError('unknown tilt in saved result')


def analyze_window(result, settings, prior_rate, bandwidth):
    """Diagnose one saved tilt and reconstruct its prior density rate."""
    if result.metadata['status'] != 'complete':
        raise ValueError('rate plot requires complete sampling results')
    observables = {name: result.get_samples(name) for name in result.arrays}
    q = observables['Q']
    zero_counts = (q == 0).sum(axis=1)
    if np.any(q == 0) and np.any(q > 0):
        # A chain must explore both inactive and active regions, not just mix
        # within one of them. The indicator adds an explicit diagnostic.
        observables['zero_indicator'] = (q == 0).astype(float)
    diagnostic, _ = diagnostics_and_mcse(observables, chain_info=result.chain_info)

    # A color denotes one tilt. Its horizontal range is the central 70% of
    # positive retained Q draws, regardless of the KDE bandwidth. This avoids claiming
    # measurements far beyond the region visited by this particular tilt.
    tilt_metadata = result.metadata['posterior']['tilt']
    tilt = tilt_from_metadata(tilt_metadata)
    log_normalizer = central_ewa_log_normalizer(
        tilt, width=settings.width, layers=settings.layers)
    # The first threshold estimates an atom at zero. Other thresholds report
    # left-tail probabilities directly, without KDE smoothing or modal shifts.
    left_probability = estimate_left_probability(
        q, np.r_[0., np.sort(np.unique(settings.means))], tilt=tilt,
        width=settings.width, layers=settings.layers, log_normalizer=log_normalizer)
    row = dict(tilt=tilt_metadata, actual_mean_Q=float(q.mean()),
               diagnostic=diagnostic, chain_info=result.chain_info,
               zero_count_per_chain=zero_counts, left_probability=left_probability)
    positive = q[q > 0]
    if not positive.size:
        # Preserve the atom and diagnostics rather than fabricate a log-Q KDE.
        return dict(row, rate=None, continuous_density_status='no positive draws')
    bounds = np.quantile(positive, [.15, .85])
    points = (np.geomspace(*bounds, 7) if min(settings.means) < 1
              else np.linspace(*bounds, 7))
    rate_arguments = dict(tilt=tilt, width=settings.width, layers=settings.layers,
                          P=settings.P, log_normalizer=log_normalizer)
    rate = estimate_density_rate(q, points, bandwidth=bandwidth, **rate_arguments)
    wider = estimate_density_rate(q, points, bandwidth=bandwidth*np.sqrt(2), **rate_arguments)

    # Every tilt shares the same independently estimated prior-modal offset.
    # Its variance adds here. For the black curve, covariance matters because
    # the density and its modal offset use the *same* iid prior draws.
    shifted_rate = rate['rate'] + prior_rate['modal_shift']
    shifted_mcse = np.hypot(rate['mcse_rate'], prior_rate['mcse_modal_shift'])
    local_support = (np.all(rate['hits_per_chain'] >= 10, axis=0)
                     & (rate['contribution_ess'] >= 100))
    return dict(row, rate=rate, continuous_density_status='estimated from all draws',
                shifted_rate=shifted_rate, mcse_shifted_rate=shifted_mcse,
                bandwidth_sensitivity=np.abs(wider['rate'] - rate['rate']),
                local_support_pass=local_support)


def plot_panel(ax, activation, settings, prior_rate, windows):
    """Plot pointwise errors; masks leave gaps rather than connecting across them."""
    grid = prior_rate['points']
    usable = prior_rate['plot_mask']
    prior_curve = np.where(usable, prior_rate['shifted_rate'], np.nan)
    prior_error = prior_rate['mcse_shifted_rate']
    ax.plot(grid, prior_curve, color='black', lw=2, label='Independent prior MC')
    ax.fill_between(grid, prior_curve - 2*prior_error, prior_curve + 2*prior_error,
                    color='black', alpha=.18)

    continuous_windows = [row for row in windows if row['rate'] is not None]
    left_tail = min(settings.means) < 1
    maximum_q = max(1.15, max(settings.means)*1.5,
                    max((row['rate']['points'][-1]*1.08 for row in continuous_windows), default=0.))
    minimum_q = .6
    if left_tail:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        positive_limits = [row['rate']['points'][0] for row in continuous_windows]
        positive_limits.extend(grid[usable])
        minimum_q = .85 * min(positive_limits, default=min(settings.means))
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    theory_grid = np.geomspace(minimum_q, maximum_q, 180)
    if settings.layers <= 2:
        # The comparison curve uses its *own* mode. No empirical tilt is
        # aligned to the theory or to another window.
        mode = minimize_scalar(
            lambda log_q: -finite_ewa_log_density(np.exp(log_q), settings.width, settings.layers),
            bounds=(-8, 1), method='bounded')
        theory = (-finite_ewa_log_density(theory_grid, settings.width, settings.layers)
                  - mode.fun) / settings.P
        theory_label = 'Central EWA (finite N)'
    else:
        theory = central_ewa_rate(theory_grid, width=settings.width,
                                  layers=settings.layers, P=settings.P)
        theory_label = 'Central EWA (large N)'
    ax.plot(theory_grid, theory, '--', color='0.45', label=theory_label)

    for desired_mean, row in zip(settings.means, windows):
        label = (f'Power: EWA mean {desired_mean:g}' if row['tilt']['kind'] == 'PowerTilt'
                 else f"Root: theta={row['tilt']['theta']:.3g}")
        if row['rate'] is None:
            ax.plot([], [], label=label + ' (only Q=0 observed)')
            continue
        points = row['rate']['points']
        curve, error = row['shifted_rate'], row['mcse_shifted_rate']
        line, = ax.plot(points, curve, 'o-', ms=4, label=label,
                        markerfacecolor=None if row['diagnostic']['passed'] else 'none')
        ax.fill_between(points, curve - 2*error, curve + 2*error,
                        color=line.get_color(), alpha=.2)
        weak_support = ~row['local_support_pass']
        ax.scatter(points[weak_support], curve[weak_support], marker='x', color='red', s=30)

    passed = sum(row['diagnostic']['passed'] for row in windows)
    ax.set(title=f'{activation.upper()} | diagnostic gate {passed}/{len(windows)}',
           xlabel=r'$\mathcal{Q}$', ylabel=r'$[-\log\rho_0(\mathcal{Q})+\log\rho_{0,\max}]/P$',
           xlim=(minimum_q, maximum_q))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def plot_diagnostic_row(axes, activation, desired_mean, settings, result, analysis):
    """Show retained traces and distinguish chain mixing from weight concentration.

    For p_tilt proportional to p_prior * exp(b(Q)), inverse-bias log weights
    are -b(Q). The omitted log normalizer is constant and does not affect
    their fluctuations or Kish ESS. This global weight ESS ignores temporal
    correlations; it is not the ESS of a local density-rate estimate.
    """
    history_ax, weight_ax, ess_ax = axes
    q = result.get_samples('Q')  # (chains, retained draws), including rejections
    tilt = tilt_from_metadata(analysis['tilt'])
    with np.errstate(divide='ignore', invalid='ignore'):
        log_weights = -tilt.log_weight(q, width=settings.width, layers=settings.layers)
    finite_weights = np.isfinite(log_weights)
    weight_ess = None
    if finite_weights.all():
        # Subtract a common maximum before exponentiation to avoid overflow.
        weights = np.exp(log_weights - log_weights.max())
        weight_ess = float(weights.sum()**2 / np.square(weights).sum())

    draw_index = np.arange(1, q.shape[1] + 1)
    for chain, (q_chain, log_w_chain) in enumerate(zip(q, log_weights)):
        chain_id = result.chain_info[chain]['chain']
        line, = history_ax.plot(draw_index, q_chain, lw=.7, alpha=.8,
                                label=f'Chain {chain_id}')
        weight_ax.plot(draw_index, np.where(np.isfinite(log_w_chain), log_w_chain, np.nan),
                       color=line.get_color(), lw=.7, alpha=.8)
    parameter = (f'power κ={tilt.kappa:.3g}' if isinstance(tilt, PowerTilt)
                 else f'root θ={tilt.theta:.3g}')
    history_ax.set(title=f'{activation.upper()} | {parameter} | mean guide {desired_mean:g}',
                   xlabel='Retained draw (warmup excluded)', ylabel=r'$\mathcal{Q}$')
    history_ax.legend(fontsize=7, ncol=min(4, q.shape[0]), loc='upper right')
    weight_ax.set(title='Inverse-bias log weights', xlabel='Retained draw',
                  ylabel=r'$\log w=-b(\mathcal{Q})$ (constant omitted)')
    if not finite_weights.all():
        weight_ax.text(.02, .97, f'{(~finite_weights).sum()} nonfinite weights; ESS unavailable',
                       transform=weight_ax.transAxes, va='top', color='crimson', fontsize=8)

    # Reuse the full-chain ArviZ diagnostics already computed for the rate
    # report. A constant Q must not be presented as a healthy ESS estimate.
    q_diagnostic = analysis['diagnostic']['observables']['Q']
    constant_q = q_diagnostic['constant_coordinates'] > 0
    values = [None if constant_q else q_diagnostic['min_bulk_ess'],
              None if constant_q else q_diagnostic['min_tail_ess'], weight_ess]
    labels = ['Q bulk (MCMC)', 'Q tail (MCMC)', 'Weights (Kish)']
    colors = ['tab:blue', 'tab:blue', 'tab:orange']
    ess_ax.set_xscale('log')
    ess_ax.set_xlim(.7, 2*max([q.size] + [v for v in values if v is not None and np.isfinite(v)]))
    for index, (value, color) in enumerate(zip(values, colors)):
        if value is None or not np.isfinite(value) or value <= 0:
            ess_ax.text(.03, index, 'unavailable', transform=ess_ax.get_yaxis_transform(),
                        va='center', color='crimson', fontsize=8)
        else:
            ess_ax.plot(value, index, 'o', color=color)
            ess_ax.annotate(f'{value:.0f}', (value, index), xytext=(5, 0),
                            textcoords='offset points', va='center', fontsize=8)
    ess_ax.set_yticks(range(3), labels)
    ess_ax.set_ylim(2.6, -.6)
    ess_ax.axvline(q.size, color='.5', ls=':', lw=1)
    ess_ax.set(title=f'ESS across all chains | {q.size} draws', xlabel='Effective sample size (log scale)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.title.set_fontsize(10)
    return dict(reweighting_ess=weight_ess, total_draws=q.size,
                nonfinite_log_weights=int((~finite_weights).sum()))


def plot_saved_run(source, output, bandwidth=None):
    """Analyze saved chains without running HMC. output must already exist.

    Older runs did not save prior draws. Only for those runs, regenerate the
    inexpensive iid baseline from the archived inputs, device and seed; record
    this in the report. All newly sampled runs archive the baseline as well.
    """
    settings = argparse.Namespace(**json.loads((source / 'settings.json').read_text()))
    bandwidth = settings.bandwidth if bandwidth is None else bandwidth
    report = dict(source_run=str(source.resolve()), settings=vars(settings), bandwidth=bandwidth,
                  rate_convention='-log rho_Q / P, one prior-modal offset per activation',
                  uncertainty='Pointwise delta-method MCSE including prior-modal offset; '
                              'excludes EWA error, smoothing bias, missed modes and mode selection',
                  activations={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(settings.activations),
                               figsize=(6*len(settings.activations), 5.1), squeeze=False)
    diagnostic_rows = len(settings.activations) * len(settings.means)
    diagnostics_figure, diagnostic_axes = plt.subplots(
        diagnostic_rows, 3, figsize=(15, 2.8*diagnostic_rows + 1),
        squeeze=False, layout='constrained')
    diagnostics_figure.suptitle(
        f'Retained HMC diagnostics | P={settings.P}, N={settings.width}, L={settings.layers}\n'
        'Chain colors match between traces. ESS: blue includes MCMC dependence; '
        'orange measures weight concentration only.\n'
        'Dotted line: retained draw count. Global weight ESS does not certify local tail accuracy.',
        fontsize=11)
    start = time.perf_counter()
    for activation_index, (ax, activation) in enumerate(zip(axes[0], settings.activations)):
        prior_file = source / f'{activation}_prior_Q.npy'
        if prior_file.exists():
            q_prior = np.load(prior_file, allow_pickle=False)
            prior_source = 'archived iid draws'
        else:
            with np.load(source / 'inputs.npz', allow_pickle=False) as data:
                prior = KernelTailTarget(data['X'], data['bar_f'], width=settings.width,
                                        layers=settings.layers, activation=activation,
                                        device=settings.device)
            q_prior = draw_prior(prior, settings.prior_draws, settings.seed + 811)
            prior_source = 'regenerated iid draws from saved inputs, device and seed'
        if source.resolve() != output.resolve() or not prior_file.exists():
            np.save(output / prior_file.name, q_prior, allow_pickle=False)
        prior_rate = analyze_prior(q_prior, P=settings.P, bandwidth=bandwidth)
        windows = []
        for window in range(len(settings.means)):
            result = SamplingResult.load(source / f'{activation}_{window}_samples')
            row = analyze_window(result, settings, prior_rate, bandwidth)
            row['trace_diagnostics'] = plot_diagnostic_row(
                diagnostic_axes[activation_index*len(settings.means) + window],
                activation, settings.means[window], settings, result, row)
            windows.append(row)
        plot_panel(ax, activation, settings, prior_rate, windows)
        report['activations'][activation] = dict(prior=prior_rate, prior_source=prior_source,
                                                prior_draws=len(q_prior), windows=windows)
    report['analysis_seconds'] = time.perf_counter() - start
    write_json(output / 'report.json', report)
    figure.suptitle(f'Standard Pyro HMC | P={settings.P}, N={settings.width}, L={settings.layers}'
                   f' | {settings.tilt} tilt | log-Q bandwidth {bandwidth:g}',
                   fontsize=11 if len(settings.activations) == 1 else 12)
    figure.text(.5, .015, 'Shading: pointwise ±2 MCSE, including the shared prior-modal offset.\n'
                'EWA and smoothing bias excluded. Colors: central 70% of positive draws.\n'
                'Open markers: chain diagnostics inconclusive.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .09, 1, .94))
    figure.savefig(output / 'rate_function.png', dpi=170)
    diagnostics_figure.savefig(output / 'sampling_diagnostics.png', dpi=150)
    plt.close(figure)
    plt.close(diagnostics_figure)
    print(f'Saved {output / "rate_function.png"}', flush=True)
    print(f'Saved {output / "sampling_diagnostics.png"}', flush=True)
    return report


def parse_sampling_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--P', type=int, default=16)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--activations', nargs='+', choices=['id', 'erf', 'relu'], default=['erf', 'relu'])
    parser.add_argument('--means', type=float, nargs='+', default=[1.5, 2., 2.5],
                        help='Positive EWA mean guides; values <1 target the left tail '
                             '(use root for nonlinear activations)')
    parser.add_argument('--tilt', choices=['power', 'root'], default='power')
    parser.add_argument('--data', type=Path, help='Prepared NPZ containing X and bar_f; overrides P')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=250)
    parser.add_argument('--draws', type=int, default=600)
    parser.add_argument('--prior-draws', type=int, default=4000,
                        help='samples from the raw prior, black curve')
    parser.add_argument('--bandwidth', type=float, default=.08,
                        help='Gaussian KDE standard deviation in log Q; sets the amount of local smoothing.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path, default=Path('runs/tilted_prior_demo'))
    args = parser.parse_args()
    if args.chains < 2 or args.draws < 8 or args.prior_draws < 100 or args.threads < 1:
        parser.error('need chains>=2, draws>=8, prior-draws>=100 and threads>=1')
    if args.P < 2 or args.width < 3 or args.layers < 1:
        parser.error('need P>=2, width>=3 and layers>=1')
    if not np.isfinite(args.bandwidth) or args.bandwidth <= 0:
        parser.error('bandwidth must be finite and positive')
    if any(not np.isfinite(mean) or mean <= 0 for mean in args.means):
        parser.error('EWA target means must be finite and positive')
    if args.tilt == 'power' and min(args.means) < 1 and any(act != 'id' for act in args.activations):
        parser.error('negative powers require --activations id; use --tilt root for nonlinear left tails')
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    return args


def sample_run(args):
    """Prepare fixed inputs, save iid draws, then run ordinary HMC per tilt."""
    torch.set_num_threads(args.threads)
    if args.data:
        with np.load(args.data, allow_pickle=False) as data:
            X, direction = data['X'], data['bar_f']
        args.P = len(X)
    else:
        X = np.sqrt(args.P) * np.eye(args.P)
        direction = np.random.default_rng(args.seed).normal(size=args.P)
        direction -= direction.mean()  # Suppress first-layer ReLU mean overlap.
        direction /= np.linalg.norm(direction)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'inputs.npz', X=X, bar_f=direction)
    write_json(args.output / 'settings.json', vars(args))
    start = time.perf_counter()
    for activation in args.activations:
        target_arguments = dict(width=args.width, layers=args.layers,
                                activation=activation, device=args.device)
        prior = KernelTailTarget(X, direction, **target_arguments)
        q_prior = draw_prior(prior, args.prior_draws, args.seed + 811)
        np.save(args.output / f'{activation}_prior_Q.npy', q_prior, allow_pickle=False)
        for window, mean in enumerate(args.means):
            tilt = choose_tilt(mean, prior, args.tilt)
            target = KernelTailTarget(X, direction, tilt=tilt, **target_arguments)
            config = SamplerConfig(sampler='hmc', chains=args.chains, warmup=args.warmup,
                                   draws=args.draws, trajectory_length=1.5, target_accept=.8,
                                   seed=args.seed + 1009*(window+1))
            result = sample_tilted_prior(target, config,
                                        output_dir=args.output / f'{activation}_{window}_samples')
            print(f'{activation}, {tilt}: mean Q={result.get_samples("Q").mean():.3f}, '
                  f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
    sampling_seconds = time.perf_counter() - start
    report = plot_saved_run(args.output, args.output)
    report['sampling_seconds'] = sampling_seconds
    write_json(args.output / 'report.json', report)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'replot':
        # A separate command has no sampling options to accidentally override.
        parser = argparse.ArgumentParser(description='Replot saved draws without rerunning HMC')
        parser.add_argument('source', type=Path)
        parser.add_argument('--output', type=Path, required=True, help='New analysis directory')
        parser.add_argument('--bandwidth', type=float, help='Log-Q KDE SD; default: saved value')
        args = parser.parse_args(sys.argv[2:])
        if args.output.exists():
            parser.error('output already exists; choose a new directory')
        if args.bandwidth is not None and (not np.isfinite(args.bandwidth) or args.bandwidth <= 0):
            parser.error('bandwidth must be finite and positive')
        torch.set_num_threads(1)
        args.output.mkdir(parents=True)
        plot_saved_run(args.source, args.output, args.bandwidth)
    else:
        sample_run(parse_sampling_args())


if __name__ == '__main__':
    main()
