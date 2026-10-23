"""Adaptive bridge sampling of both prior tails, with measured normalization.

Small CPU example (P=N=16, L=2, Erf; intended to finish within five minutes):
    python examples/sample_bridge_prior.py --output runs/bridge_demo
    python examples/sample_bridge_prior.py --device cuda --output runs/bridge_gpu
Replot a saved result, without sampling:
    python examples/sample_bridge_prior.py --replot runs/bridge_demo

The three figures show unsmoothed rate estimates, overlapping windows and
normalizers, and histories/mixing. Discovery chooses the ladder; its samples
are excluded from the frozen-ladder production estimate. EWA is a guide only.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import torch

from deepbays.sampling import (BridgeResult, KernelTailTarget, PowerTilt, RootTilt,
    SamplerConfig, central_ewa_rate, estimate_histogram_rate, sample_adaptive_bridge)
from sample_tilted_prior import finite_ewa_log_density


def write_json(path, data):
    """Keep unresolved uncertainties as JSON null, rather than nonstandard NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, Path):
            return str(value)
        return None if isinstance(value, float) and not np.isfinite(value) else value
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+'\n')


def prepare_target(args):
    """Supply a covariance directly, or use a small correlated synthetic input."""
    if args.data:
        with np.load(args.data, allow_pickle=False) as data:
            covariance, direction = data['C'], data['bar_f']
        args.P = len(covariance)
    else:
        distance = np.abs(np.arange(args.P)[:, None]-np.arange(args.P))
        covariance = .3**distance
        direction = np.random.default_rng(args.direction_seed).normal(size=args.P)
        direction -= direction.mean()  # An explicit easy ReLU disorder convention.
        direction /= np.linalg.norm(direction)
    target = KernelTailTarget.from_covariance(covariance, direction, width=args.width,
        layers=args.layers, activation=args.activation, device=args.device, precisions=1.)
    return target, covariance, direction


def log_bias(row, q, metadata):
    spec = row['tilt']
    tilt = PowerTilt(spec['kappa']) if spec['kind'] == 'PowerTilt' else RootTilt(spec['theta'])
    return tilt.log_weight(q, width=metadata['N'], layers=metadata['L'])


def plot_results(result, rate, output):
    """Plot measured estimates; no bias learning or vertical tail alignment here."""
    report, windows = result.report, result.report['windows']
    metadata, rows = report['target'], report['production']
    P, N, L = metadata['P'], metadata['N'], metadata['L']
    title = f"{metadata['activation'].upper()} | P={P}, N={N}, L={L} | adaptive bridge"
    q_arrays = [r.get_samples('Q') for r in result.production]
    centers, edges = rate['points'], rate['edges']
    lower, upper = report['lower'], report['upper']
    colors = plt.colormaps['viridis'](np.linspace(.05, .95, len(windows)))
    prefix = 'Both endpoint means reached' if report['coverage_complete'] else 'Partial attained coverage'

    # The common mode is chosen from the iid prior and held fixed in bootstrap.
    # Both curves use that same bin, rather than shifting each tail separately.
    fig, ax = plt.subplots(figsize=(7.5, 5.1))
    keep = (centers >= lower) & (centers <= upper) & np.isfinite(rate['mcse_shifted_rate'])
    ax.errorbar(centers[keep], rate['shifted_rate'][keep],
        yerr=2*rate['mcse_shifted_rate'][keep], fmt='o', ms=4, capsize=2,
        label='Bridge MBAR, histogram ±2 MCSE')
    unresolved = ((centers >= lower) & (centers <= upper)
                  & np.isfinite(rate['shifted_rate']) & ~np.isfinite(rate['mcse_shifted_rate']))
    if unresolved.any():
        ax.plot(centers[unresolved], rate['shifted_rate'][unresolved], 'o',
                mfc='none', color='C0', ms=4, label='MCSE unresolved')
    prior = estimate_histogram_rate(q_arrays[0].ravel(), edges, P=P)
    mode = rate['modal_index']
    hits = prior['hits_per_chain'][0]
    visible = (hits >= 20) & (centers >= lower) & (centers <= upper)
    prior_rate = -(prior['log_density']-prior['log_density'][mode])/P
    prior_se = np.sqrt(1/np.maximum(hits, 1)+1/hits[mode])/P
    prior_se[mode] = 0
    ax.errorbar(centers[visible], prior_rate[visible], yerr=2*prior_se[visible],
                fmt='s', ms=4, mfc='none', color='black', label='Independent prior, ≥20 hits/bin')
    q = np.linspace(lower, upper, 400)
    if L <= 2:
        # Use the same finite-N reference as sample_tilted_prior.py. Theory is
        # shifted to its own mode; empirical estimates keep their prior-bin anchor.
        mode = minimize_scalar(
            lambda log_q: -finite_ewa_log_density(np.exp(log_q), N, L),
            bounds=(-8, 1), method='bounded')
        theory = (-finite_ewa_log_density(q, N, L) - mode.fun) / P
        theory_label = 'Central EWA (finite N)'
    else:
        theory = central_ewa_rate(q, width=N, layers=L, P=P)
        theory_label = 'Central EWA (large N)'
    ax.plot(q, theory, '--', color='.5', label=theory_label)
    ax.set(xlim=(0, upper), xlabel=r'$Q$', ylabel=r'$I(Q)-I(Q_{\rm modal})$', title=title)
    ax.legend(fontsize=9)
    fig.text(.5, .01, prefix+'; '+r'$I=-\log\rho/P$'+'. Bin averages, no smoothing.\n'
             'Errors exclude finite-bin bias, missed modes and disorder variation.', ha='center', fontsize=9)
    save_figure(fig, output/'rate_function.png')

    # These densities retain their full-sample denominator, including draws
    # outside the displayed range. Bias curves are shifted only for display.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    plot_edges = np.linspace(0, upper*1.2, 70)
    grid = np.linspace(lower, upper, 250)
    for i, (sample, row, color) in enumerate(zip(q_arrays, windows, colors)):
        density = np.histogram(sample, plot_edges)[0]/(sample.size*np.diff(plot_edges))
        label = row['id'] if i % max(1, int(np.ceil(len(windows)/7))) == 0 else None
        axes[0, 0].stairs(density, plot_edges, color=color, alpha=.8, label=label)
        axes[0, 1].plot(grid, log_bias(row, grid, metadata)-log_bias(row, np.array(1.), metadata), color=color)
    positions = np.arange(len(windows))
    axes[1, 0].errorbar(positions, rate['log_normalizers'], yerr=2*rate['mcse_log_normalizers'],
                        fmt='o', label='Joint MBAR ±2 MCSE')
    axes[1, 0].plot(positions, [r['log_normalizer_bar'] for r in rows], 'x', label='Sequential BAR')
    axes[1, 1].plot(positions[1:], [r['overlap'] for r in rows[1:]], 'o')
    axes[1, 1].axhline(report['min_overlap'], color='.5', ls='--', label='Discovery overlap minimum')
    axes[0, 0].set(xlabel='Q', ylabel='Biased density', title='Frozen production windows')
    axes[0, 1].set(xlabel='Q', ylabel='b(Q) − b(1)', title='Power and root biases')
    axes[1, 0].set(xlabel='Window: prior, right, left', ylabel='log Z', title='Normalization anchored at the prior')
    axes[1, 1].set(xlabel='Window', ylabel='BAR overlap', title='Overlap is separate from chain mixing')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output/'windows_and_normalization.png')

    # Endpoint histories show retained rejections. The diagnostics below use
    # every window and the slowest coordinate across all signed layer projections.
    endpoints = [report['endpoints'][family]['last_window'] for family in ('root', 'power')]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for row_index, identifier in enumerate(endpoints):
        index = next(i for i, row in enumerate(windows) if row['id'] == identifier)
        samples, spec = result.production[index], windows[index]
        values = (samples.get_samples('Q'), -log_bias(spec, samples.get_samples('Q'), metadata),
                  samples.get_samples('signed_projection')[..., -1])
        for column, (value, name) in enumerate(zip(values, ('Q', 'Inverse log bias −b(Q)', 'Last-layer signed projection'))):
            for chain in range(value.shape[0]):
                axes[row_index, column].plot(value[chain], lw=.65, alpha=.8, label=f'Chain {chain+1}')
            axes[row_index, column].set(title=f'{identifier}: {name}', xlabel='Retained draw')
    for name, label in (('Q', 'Q'), ('signed_projection', 'Signed: all layers')):
        diag = [r['diagnostics']['observables'][name] for r in rows[1:]]
        axes[2, 0].plot(positions[1:], [d['min_bulk_ess'] for d in diag], 'o-', label=label)
        axes[2, 1].plot(positions[1:], [d['max_rhat'] for d in diag], 'o-', label=label)
    axes[2, 0].set(xlabel='Window', ylabel='Bulk ESS', yscale='log')
    axes[2, 1].set(xlabel='Window', ylabel='Rank/folded Rhat')
    axes[2, 1].axhline(1.05, color='.5', ls='--')
    axes[2, 2].plot(positions, [r.get_samples('Q').mean() for r in result.discovery], 'o', label='Discovery')
    axes[2, 2].plot(positions, [r.get_samples('Q').mean() for r in result.production], 'x', label='Reverse production')
    axes[2, 2].set(xlabel='Window', ylabel='Mean Q', title='Forward/reverse comparison')
    for ax in axes[2]:
        ax.legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output/'histories_and_diagnostics.png')


def save_figure(fig, filename):
    for ax in fig.axes:
        ax.grid(alpha=.18)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/bridge_demo'))
    parser.add_argument('--replot', type=Path, help='reload this completed run without sampling')
    parser.add_argument('--activation', choices=['relu', 'id', 'erf'], default='erf')
    parser.add_argument('--P', type=int, default=16)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lower', type=float, default=.2)
    parser.add_argument('--upper', type=float, default=2.5)
    parser.add_argument('--data', type=Path, help='prepared NPZ with covariance C and bar_f')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--draws', type=int, default=600)
    parser.add_argument('--discovery-draws', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=80)
    parser.add_argument('--prior-draws', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=81)
    parser.add_argument('--direction-seed', type=int, default=12)
    parser.add_argument('--bootstrap', type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(1)  # Example policy only; the library does not change threads.
    start = time.perf_counter()
    if args.replot:
        result = BridgeResult.load(args.replot)
        output = args.replot
    else:
        target, covariance, direction = prepare_target(args)
        config = SamplerConfig(sampler='hmc', chains=2, draws=args.draws, warmup=args.warmup,
            trajectory_length=1.1, target_accept=.9, seed=args.seed)
        result = sample_adaptive_bridge(target, lower=args.lower, upper=args.upper, config=config,
            discovery_config=replace(config, draws=args.discovery_draws), prior_draws=args.prior_draws,
            output_dir=args.output)
        output = args.output
        np.savez_compressed(output/'inputs.npz', C=covariance, bar_f=direction)
        write_json(output/'settings.json', vars(args))
    report = result.report
    # Narrow bins make the point approximation less misleading in steep tails;
    # the estimator itself always reports the exact weighted bin average.
    width = min(.1, 1/report['target']['N'])
    edges = np.arange(0, report['upper']+width, width)
    rate = result.estimate_rate(edges, bootstrap=args.bootstrap, seed=371)
    plot_results(result, rate, output)
    write_json(output/'rate.json', rate)
    summary_path = output/'example_summary.json'
    summary = json.loads(summary_path.read_text()) if args.replot and summary_path.exists() else {}
    summary['replot_seconds' if args.replot else 'total_seconds'] = time.perf_counter()-start
    summary.update(sampling_seconds=report['seconds'], windows=len(report['windows']),
        coverage_complete=report['coverage_complete'], endpoints=report['endpoints'],
        retained_production_draws=sum(r.get_samples('Q').size for r in result.production),
        figures=['rate_function.png', 'windows_and_normalization.png', 'histories_and_diagnostics.png'])
    write_json(summary_path, summary)
    print(f"Saved {output}; {len(report['windows'])} windows; {time.perf_counter()-start:.1f} seconds", flush=True)


if __name__ == '__main__':
    main()
