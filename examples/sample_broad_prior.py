"""Calibrate a two-sided umbrella, freeze it, and estimate the prior Q density.

Small demonstration (P=N=16, L=2, Erf/ReLU; a few minutes on the tested GPU):
    python examples/sample_broad_prior.py --output runs/umbrella_demo
    python examples/sample_broad_prior.py --device cuda --output runs/umbrella_gpu
Replot saved samples without sampling again:
    python examples/sample_broad_prior.py replot runs/umbrella_demo --output runs/umbrella_replot

Four figures show histogram rate estimates, bias/occupancy calibration,
retained histories, and diagnostics. EWA supplies an initial bias shape;
production samples supply normalization. Every density conditions on Q>0.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from deepbays.sampling import (UmbrellaTilt, KernelTailTarget, SamplerConfig,
    calibrate_umbrella, sample_tilted_prior, SamplingResult, diagnostics_and_mcse,
    estimate_histogram_rate, central_ewa_rate)
from sample_tilted_prior import draw_prior, write_json, finite_ewa_log_density


def ewa_flattening_bias(q, *, width, layers):
    """−log rho_EWA(Q)+log rho_EWA(1), without sampling guards or softplus.

    Use exact finite-N EWA at L=1,2 and its saddle density including the
    prefactor at L>2. This would flatten that reference density on a finite
    Q interval. It is a plotting reference, not a globally proper target.
    """
    q = np.asarray(q, dtype=float)
    if np.any(q <= 0) or not np.isfinite(q).all():
        raise ValueError('EWA flattening reference needs finite positive Q')
    if layers <= 2:
        return finite_ewa_log_density(1., width, layers)-finite_ewa_log_density(q, width, layers)
    # The saddle prefactor is Q^(-(3L-1)/(2L)); its log term matters at small N.
    return (central_ewa_rate(q, width=width, layers=layers, P=1)
            +(3*layers-1)/(2*layers)*np.log(q))


def prepare_inputs(args):
    """Use prepared data verbatim, or an explicit centered synthetic projection."""
    if args.data:
        with np.load(args.data) as data:
            inputs, direction = data['C' if args.covariance else 'X'], data['bar_f']
        if args.covariance:
            args.covariance_antisymmetry = float(np.max(np.abs(inputs-inputs.T)))
            inputs = (inputs+inputs.T)/2
        args.P = len(inputs)
        args.direction_convention = 'supplied fixed projection'
    else:
        inputs = np.sqrt(args.P)*np.eye(args.P)
        direction = np.random.default_rng(args.direction_seed).normal(size=args.P)
        if not args.uncentered_direction:
            direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        args.direction_convention = 'iid Gaussian' if args.uncentered_direction else 'centered Gaussian'
    return inputs, direction


def analyze(result, prior_q, bias, settings):
    """Use independent prior draws for a common modal reference; no curve alignment."""
    positive = prior_q[prior_q > 0]
    edges = np.arange(0, settings.upper+settings.bin_width*.5, settings.bin_width)
    if edges[-1] < settings.upper:
        edges = np.r_[edges, settings.upper]
    prior = estimate_histogram_rate(positive, edges, P=settings.P)
    mode = int(np.nanargmax(prior['log_density']))
    shift = float(prior['log_density'][mode]/settings.P)
    # Resample iid histogram counts to include uncertainty in selecting the mode.
    counts = prior['hits_per_chain'][0]
    outside = len(positive)-counts.sum()
    probabilities = np.r_[counts, outside]/len(positive)
    bootstrap = np.random.default_rng(settings.seed+91).multinomial(len(positive), probabilities, size=400)
    modal = np.max(bootstrap[:, :-1]/len(positive)/np.diff(edges), axis=1)
    shift_se = float(np.std(np.log(modal)/settings.P, ddof=1))
    prior['shifted_rate'] = prior['rate']+shift
    # For the baseline itself the common-mode covariance matters. Display errors
    # conditional on the selected modal bin; its own shifted value is exactly 0.
    baseline_se = np.full(len(counts), np.inf)
    occupied = counts > 0
    baseline_se[occupied] = np.sqrt(1/counts[occupied]+1/counts[mode])/settings.P
    baseline_se[mode] = 0.
    prior['mcse_shifted_rate'] = baseline_se
    rate = estimate_histogram_rate(result.get_samples('Q'), edges, P=settings.P,
        tilt=bias, width=settings.width, layers=settings.layers)
    rate['shifted_rate'] = rate['rate']+shift
    rate['mcse_shifted_rate'] = np.hypot(rate['mcse_rate'], shift_se)
    diagnostics, _ = diagnostics_and_mcse(
        {name: result.get_samples(name) for name in result.arrays}, chain_info=result.chain_info)
    return dict(bias=asdict(bias), target=result.metadata['posterior'], rate=rate,
        prior_rate=prior, modal_shift=shift, mcse_modal_shift=shift_se,
        diagnostics=diagnostics, chain_info=result.chain_info,
        zero_prior_draws=int(np.sum(prior_q == 0)), support='conditional on Q>0')


def plot(reports, settings, output, source):
    """Plot archived observations. No fit or sampling is performed here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(reports)
    ratefig, raxes = plt.subplots(1, len(names), figsize=(6*len(names), 4.8), squeeze=False)
    biasfig, baxes = plt.subplots(len(names), 3, figsize=(14, 3.5*len(names)), squeeze=False)
    tracefig, taxes = plt.subplots(len(names), 3, figsize=(14, 3.4*len(names)), squeeze=False)
    diagfig, daxes = plt.subplots(len(names), 2, figsize=(11, 3.5*len(names)), squeeze=False)
    for row, name in enumerate(names):
        report = reports[name]
        result = SamplingResult.load(source/f'{name}_production')
        q, log_bias = result.get_samples('Q'), result.get_samples('log_bias')
        prior, rate = report['prior_rate'], report['rate']
        ax = raxes[0, row]
        for estimate, color, label, iid in ((prior, 'black', 'Independent prior', True),
                (rate, 'C0', 'Frozen umbrella; sampled normalization', False)):
            x, y, se = (np.asarray(estimate[k], dtype=float) for k in
                        ('points', 'shifted_rate', 'mcse_shifted_rate'))
            counts = np.asarray(estimate['hits_per_chain']).sum(axis=0)
            mask = np.isfinite(y) & np.isfinite(se) & (counts >= (10 if iid else 1))
            ax.errorbar(x[mask], y[mask], yerr=2*se[mask], fmt='o', ms=3,
                        color=color, mfc='white' if not iid else color, capsize=1.5, label=label)
        x = np.geomspace(max(.03, settings.lower/3), settings.upper, 300)
        ax.plot(x, central_ewa_rate(x, width=settings.width, layers=settings.layers, P=settings.P),
                '--', color='.5', label='Leading EWA rate')
        ax.set(xlabel='Q', ylabel='Shifted density rate I(Q)', xlim=(0, settings.upper), title=name.upper())
        ax.legend(fontsize=8)
        grid = np.linspace(0, settings.upper+.6, 350)
        calibration = report.get('calibration')
        stages = calibration['stages'] if calibration else []
        histogram_edges = np.linspace(0, settings.upper+.6, 40)
        for epoch, stage in enumerate(stages):
            b = UmbrellaTilt(**stage['bias'])
            baxes[row, 0].plot(grid, b.log_weight(grid, width=settings.width, layers=settings.layers),
                              lw=1, alpha=.7, label=f'Pilot {epoch+1}')
            pq = np.load(source/f'{name}_calibration/pilot_{epoch:02d}/Q.npy')
            counts = np.histogram(pq, histogram_edges)[0]
            baxes[row, 1].stairs(counts/pq.size/np.diff(histogram_edges), histogram_edges,
                                alpha=.65, label=f'Pilot {epoch+1}')
        frozen = UmbrellaTilt(**report['bias'])
        baxes[row, 0].plot(grid, frozen.log_weight(grid, width=settings.width, layers=settings.layers),
                          color='black', lw=1.6, label='Frozen')
        # Compare the inverse EWA density on the requested interval, anchored
        # at Q=1 like the sampled bias. Do not clip it at zero or apply guards:
        # those would prevent exact flattening of the reference density.
        reference_grid = np.geomspace(settings.lower, settings.upper, 350)
        reference = ewa_flattening_bias(reference_grid, width=settings.width, layers=settings.layers)
        reference_kind = 'finite N' if settings.layers <= 2 else 'asymptotic'
        baxes[row, 0].plot(reference_grid, reference, '--', color='C4', lw=1.7,
                          label=f'Naive EWA ({reference_kind})')
        counts = np.histogram(q, histogram_edges)[0]
        baxes[row, 1].stairs(counts/q.size/np.diff(histogram_edges), histogram_edges,
                            color='black', lw=1.6, label='Production')
        for chain in range(q.shape[0]):
            chain_counts = np.histogram(q[chain], histogram_edges)[0]
            baxes[row, 2].stairs(chain_counts/q.shape[1]/np.diff(histogram_edges), histogram_edges,
                                label=f'Chain {chain+1}', alpha=.8)
            color = f'C{chain % 10}'
            taxes[row, 0].plot(q[chain], color=color, lw=.6, label=f'Chain {chain+1}')
            taxes[row, 1].plot(-log_bias[chain], color=color, lw=.6)
            taxes[row, 2].plot(result.get_samples('signed_projection')[chain, :, -1], color=color, lw=.6)
        for col, title in enumerate(('Log bias B(Q)', 'Biased occupancy during calibration', 'Frozen-target chain histograms')):
            baxes[row, col].set(title=f'{name.upper()} | {title}', xlabel='Q',
                               ylabel='Log weight' if col == 0 else 'Density')
            baxes[row, col].axvspan(settings.lower, settings.upper, color='gray', alpha=.05)
            baxes[row, col].legend(fontsize=7)
        for col, ylabel in enumerate(('Q', 'Inverse log weight −B(Q)', 'Signed last-layer projection')):
            taxes[row, col].set(xlabel='Retained draw', ylabel=ylabel, title=name.upper())
        taxes[row, 0].legend(fontsize=7, ncol=2)
        keys = ['Q', 'signed_projection', 'gaussian_energy', 'inverse_bias']
        labels = ['Q', 'Signed\nprojection', 'Gaussian\nenergy', 'Inverse\nweight']
        diagnostic = report['diagnostics']['observables']
        ess = [diagnostic[k]['min_bulk_ess'] or np.nan for k in keys]
        rhat = [diagnostic[k]['max_rhat'] or np.nan for k in keys]
        daxes[row, 0].bar(labels, ess)
        daxes[row, 0].set(ylabel='Minimum bulk ESS', title=f'{name.upper()} | worst layer where applicable')
        daxes[row, 1].plot(labels, rhat, 'o')
        daxes[row, 1].axhline(1.01, color='.5', ls=':')
        normalizer_se = rate['mcse_log_normalizer']
        if normalizer_se is None:  # Nonfinite errors are stored as JSON null.
            normalizer_se = np.nan
        daxes[row, 1].set(ylabel='Maximum rank/folded R-hat',
            title=f'log Z = {rate["log_normalizer"]:.3f} ± {normalizer_se:.3f}; '
                  f'weight Kish = {rate["weight_kish_ess"]:.0f}')
    ratefig.suptitle(f'P={settings.P}, N={settings.width}, L={settings.layers} | Q>0 | unsmoothed histogram points')
    ratefig.text(.5, .012, 'Bars: ±2 pointwise MCSE; common independent prior-modal reference. Finite-bin bias and missed modes excluded.',
                 ha='center', fontsize=9)
    biasfig.suptitle('Discarded fixed-bias pilots → frozen production | histogram mass outside the displayed range is retained in normalization')
    if any(report.get('calibration') for report in reports.values()):
        biasfig.text(.5, .012,
            f'Each pilot: 2 chains × {settings.pilot_draws} retained draws; '
            f'{settings.pilot_warmup} warmup/chain. Final pilot uses the frozen bias.',
            ha='center', fontsize=9)
    tracefig.suptitle(f'{settings.chains} chains × {settings.draws} retained draws; {settings.warmup} warmup/chain | all retained rejections shown')
    diagfig.suptitle('Diagnostics are reported without suppressing rate estimates; short runs may be inconclusive')
    for figure, filename in ((ratefig, 'rate_function.png'), (biasfig, 'bias_and_occupancy.png'),
                             (tracefig, 'sampling_histories.png'), (diagfig, 'diagnostics.png')):
        for ax in figure.axes:
            ax.grid(alpha=.18)
        figure.tight_layout(rect=(0, .04, 1, .95))
        figure.savefig(output/filename, dpi=150)
        plt.close(figure)


def run(args):
    """Prepare a prior baseline, learn/freeze the bias, then sample fresh chains."""
    inputs, direction = prepare_inputs(args)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'inputs.npz', bar_f=direction,
                        **{'C' if args.covariance else 'X': inputs})
    write_json(args.output/'settings.json', vars(args))
    constructor = KernelTailTarget.from_covariance if args.covariance else KernelTailTarget
    saved = json.loads(args.bias_file.read_text())['activations'] if args.bias_file else {}
    reports, start = {}, time.perf_counter()
    for activation in args.activations:
        parameters = dict(width=args.width, layers=args.layers, activation=activation,
                          precisions=1., device=args.device)
        prior_target = constructor(inputs, direction, **parameters)
        if saved:
            previous, current = saved[activation]['target'], prior_target.metadata()
            keys = ('P', 'N', 'L', 'activation', 'precisions', 'input_sha256', 'direction_sha256')
            mismatches = [key for key in keys if previous.get(key) != current[key]]
            if mismatches:
                raise ValueError('Saved bias belongs to a different target: '+', '.join(mismatches))
        prior_q = draw_prior(prior_target, args.prior_draws, args.seed+811)
        np.save(args.output/f'{activation}_prior.npy', prior_q)
        bias = (UmbrellaTilt(**saved[activation]['bias']) if saved else
                UmbrellaTilt(lower=args.lower, left_guard=args.lower/2, upper=args.upper,
                             left=args.tails != 'right', right=args.tails != 'left'))
        calibration = None
        if args.calibrations:
            pilot = SamplerConfig(sampler='hmc', chains=2, draws=args.pilot_draws,
                warmup=args.pilot_warmup, trajectory_length=args.trajectory_length,
                target_accept=.9, seed=args.seed)
            bias, calibration = calibrate_umbrella(prior_target, tilt=bias, prior_q=prior_q,
                rounds=args.calibrations, config=pilot, bulk_fraction=args.bulk_fraction,
                output_dir=args.output/f'{activation}_calibration')
            print(f'{activation}: frozen bias; pilot coverage={calibration["coverage_ok"]}; '
                  f'{calibration["elapsed_seconds"]:.1f}s calibration', flush=True)
        target = constructor(inputs, direction, tilt=bias, **parameters)
        config = SamplerConfig(sampler='hmc', chains=args.chains, draws=args.draws, warmup=args.warmup,
            trajectory_length=args.trajectory_length, target_accept=.9, seed=args.seed+50021)
        # Fresh independent Gaussian starts and fresh adaptation; no calibration
        # observation or weight contributes to the production rate estimate.
        result = sample_tilted_prior(target, config, output_dir=args.output/f'{activation}_production',
            on_chain_end=lambda _, chain: print(f'{activation}: production chain {chain+1} complete', flush=True))
        reports[activation] = analyze(result, prior_q, bias, args)
        reports[activation]['calibration'] = calibration
        write_json(args.output/'report.json', dict(settings=vars(args), activations=reports,
                                                  elapsed_seconds=time.perf_counter()-start))
        plot(reports, args, args.output, args.output)
        print(f'{activation}: Q diagnostics {reports[activation]["diagnostics"]["observables"]["Q"]}; '
              f'total {time.perf_counter()-start:.1f}s', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--P', type=int, default=16)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--data', type=Path, help='NPZ containing prepared X and bar_f (or C with --covariance)')
    parser.add_argument('--covariance', action='store_true')
    parser.add_argument('--activations', nargs='+', choices=['id','erf','relu'], default=['erf','relu'])
    parser.add_argument('--direction-seed', type=int, default=1234)
    parser.add_argument('--uncentered-direction', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--draws', type=int, default=300)
    parser.add_argument('--trajectory-length', type=float, default=1.5)
    parser.add_argument('--calibrations', type=int, default=3, help='fixed pilot budget; final pilot validates without refitting')
    parser.add_argument('--pilot-warmup', type=int, default=80)
    parser.add_argument('--pilot-draws', type=int, default=150)
    parser.add_argument('--prior-draws', type=int, default=3000)
    parser.add_argument('--lower', type=float, default=.2)
    parser.add_argument('--upper', type=float, default=5.)
    parser.add_argument('--tails', choices=['both', 'left', 'right'], default='both')
    parser.add_argument('--bulk-fraction', type=float, default=.3)
    parser.add_argument('--bin-width', type=float, default=.1)
    parser.add_argument('--bias-file', type=Path, help='saved report.json; --calibrations 0 reuses its frozen bias')
    args = parser.parse_args()
    if (args.output.exists() or args.P < 2 or args.width < 2 or args.layers < 1
            or args.chains < 2 or args.draws < 8 or args.warmup < 1 or args.calibrations < 0
            or args.pilot_draws < 16 or args.pilot_warmup < 1 or args.prior_draws < 100
            or not np.isfinite([args.lower,args.upper,args.bulk_fraction,args.bin_width,args.trajectory_length]).all()
            or not 0 < args.lower < .8 or args.upper <= 1.2 or not 0 < args.bulk_fraction < 1
            or args.bin_width <= 0 or args.trajectory_length <= 0
            or len(set(args.activations)) != len(args.activations)):
        parser.error('use a fresh output directory and valid positive sizes and support')
    if args.covariance and not args.data:
        parser.error('--covariance requires --data')
    if not args.calibrations and not args.bias_file:
        parser.error('--calibrations 0 requires a frozen --bias-file')
    if args.bias_file:
        saved = json.loads(args.bias_file.read_text())['activations']
        for activation in args.activations:
            b = saved[activation]['bias']
            if 'lower' not in b or b['lower'] != args.lower or b['upper'] != args.upper:
                parser.error('saved umbrella must match --lower/--upper')
            if b.get('left', True) != (args.tails != 'right') or b.get('right', True) != (args.tails != 'left'):
                parser.error('saved bias must match --tails')
    return args


def main():
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == 'replot':
        parser = argparse.ArgumentParser(description='Replot archived umbrella results; no sampling')
        parser.add_argument('source', type=Path)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        args.output.mkdir(parents=True, exist_ok=False)
        report = json.loads((args.source/'report.json').read_text())
        plot(report['activations'], argparse.Namespace(**report['settings']), args.output, args.source)
    else:
        run(parse_args())


if __name__ == '__main__':
    main()
