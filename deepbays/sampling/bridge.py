"""Adaptive overlapping power/root windows using the shared Pyro sampler.

Discovery selects a ladder. Fresh production traverses that frozen ladder in
reverse; only production enters the prior-anchored MBAR estimate. No EWA
normalizer or new transition kernel is used. All distributions condition on Q>0.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostics import diagnostics_and_mcse
from .mcmc import SamplerConfig
from .rate_functions import (_bridge_bar, _bridge_ess_fraction, _bridge_increment,
                             estimate_bridge_rate)
from .storage import SampleStore, SamplingResult, _write_json
from .tilted_prior import KernelTailTarget, PowerTilt, RootTilt, _integer, sample_tilted_prior


@dataclass
class BridgeResult:
    """Completed discovery and production windows, in prior/right/left order.

    Entries are ordinary SamplingResult objects with Q and every layer observable.
    ``report`` contains the schedule, attained endpoints, overlap and diagnostics.
    ``estimate_rate`` uses only production. A completed run may have partial
    coverage when a finite discovery limit was reached; inspect the report.
    """
    discovery: list
    production: list
    report: dict

    def estimate_rate(self, edges, *, bootstrap=80, block_lengths=None, seed=1234):
        """Unsmoothed prior density, measured normalizers and joint MC errors."""
        if not self.report['production_complete']:
            raise ValueError('complete frozen-ladder production is required')
        tilts = [_tilt(row['tilt']) for row in self.report['windows']]
        metadata = self.report['target']
        return estimate_bridge_rate([r.get_samples('Q') for r in self.production],
            tilts, edges, width=metadata['N'], layers=metadata['L'], P=metadata['P'],
            bootstrap=bootstrap, block_lengths=block_lengths, seed=seed,
            chain_groups=[row['family'] for row in self.report['windows']])

    @classmethod
    def load(cls, directory):
        """Reload a completed run for analysis without further sampling.

        Interrupted native windows remain inspectable with SamplingResult.load;
        this method does not resume HMC or silently substitute discovery samples.
        """
        directory = Path(directory)
        report = json.loads((directory/'bridge.json').read_text())
        if report.get('schema_version') != 1 or not report.get('production_complete'):
            raise ValueError('a completed bridge production run is required')
        def load(rows):
            if len(rows) != len(report['windows']):
                raise ValueError('window count differs from the frozen schedule')
            results = []
            for row, spec in zip(rows, report['windows']):
                path = Path(row['directory'])
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('invalid window path')
                result = SamplingResult.load(directory/path)
                config = result.metadata['config']
                if (result.metadata['status'] != 'complete'
                        or result.completed_draws != (config['draws'],)*config['chains']):
                    raise ValueError('incomplete native window')
                actual = result.metadata['posterior']
                keys = ('P', 'N', 'L', 'activation', 'precisions', 'denominator',
                        'input_sha256', 'direction_sha256', 'support')
                if (row['id'] != spec['id'] or row['tilt'] != spec['tilt']
                        or actual['tilt'] != spec['tilt']
                        or any(actual[key] != report['target'][key] for key in keys)):
                    raise ValueError('window target differs from the frozen bridge schedule')
                results.append(result)
            return results
        return cls(load(report['windows']), load(report['production']), report)


def _tilt(spec):
    if spec['kind'] == 'PowerTilt':
        return PowerTilt(spec['kappa'])
    if spec['kind'] == 'RootTilt':
        return RootTilt(spec['theta'])
    raise ValueError('unsupported bridge bias')


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _config_seed(config, phase, index):
    # Independent streams without touching global RNGs, including chain offsets.
    seed = np.random.SeedSequence([config.seed, phase, index, 0xB71D]).generate_state(1)[0]
    maximum = 2**32-104729*(config.chain_offset+config.chains-1)
    return replace(config, seed=int(seed) % maximum)


def _reference(target, config, folder, buffer_size):
    """Exact positive-prior draws, in the existing streaming storage format."""
    metadata = dict(schema_version=1, status='running',
        config=dict(asdict(config), sampler='iid_gaussian', warmup=0),
        device=str(target.device), dtype=str(target.dtype), posterior=target.metadata(),
        arrays={}, completed_draws=[0]*config.chains, chains=[], independent=True)
    store = SampleStore(metadata, folder)
    states, started = [], time.perf_counter()
    with torch.no_grad():
        for chain in range(config.chains):
            seed = int(np.random.SeedSequence([config.seed, config.chain_offset+chain])
                       .generate_state(1, dtype=np.uint64)[0])
            generator = torch.Generator(device=target.device).manual_seed(seed)
            info = dict(chain=config.chain_offset+chain, seed=seed, status='running',
                        proposals=0, zero_rejections=0)
            metadata['chains'].append(info)
            count, batch = 0, []
            while count < config.draws:
                if info['proposals'] >= 1000*config.draws:
                    raise RuntimeError('too few positive prior draws')
                z = torch.randn(target.dimension, generator=generator,
                                device=target.device, dtype=target.dtype)
                values = target.observables(z)
                q = float(values['Q'])
                if not np.isfinite(q) or q < 0:
                    raise RuntimeError('invalid prior Q')
                info['proposals'] += 1
                if q == 0:
                    info['zero_rejections'] += 1
                    continue
                batch.append(values)
                count += 1
                if len(batch) == buffer_size or count == config.draws:
                    store.append(chain, {key: torch.stack([v[key] for v in batch]).cpu().numpy()
                                         for key in values})
                    batch = []
            states.append(z.cpu())
            info['status'] = 'complete'
            store.checkpoint()
    metadata['status'] = 'complete'
    metadata['seconds'] = time.perf_counter()-started
    store.checkpoint()
    return store.result, torch.stack(states)


def _window(target, tilt, config, folder, initial, buffer_size):
    started, states = time.perf_counter(), {}
    result = sample_tilted_prior(target.with_tilt(tilt), config, initial=initial,
        output_dir=folder, buffer_size=buffer_size,
        on_chain_state=lambda chain, z: states.__setitem__(chain, z))
    q = result.get_samples('Q')
    if np.any(q <= 0):
        raise RuntimeError('bridge draw violated the common Q>0 support')
    diagnostics, mcse = diagnostics_and_mcse(
        {name: result.get_samples(name) for name in ('Q', 'signed_projection', 'gaussian_energy')},
        chain_info=result.chain_info, min_chains=2, min_ess=100, max_rhat=1.05)
    terminal = torch.stack([states[config.chain_offset+i] for i in range(config.chains)])
    if folder is not None:
        np.save(folder/'terminal_states.npy', terminal.numpy(), allow_pickle=False)
    return result, terminal, dict(seconds=time.perf_counter()-started, mean_Q=float(q.mean()),
        diagnostics=diagnostics, mcse=mcse, config=asdict(config))


def _bar(previous, current, old_tilt, new_tilt, target):
    def delta(q):
        return (new_tilt.log_weight(q, width=target.width, layers=target.layers)
                -old_tilt.log_weight(q, width=target.width, layers=target.layers))
    return _bridge_bar(delta(previous.get_samples('Q')), delta(current.get_samples('Q')))


def sample_adaptive_bridge(target, *, lower=.2, upper=5., config=None,
                           discovery_config=None, prior_draws=2000, step_ess=.65,
                           min_overlap=.2, min_ess=50., max_rhat=1.15,
                           max_windows=32, max_power=None, max_root=20.,
                           output_dir=None, buffer_size=64):
    """Discover overlapping tilts, freeze the ladder, and sample it in reverse.

    Right windows use increasing PowerTilt(kappa); left windows use increasingly
    negative RootTilt(theta). Set lower or upper to None for one tail. The
    empirical importance ESS selects each increment; actual bidirectional BAR
    overlap and modest Q mixing checks validate it. Signed/energy diagnostics
    are reported without excluding a window solely for their slow mixing.

    Defaults: two HMC chains, 250 discovery / 600 production draws and 150 warmup
    per window. ``config`` controls production; ``discovery_config`` defaults to
    the same settings with 250 draws. They must share chain count/offset. Every
    window adapts afresh with distinct random numbers; only coordinates carry
    over. Discovery is excluded from the final estimator. Production inherits
    endpoint coordinates, so its warmup still must remove initialization effects.

    Discovery has at most three attempts per step. Poor Q mixing retries with
    twice the initial pilot draws; poor overlap halves the step. Failure or a
    parameter/window cap freezes a partial ladder, explicitly recording attained
    support, and production still completes that ladder. Production diagnostics
    are descriptive, not adaptive stopping rules. This is bounded orchestration,
    not an automatic guarantee of convergence or endpoint coverage.

    The target is copied with positive support; the caller is unchanged. Exact
    iid Gaussian draws conditioned on Q>0 anchor Z=1. Thus results exclude any
    ReLU atom at zero. Existing native storage streams observables and saves one
    terminal state per window, not weight trajectories. A new output_dir records
    progress in bridge.json; interrupted chains are inspectable but not resumable.
    """
    if not isinstance(target, KernelTailTarget):
        raise TypeError('target must be KernelTailTarget')
    if (lower is None and upper is None or lower is not None and not 0 < lower < 1
            or upper is not None and (not np.isfinite(upper) or upper <= 1)):
        raise ValueError('enable at least one tail, with 0<lower<1<upper')
    config = (SamplerConfig(sampler='hmc', chains=2, draws=600, warmup=150,
              trajectory_length=1.1, target_accept=.9) if config is None else config)
    if not isinstance(config, SamplerConfig):
        raise TypeError('config must be SamplerConfig')
    pilot = replace(config, draws=250) if discovery_config is None else discovery_config
    if not isinstance(pilot, SamplerConfig):
        raise TypeError('discovery_config must be SamplerConfig')
    if any(c.chains < 2 or c.draws < 16 or c.warmup < 1 for c in (config, pilot)):
        raise ValueError('bridge sampling needs >=2 chains, >=16 draws and positive warmup')
    if (config.chains, config.chain_offset) != (pilot.chains, pilot.chain_offset):
        raise ValueError('discovery and production must share chain count and offset')
    prior_draws, max_windows = _integer(prior_draws, 'prior_draws'), _integer(max_windows, 'max_windows')
    buffer_size = _integer(buffer_size, 'buffer_size')
    max_power = 6.*target.width if max_power is None else max_power
    if (not 0 < step_ess < 1 or not 0 < min_overlap < 1 or not np.isfinite(min_ess)
            or min_ess <= 0 or not np.isfinite(max_rhat) or max_rhat <= 1
            or not np.isfinite(max_power) or max_power <= 0
            or not np.isfinite(max_root) or max_root <= 0):
        raise ValueError('invalid overlap, mixing or parameter limits')
    target = target.with_tilt(PowerTilt(0), positive_only=True)
    root = None if output_dir is None else Path(output_dir)
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
    def path(name):
        return None if root is None else root/name
    started = time.perf_counter()
    report = dict(schema_version=1, status='discovery', production_complete=False,
        target=target.metadata(), support='conditional on Q>0', lower=lower, upper=upper,
        config=asdict(config), discovery_config=asdict(pilot), step_ess=step_ess,
        min_overlap=min_overlap, min_ess=min_ess, max_rhat=max_rhat,
        max_windows=max_windows, max_power=max_power, max_root=max_root,
        prior_draws=prior_draws, windows=[], attempts=[], production=[], endpoints={})
    def checkpoint():
        report['seconds'] = time.perf_counter()-started
        if root is not None:
            _write_json(root/'bridge.json', _json_value(report))
    checkpoint()
    discovery, production = [], []
    try:
        prior, prior_states = _reference(target, replace(_config_seed(pilot, 0, 0), draws=prior_draws),
                                         path('discovery/prior'), buffer_size)
        discovery.append(prior)
        prior_row = dict(id='prior', family='prior', parent=None, tilt=dict(kind='PowerTilt', kappa=0.),
                         directory='discovery/prior', mean_Q=float(prior.get_samples('Q').mean()),
                         log_normalizer_bar=0., overlap=1., seconds=prior.metadata['seconds'])
        report['windows'].append(prior_row)
        checkpoint()
        counter, endpoint_states = 0, {}
        for family, goal in (('power', upper), ('root', lower)):
            if goal is None:
                continue
            previous, initial, old_tilt, parameter = prior, prior_states, PowerTilt(0), 0.
            parent, cumulative, reason = 'prior', 0., 'window limit'
            for index in range(max_windows):
                mean = float(previous.get_samples('Q').mean())
                if (mean >= goal if family == 'power' else mean <= goal):
                    reason = 'endpoint reached'
                    break
                score = (np.log(previous.get_samples('Q')) if family == 'power' else
                         -target.width*previous.get_samples('Q')**(1/target.layers))
                remaining = (max_power if family == 'power' else max_root)-parameter
                if remaining < 1e-10 or np.std(score) < 1e-10:
                    reason = 'parameter limit or constant Q'
                    break
                step = _bridge_increment(score, remaining, step_ess)
                draws, accepted = pilot.draws, False
                identifier = f'{family}_{index+1:03d}'
                for attempt in range(3):
                    bias = PowerTilt(parameter+step) if family == 'power' else RootTilt(-parameter-step)
                    directory = f'discovery/{identifier}_attempt_{attempt}'
                    current, terminal, row = _window(target, bias,
                        replace(_config_seed(pilot, 1, counter), draws=draws), path(directory), initial, buffer_size)
                    counter += 1
                    ratio, overlap = _bar(previous, current, old_tilt, bias, target)
                    diag = row['diagnostics']['observables']['Q']
                    mixed = ((diag['min_bulk_ess'] or 0) >= min_ess
                             and (diag['max_rhat'] or np.inf) <= max_rhat
                             and diag['constant_coordinates'] == 0)
                    accepted = mixed and overlap >= min_overlap
                    row.update(id=identifier, family=family, parent=parent, directory=directory,
                        tilt=dict(kind=type(bias).__name__, **asdict(bias)), accepted=accepted,
                        overlap=overlap, incremental_ess=_bridge_ess_fraction(score, step),
                        log_normalizer_bar=cumulative+ratio)
                    report['attempts'].append(row)
                    checkpoint()
                    if accepted:
                        break
                    if overlap < min_overlap:
                        step *= .5
                    if not mixed:
                        draws = 2*pilot.draws
                if not accepted:
                    reason = 'mixing or overlap attempts exhausted'
                    break
                parameter += step
                cumulative += ratio
                parent, old_tilt, previous, initial = identifier, bias, current, terminal
                report['windows'].append(row)
                discovery.append(current)
                checkpoint()
            mean = float(previous.get_samples('Q').mean())
            reached = mean >= goal if family == 'power' else mean <= goal
            report['endpoints'][family] = dict(goal=goal, mean_Q=mean, reached=bool(reached),
                last_window=parent, reason='endpoint reached' if reached else reason)
            endpoint_states[family] = initial
            checkpoint()
        report['coverage_complete'] = all(row['reached'] for row in report['endpoints'].values())
        report['status'] = 'production'
        checkpoint()
        reference, _ = _reference(target, replace(_config_seed(config, 2, 0), draws=prior_draws),
                                   path('production/prior'), buffer_size)
        results = {'prior': reference}
        rows = {'prior': dict(prior_row, directory='production/prior',
                             mean_Q=float(reference.get_samples('Q').mean()), seconds=reference.metadata['seconds'])}
        report['production'].append(rows['prior'])
        for family in ('power', 'root'):
            indices = [i for i, row in enumerate(report['windows']) if row['family'] == family]
            if not indices:
                continue
            # Reconstructing initial coordinates from the discovery result would
            # require full weight histories. Retain only its terminal states.
            initial = endpoint_states[family]
            for index in reversed(indices):
                spec = report['windows'][index]
                directory = f'production/{spec["id"]}'
                current, initial, row = _window(target, _tilt(spec['tilt']),
                    _config_seed(config, 3, index), path(directory), initial, buffer_size)
                row.update(id=spec['id'], family=family, parent=spec['parent'],
                           tilt=spec['tilt'], directory=directory)
                results[spec['id']], rows[spec['id']] = current, row
                report['production'].append(row)
                checkpoint()
        for spec in report['windows'][1:]:
            parent = next(row for row in report['windows'] if row['id'] == spec['parent'])
            ratio, overlap = _bar(results[parent['id']], results[spec['id']],
                                  _tilt(parent['tilt']), _tilt(spec['tilt']), target)
            rows[spec['id']].update(overlap=overlap,
                log_normalizer_bar=rows[parent['id']]['log_normalizer_bar']+ratio)
        production = [results[row['id']] for row in report['windows']]
        report['production'] = [rows[row['id']] for row in report['windows']]
        report.update(status='complete', production_complete=True)
        checkpoint()
    except BaseException as error:
        report.update(status='interrupted or failed', error=f'{type(error).__name__}: {error}')
        checkpoint()
        raise
    return BridgeResult(discovery, production, _json_value(report))
