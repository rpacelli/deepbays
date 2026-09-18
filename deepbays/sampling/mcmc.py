"""Pyro HMC/NUTS orchestration; priors and losses belong in posteriors.py.

Pyro runs trajectories and adapts during warmup. StreamingMCMC(save_params=[])
prevents Pyro from retaining weight trajectories on the GPU. Our hook streams
only the requested values in bounded chunks to CPU arrays or memory-mapped files.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.metadata
import math
import random
import re
import time
import numpy as np
import torch
from .storage import SampleStore


@dataclass(frozen=True)
class SamplerConfig:
    """Sampler settings; draws excludes discarded warmup, for each chain.

    HMC uses a fixed trajectory_length, not a fixed leapfrog step count while
    adapting. NUTS stops trajectories adaptively up to max_tree_depth. Diagonal
    mass adaptation is the scalable default. Dense mass needs O(dimension²).
    chain_offset gives independent reproducible chain IDs across processes/GPUs.
    cudnn_benchmark autotunes deterministic convolutions for fixed input shapes.
    """
    sampler: str = 'nuts'
    chains: int = 4
    warmup: int = 1000
    draws: int = 2000
    seed: int = 0
    chain_offset: int = 0
    target_accept: float = .85
    initial_step: float = .1
    max_tree_depth: int = 7
    trajectory_length: float = 1.
    adapt_step_size: bool = True
    adapt_mass_matrix: bool = True
    dense_mass: bool = False
    progress: bool = False
    cudnn_benchmark: bool = False

    def __post_init__(self):
        if self.sampler not in ('hmc', 'nuts'):
            raise ValueError("sampler must be 'hmc' or 'nuts'")
        if not isinstance(self.cudnn_benchmark, bool):
            raise TypeError('cudnn_benchmark must be a bool')
        for name in ('chains', 'draws', 'max_tree_depth'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('warmup', 'seed', 'chain_offset'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('initial_step', 'trajectory_length'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.target_accept < 1:
            raise ValueError('target_accept must be in (0,1)')
        if self.seed + 104729*(self.chain_offset+self.chains-1) >= 2**32:
            raise ValueError('chain seeds must fit in 32 bits')


@contextmanager
def _random_state(seed, device):
    """Seed a chain without changing the caller's Python/NumPy/Torch RNG state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [] if device.type == 'cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Unlike torch.manual_seed, this does not reseed unrelated GPUs.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@contextmanager
def _cuda_precision(device, *, cudnn_benchmark=False):
    """Use full precision and deterministic convolutions, with optional autotuning."""
    if device.type != 'cuda':
        yield
        return
    matmul, conv = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = cudnn_benchmark
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = conv
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def sample_posterior(posterior, config=None, *, observables=None, save_weights=True,
                     output_dir=None, buffer_size=16, initial=None, on_chunk=None,
                     on_chain_end=None, run_metadata=None):
    """Run independent sequential chains, returning a SamplingResult.

    posterior supplies potential(theta), dimension, device and dtype. Observables
    maps names to deterministic tensor-valued functions of theta (e.g. logits on
    a held-out batch). Set save_weights=False to retain only these observables.
    Default output is preallocated CPU memory. output_dir instead uses .npy memory
    maps and an atomic progress manifest, refusing to overwrite an existing run.

    Only buffer_size draws of selected tensors reside on the sampling device.
    on_chunk(chain_index, first_draw, cpu_arrays) runs after a committed write;
    on_chain_end(result, chain_index) can checkpoint downstream analyses.
    initial is optional shape (chains, dimension), in *whitened* coordinates.
    Callbacks/observables must not mutate the target or sampling coordinates.
    """
    try:
        from pyro.infer.mcmc import HMC, NUTS, StreamingMCMC
    except ImportError as exc:
        raise ImportError('Install optional sampling dependencies: pip install "deepbays[sampling]"') from exc
    config = SamplerConfig() if config is None else config
    if not isinstance(config, SamplerConfig):
        raise TypeError('config must be a SamplerConfig')
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 1:
        raise ValueError('buffer_size must be a positive integer')
    observables = {} if observables is None else dict(observables)
    if not save_weights and not observables:
        raise ValueError('save weights or supply at least one observable')
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)
           or name == 'theta' for name in observables):
        raise ValueError('observable names must be simple identifiers other than theta')
    device, dtype = torch.device(posterior.device), posterior.dtype
    if initial is not None:
        initial = torch.as_tensor(initial, device=device, dtype=dtype).detach().clone()
        if initial.shape != (config.chains, posterior.dimension) or not torch.isfinite(initial).all():
            raise ValueError('initial must be finite with shape (chains, dimension)')
    metadata = dict(schema_version=1, status='running', config=asdict(config), device=str(device),
                    dtype=str(dtype), posterior=posterior.metadata(), buffer_size=buffer_size,
                    versions={p:importlib.metadata.version(p) for p in ('torch','numpy','pyro-ppl')},
                    arrays={}, completed_draws=[0]*config.chains,
                    chains=[dict(chain=config.chain_offset+i, status='pending') for i in range(config.chains)],
                    user={} if run_metadata is None else run_metadata)
    store = SampleStore(metadata, output_dir)
    with _cuda_precision(device, cudnn_benchmark=config.cudnn_benchmark):
        for chain in range(config.chains):
            chain_id = config.chain_offset+chain
            seed = config.seed+104729*chain_id
            row = metadata['chains'][chain]
            row.update(seed=seed, status='running')
            buffers = {}
            warmup_end = None
            evaluations = 0
            start = time.perf_counter()
            kernel = None

            def flush():
                if not buffers or not len(next(iter(buffers.values()))):
                    return
                values = {name:torch.stack(tensors).cpu().numpy() for name,tensors in buffers.items()}
                for tensors in buffers.values():
                    tensors.clear()
                first = store.append(chain, values)
                if on_chunk is not None:
                    on_chunk(chain_id, first, values)

            def update_diagnostics():
                diag = kernel.diagnostics()
                row.update(step_size=float(kernel.step_size), acceptance=float(diag['acceptance rate']),
                           divergences=len(diag['divergences']), divergence_indices=list(diag['divergences']))

            def hook(current_kernel, params, stage, index):
                nonlocal warmup_end
                if stage.startswith('Warmup'):
                    if index == config.warmup-1:
                        warmup_end = time.perf_counter()
                    return
                with torch.no_grad():
                    theta = params['theta']
                    values = {name:fn(theta) for name,fn in observables.items()}
                    if save_weights:
                        values['theta'] = theta
                    for name, value in values.items():
                        if not torch.is_tensor(value):
                            raise TypeError(f'observable {name} must return a torch tensor')
                        buffers.setdefault(name, []).append(value.detach().clone())
                if (index+1) % buffer_size == 0 or index+1 == config.draws:
                    flush()
                if index+1 == config.draws:
                    update_diagnostics()

            def pyro_potential(params):
                nonlocal evaluations
                evaluations += 1
                return posterior.potential(params['theta'])

            try:
                with _random_state(seed, device), torch.enable_grad():
                    theta = (torch.randn(posterior.dimension, device=device, dtype=dtype)
                             if initial is None else initial[chain].clone())
                    # Fail on an invalid initial potential/gradient before adapting.
                    probe = theta.detach().requires_grad_(True)
                    energy = posterior.potential(probe)
                    if energy.ndim != 0 or not torch.isfinite(energy):
                        raise ValueError('initial potential must be a finite scalar')
                    grad, = torch.autograd.grad(energy, probe)
                    if not torch.isfinite(grad).all():
                        raise ValueError('initial potential gradient is not finite')
                    del probe, energy, grad
                    settings = dict(potential_fn=pyro_potential, step_size=config.initial_step,
                                    target_accept_prob=config.target_accept,
                                    adapt_step_size=config.adapt_step_size,
                                    adapt_mass_matrix=config.adapt_mass_matrix, full_mass=config.dense_mass)
                    kernel = (NUTS(max_tree_depth=config.max_tree_depth, **settings) if config.sampler == 'nuts'
                              else HMC(trajectory_length=config.trajectory_length, **settings))
                    setup_end = time.perf_counter()
                    run = StreamingMCMC(kernel, num_samples=config.draws, warmup_steps=config.warmup,
                                        initial_params={'theta':theta}, num_chains=1, save_params=[],
                                        transforms={}, hook_fn=hook, disable_progbar=not config.progress)
                    run.run()
                    flush()
                    # StreamingMCMC must not have accumulated coordinate statistics.
                    if run.get_statistics():
                        raise RuntimeError('unexpected latent retention by Pyro StreamingMCMC')
                    end = time.perf_counter()
                    warmup_end = setup_end if warmup_end is None else warmup_end
                    row.update(status='complete', seconds=end-start, setup_seconds=setup_end-start,
                               warmup_seconds=warmup_end-setup_end, sampling_seconds=end-warmup_end,
                               potential_evaluations=evaluations)
                    store.checkpoint()
                    if on_chain_end is not None:
                        on_chain_end(store.result, chain_id)
                    del run, kernel, theta
            except BaseException as exc:
                # Preserve already committed data, including KeyboardInterrupt.
                # A failed observable/store/callback must not trigger its callback twice.
                row.update(status='interrupted', seconds=time.perf_counter()-start,
                           potential_evaluations=evaluations, error=f'{type(exc).__name__}: {exc}')
                metadata['status'] = 'interrupted'
                store.checkpoint()
                raise
    metadata['status'] = 'complete'
    store.checkpoint()
    return store.result
