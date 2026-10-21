# Pyro posterior sampling

`deepbays.sampling` provides HMC and NUTS for the existing FC/CNN networks, other deterministic floating-input PyTorch modules, and fixed-kernel GP controls. It is an additive API: the previous optimizers remain available. All weights, including convolution weights and optional biases, are sampled. Posterior definitions, trajectory sampling, storage, predictive utilities, and diagnostics are separate modules.

The same engine supports [tilted FC priors and kernel-projection rate functions](tilted_prior_sampling.md).
See the [sampling machinery review](sampling_review.md) for the shared pipeline
and proposed follow-ups. The extension uses ordinary Pyro HMC.

## Installation

In an environment with your chosen CPU or CUDA PyTorch/torchvision build
(PyTorch >= 2.0), use the [general installation flow](installation.md):

```bash
cd ~/deepbays
python -m pip install --no-deps -e .
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

For only the sampling/diagnostics additions, install the core and then run
`python -m pip install -e '.[sampling]'`. The complete optional file also
installs plotting, benchmark, structured-kernel, and test dependencies.
The tested CPU environment was `~/.venvs/deepbays-classification` (Python 3.12,
torch 2.8.0+cpu, Pyro 1.9.1, ArviZ 0.22.0). The general requirements contain no
CPU/CUDA build pins or indexes.

Pyro and ArviZ are imported lazily. Existing deepbays users do not need these dependencies unless they sample or compute the new diagnostics.

## A small complete example

```python
import numpy as np
from deepbays.NN import FCNet
from deepbays.sampling import (
    NetworkPosterior, SamplerConfig, sample_posterior,
    classification_statistics, diagnostics_and_mcse, logit_contrasts,
)

rng = np.random.default_rng(12)
X = rng.normal(size=(20, 8))
y = (X[:, 0] > 0).astype(np.int64)
Z = rng.normal(size=(10, 8))
z = (Z[:, 0] > 0).astype(np.int64)
network = FCNet(8, 16, 1, act='erf', D=2).Sequential()

posterior = NetworkPosterior(
    network, X, y, task='classification', temperature=.2,
    priors=1., device='cpu',
)
config = SamplerConfig(sampler='nuts', chains=4, warmup=200, draws=500, seed=12)
result = sample_posterior(
    posterior, config,
    observables={'logits': lambda theta: posterior.predict(theta, Z)},
    save_weights=False, output_dir='runs/my_posterior',
)
logits = result.get_samples('logits')  # (chains, draws, test examples, classes)
summary = classification_statistics(logits, z)
print(summary['mean_predictor_accuracy'], summary['posterior_sample_accuracy'])
contrast = logit_contrasts(logits)
diagnostics, mcse = diagnostics_and_mcse(
    {'contrast': contrast, 'accuracy': summary['accuracy_draws']},
    chain_info=result.chain_info,
)
print(diagnostics['passed'])  # A short example is not guaranteed to pass.
```

There is also a standalone example, requiring no theory/kernel calculation:

```bash
python examples/sample_posterior.py --output runs/small_pyro
python examples/sample_posterior.py --network cnn --sampler hmc \
  --warmup 300 --draws 1000 --output runs/small_cnn_pyro
```

Supply `--data data.npz` for your data. Required fields are `X`, `y`; optional fields `Z`, `z` hold test inputs and targets. FC inputs are `(P,input_dimension)`, CNN inputs `(P,channels,height,width)`. `--outputs` must match the number of classes or regression outputs. A synthetic dataset is generated if `--data` is omitted. Input normalization is the caller's responsibility; the example archives the arrays actually used. It refuses to overwrite a nonempty output directory.

## Posterior conventions

Whitened coordinates have a standard-normal prior. A physical parameter in a layer of precision lambda is `theta / sqrt(lambda)`. The sampled potential is

\[
 U(\theta)=\frac12\|\theta\|^2+\frac{L(f_\theta(X),y)}{T}.
\]

- Regression: `L = 0.5 * sum((prediction - y)**2)`, so T is the observation variance.
- Classification: `L = sum(cross_entropy(raw_logits, integer_labels))`, so beta=1/T. Beta=1 is the ordinary categorical likelihood. Beta!=1 defines a tempered Gibbs posterior.
- The Gaussian weight prior is **not** divided by T. The likelihood is summed over all training examples and outputs; there is no implicit batch mean.
- `priors` can be a scalar, an exact mapping from named parameters to precisions, or one precision per named parameter tensor in `named_parameters()` order. For a bias-free L-hidden-layer deepbays network, the L+1 layer precisions match this order. For biases or a custom architecture, use a mapping.
- The module's fan-in, gamma and readout scaling are retained. Sampling does not add another width scaling. The original module is not mutated: a private copy runs in eval mode with fixed buffers. All named parameters are sampled, even if their original `requires_grad` flag was false.
- Models must be deterministic in eval mode. Dropout is disabled by eval mode. Custom stochastic operations, changing batch statistics, stochastic data augmentation or changing the training data during sampling invalidate the potential and are unsupported.

For regression use `task='regression'` with targets `(P,D)`; scalar-output targets `(P,)` are also accepted. Classification expects `(P,)` integer labels in `[0,C)`, with **C raw logits**. One-hot targets and one-logit BCE are not the API convention.

`WhitenedNetwork.parameters(theta)` reconstructs physical named parameter tensors. `NetworkPosterior.predict(theta, X, batch_size=...)` evaluates one network draw. For a custom target, `PotentialPosterior(callable, dimension, ...)` accepts a scalar potential of an unconstrained flat theta; the callable must contain its entire prior and likelihood.

Parameter reconstruction splits theta once, skips scaling for unit precisions, and shares the reconstructed weights across the chunks of one evaluation. This keeps whitened coordinates while avoiding repeated full-vector gradient accumulation and repeated weight scaling. Each new theta is reconstructed independently.

## HMC, NUTS, and initialization

`SamplerConfig(sampler='hmc')` selects fixed-trajectory HMC; `'nuts'` selects NUTS. Both use Pyro's Metropolis-corrected kernels. Step size and a diagonal mass matrix adapt during warmup and are then fixed. `warmup` draws are discarded; `draws` is the number of retained draws **per chain**. There is no thinning.

Useful controls:

| Setting | Meaning |
|---|---|
| `target_accept=.85` | Dual-averaging target; increasing it generally reduces step size. |
| `initial_step=.1` | Initial step size before warmup adaptation. |
| `trajectory_length=1.` | HMC integration time; the number of leapfrog steps changes as the step size adapts. |
| `max_tree_depth=7` | NUTS tree-depth limit; raising it may improve exploration at substantially greater cost. |
| `adapt_mass_matrix=True` | Adapt a diagonal mass matrix by default. |
| `dense_mass=False` | Full dense mass is optional and costs O(parameter_count²); use it only for small targets. |
| `chain_offset=0` | Assign distinct chain IDs to independently launched processes. |
| `progress=False` | Opt into Pyro progress bars when desired. |
| `cudnn_benchmark=False` | Opt into autotuning deterministic CUDA convolution algorithms for fixed shapes. |

Default initialization is an independent N(0,I) draw for each chain. Supply `initial` of shape `(chains, parameter_count)` to override it, in whitened coordinates. The sampler checks the initial potential and gradient for finiteness before adaptation. Independent random starts are preferable for convergence assessment; identical fitted initial states can hide poor exploration.

Chain ID `j` uses seed `config.seed + 104729*j`. The caller's Python, NumPy, and relevant Torch RNG states are restored afterward. `chain_offset` allows the same chain to be run separately without changing its seed. The regression tests compare both HMC and NUTS draws directly to native Pyro MCMC under matched initialization and seeds.

## GPU memory and throughput

```python
import torch
posterior = NetworkPosterior(
    network, X, y, task='classification', temperature=.2,
    priors=.5, device='cuda:0', dtype=torch.float64,
    checkpoint_batches=False,
)
# Prepare once, rather than transferring the same inputs for every draw.
Z_device = torch.as_tensor(Z, device=posterior.device, dtype=posterior.dtype)
# Compile once for a long run; omit compilation for quick checks.
posterior.compile()
predict = torch.compile(lambda theta: posterior.predict(theta, Z_device))
result = sample_posterior(
    posterior, SamplerConfig(sampler='hmc', warmup=1000, draws=2000, cudnn_benchmark=True),
    observables={'logits': predict},
    save_weights=False, output_dir='runs/gpu_posterior', buffer_size=4,
)
```

The two chunk sizes have different purposes:

1. **Likelihood `batch_size`** iterates over every training example on every potential evaluation. The last, smaller chunk is included without reweighting. This is exact full-data likelihood accumulation, not stochastic-gradient HMC. The library API retains `checkpoint_batches=True` for compatibility; the example CLIs default to no checkpointing and enable it with `--checkpoint-batches`. Checkpointing trades recomputation for lower activation memory and is bypassed for a single full batch. `data_device='cpu'` streams data chunks from host memory. By default all training data reside on the sampling device, which is faster when they fit.
2. **Draw `buffer_size`** controls the number of selected weight/prediction draws buffered before transfer to CPU/disk. Pyro uses `StreamingMCMC(save_params=[])`; it does not retain the whole weight trajectory. The buffer and its transient stacked copy use O(buffer_size * saved_values_per_draw) device memory, independent of the total number of draws. For very large networks use `buffer_size=1`, or save only predictions.

Pyro still needs the current state, gradients, mass adaptation and trajectory state on device. NUTS additionally retains tree states. Model activations and data also consume memory; streaming is not a guarantee that an arbitrarily large model fits.

`output_dir=None` stores requested arrays in CPU RAM. With `output_dir`, arrays are memory-mapped `.npy` files and flushed incrementally. This avoids keeping the entire trajectory in GPU or CPU application memory, though the operating system may cache file pages. Saving weights still requires approximately `chains * draws * parameter_count * dtype_bytes` **disk space**. Prediction-only runs do not retain theta, so they cannot later evaluate new test inputs unless rerun.

For a job that should write histories only after **all** chains finish, use
`output_dir=None`, then `result.save('runs/completed_samples')`. This exports a
complete result in the same `.npy`/manifest format accepted by
`SamplingResult.load`. It refuses partial results and existing destinations,
writes into a temporary sibling directory and publishes the directory only when
all arrays and metadata have been written. No streaming/checkpoint files are
created during sampling in this mode. Retained observables must fit in host RAM;
`save_weights=False` still avoids storing the weight trajectory.

Both example CLIs prepare prediction inputs on the sampling device once and default to full likelihood and prediction batches. Use `--likelihood-batch-size` and `--predict-batch-size` to chunk either calculation. `--data-on-cpu` keeps both training and prediction inputs on the CPU and transfers chunks on demand. For repeated predictions through the Python API, pass a persistent device tensor as in the example above.

`NetworkPosterior.compile()` optionally uses `torch.compile` for the scalar potential and its autograd path. This can reduce per-evaluation overhead; it does not compile Pyro's HMC/NUTS control logic. The examples' `--compile` flag also compiles the prepared prediction callable. `--compile-mode max-autotune-no-cudagraphs` spends additional startup time selecting kernels without CUDA graphs; the default mode is `default`. Compilation failures propagate. The initial potential/gradient compilation is included in setup time; prediction compilation happens on its first invocation and is included in sampling time. Do not assume compilation or GPU always wins for tiny problems.

Float64 is the reference default. Float32 is available explicitly when GPU throughput/memory benefits justify checking energy stability, divergences and predictive agreement against float64 on a smaller case. Float16/bfloat16 and AMP are not supported. During CUDA sampling, TF32 is disabled and cuDNN uses deterministic convolution algorithms. `SamplerConfig(cudnn_benchmark=True)` or `--cudnn-benchmark` permits autotuning among those algorithms for each shape. Benchmark selection may differ between separate program runs, so this option is recorded in the sampler configuration and defaults to false. Previous backend settings are restored even if sampling raises. No global thread count is changed by the library.

HMC can be faster than NUTS for large GPU workloads because NUTS makes more adaptive host-side decisions. Choose by measured effective samples per second and convergence, not raw draws per second. The earlier toy checks found HMC efficient for weight sampling, while short HMC trajectories gave poor mixing for some GP controls. See [those results](validation/posterior_sampling_p32.md).

For the existing five-layer image architecture, with an already prepared NCHW NPZ dataset:

```bash
python examples/sample_posterior.py --data cifar_subset.npz \
  --network cnn --layers 5 --width 128 --outputs 10 --activation relu \
  --mask 4 3 3 3 3 --stride 4 1 2 1 2 --padding valid same same same same \
  --precisions .5 --gamma 1 --temperature .2 --sampler hmc \
  --device cuda:0 --warmup 1000 --draws 2000 --compile \
  --compile-mode max-autotune-no-cudagraphs --cudnn-benchmark --buffer-size 2 \
  --output runs/cifar_pyro
```

That is a launch example for a GPU with enough memory for full batches, not a runtime or convergence guarantee. Add explicit batch sizes and, if needed, `--checkpoint-batches` on smaller devices. ReLU's nonsmooth potential is another reason to begin with smooth erf controls and inspect divergences carefully.

## Multiple CPUs/GPUs

The library runs chains sequentially to bound device memory and avoid pickling arbitrary model/observable closures. It does not vectorize Pyro trajectories or silently launch competing processes on one GPU. For parallel chains, launch one process per available CPU allocation or GPU with the **same data/settings/base seed** and different `chain_offset` values and output directories:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/sample_posterior.py --data data.npz \
  --device cuda --chains 1 --chain-offset 0 --output runs/chain0 &
CUDA_VISIBLE_DEVICES=1 python examples/sample_posterior.py --data data.npz \
  --device cuda --chains 1 --chain-offset 1 --output runs/chain1 &
wait
```

Use four independent chains for the default diagnostic gate. Check that posterior and data settings match before combining. For diagnostics on selected outputs, preserve the chain axis:

```python
from deepbays.sampling import SamplingResult
runs = [SamplingResult.load(f'runs/chain{i}/samples') for i in range(4)]
# Select a bounded number of test examples before copying into RAM.
logits = np.concatenate([r.get_samples('predictions')[:, :, :16] for r in runs], axis=0)
chain_info = [row for r in runs for row in r.chain_info]
```

## Files, interruption and later predictions

`SamplingResult` exposes:

- `get_samples(name='theta')`: completed chains only, retaining `(chains,draws,...)` axes. It raises if no chain finished.
- `chain(name, index)`: the valid prefix of one chain, including an interrupted chain.
- `completed_draws`: counts of committed draws per chain.
- `chain_info`: acceptance fraction, divergences and their retained-draw indices, step size, seed, timings and potential-evaluation counts.
- `metadata`: sampler/posterior configuration, versions, status, array shapes and optional user metadata.
- `save(directory)`: export all chains of a fully completed result to a new directory, without changing the result or the existing streaming API.

An atomic `manifest.json` is written after array data are flushed. `SamplingResult.load(path)` opens read-only memory maps and validates their shapes. Preallocated unwritten tails exist in the low-level `arrays` mapping; use the accessors to avoid treating them as samples. A signal/error may lose at most the currently uncommitted buffer. Only committed draws are exposed by these accessors. An interrupted run is inspectable; it is **not** an exact resumable chain checkpoint because Pyro's full adaptive/integrator/RNG state is not serialized. Starting again requires a new directory and a deliberate new run.

`on_chunk(chain_id, first_draw_index, cpu_arrays)` runs after a committed chunk. `on_chain_end(result, chain_id)` can generate reports after each completed chain. Observables and callbacks must not mutate the posterior or theta. Callbacks should not retain every chunk if bounded CPU memory is desired.

In addition to a mapping of names to callables, `observables` may be one
callable returning a name-to-tensor dictionary. Use this to compute several
statistics from one model forward pass. Names, shapes and dtypes must remain
fixed across retained draws and chains; the same storage checks apply.

If weights were saved, predict later without loading them all:

```python
from deepbays.sampling import iter_predictions
saved = SamplingResult.load('runs/weights')
for chain, draw, output in iter_predictions(posterior, saved.get_samples(), new_X, batch_size=128):
    # output is one network's CPU NumPy prediction batch.
    ...
```

Reconstruct the same architecture and precision specification. Saved theta is whitened, not a physical PyTorch state_dict.

## Cross-entropy utilities and diagnostics

Experiments using two independent chains can set
`diagnostics_and_mcse(..., min_chains=2, min_ess=100)`. This explicitly changes the
chain-count/ESS policy while retaining rank-normalized split/folded Rhat, bulk and
tail ESS and divergence checks. The default remains four chains and ESS 400.
Increasing the number of retained draws does not by itself guarantee either ESS
threshold. Diagnose offline and retain the per-observable report.

`predictive.py` is independent of HMC/NUTS. `classification_statistics` returns posterior mean softmax probabilities, their variances, and argmax frequencies. With test labels it also returns mean-predictor accuracy, expected sampled-network accuracy and per-draw joint-test accuracy. Inference temperature is not automatically applied a second time to these predictive logits. `sample_categorical_labels` separately samples class observations from softmax probabilities, using a local RNG; this includes categorical observation randomness. `logit_contrasts` removes the irrelevant common logit before diagnosing functions. Regression predictions remain raw outputs.

`diagnostics_and_mcse` uses ArviZ rank/folded split Rhat, bulk/tail ESS and autocorrelation-aware mean MCSE. By default its gate requires four chains, max Rhat <=1.01, minimum bulk/tail ESS >=400, and—when chain metadata are supplied—zero divergences and complete chains. Nonfinite inputs are rejected; constant observable coordinates cannot pass the gate. Diagnose predictive means, centered squares/cross-products, probabilities and aggregate test losses, not only weight coordinates with hidden-neuron symmetries. A constant discrete accuracy can fail even when continuous observables mix; inspect the per-observable report rather than dropping the failed gate silently. Passing these checks is evidence about the monitored observables, not proof of global exploration or theory validity.

## Validation and source map

- `posteriors.py`: whitening, full-data losses, chunking/compilation, and Gaussian function controls.
- `mcmc.py`: configuration, independent chains, Pyro hooks, RNG and CUDA settings.
- `storage.py`: CPU/memory-mapped trajectories and atomic progress manifests.
- `predictive.py`: softmax summaries, contrast projection and posthoc prediction.
- `diagnostics.py`: convergence gates and MCSE.

Inside `mcmc.py`, `sample_posterior` sets up the run and loops over chains.
The private `_run_chain` handles one native Pyro chain and its diagnostics;
`_ObservableCollector` evaluates and buffers requested tensors, then commits
chunks before calling `on_chunk`. These are internal boundaries: public calls,
results and callback conventions are unchanged. The
[orchestration validation note](validation/sampling_orchestration_refactor.md)
records compatibility checks and CPU/GPU timings.

The migrated `examples/validate_posterior_sampling.py` uses the same public API for networks and fixed-Q GP controls. It keeps its older compact NPZ/report outputs and additionally writes streaming `*_samples` directories. GPU controls include `--likelihood-batch-size`, `--predict-batch-size`, `--checkpoint-batches`, `--data-on-cpu`, `--compile`, `--compile-mode` and `--cudnn-benchmark`. Older validation results can still be reanalyzed.

Tests cover analytic regression posteriors for HMC and NUTS, equality to native Pyro draws, chunked/full and compiled/eager gradients, deep CNN normalization, bias priors, float32 controls, disk reload/interruption, RNG restoration, chain offsets, predictive probabilities versus categorical draws, and diagnostics failure cases. CUDA energy/gradient and streaming tests are included and skipped explicitly when no usable device/driver is present.

References: [Pyro MCMC and StreamingMCMC](https://docs.pyro.ai/en/stable/mcmc.html), [ArviZ diagnostics](https://python.arviz.org/en/v0.22.0/api/diagnostics.html), [rank-normalized Rhat/ESS](https://arxiv.org/abs/1903.08008).

Measured checks and reproducible commands are recorded in the [infrastructure validation report](validation/pyro_sampling_infrastructure.md).

The [GPU optimization checks](validation/pyro_gpu_speedups.md) record CUDA correctness and representative before/after timings for the faster parameter reconstruction and prepared prediction inputs.
