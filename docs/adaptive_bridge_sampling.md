# Adaptive bridge sampling of prior projection tails

`sample_adaptive_bridge` complements the calibrated `UmbrellaTilt` workflow.
Both sample the same full network with ordinary Pyro HMC/NUTS. An umbrella
allocates observations over a broad interval in one distribution; a bridge
connects a sequence of narrower, overlapping power/root distributions to the
prior. Its normalization is measured from network samples, without an EWA
normalizer or an assumed nonlinear density.

## Small example and Python API

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/sample_bridge_prior.py --output runs/bridge_demo
# Optional local GPU:
python examples/sample_bridge_prior.py --device cuda --output runs/bridge_gpu
# Replot existing samples; no further sampling:
python examples/sample_bridge_prior.py --replot runs/bridge_demo
```

Defaults are Erf, P=N=16, L=2, float64, unit precisions, a correlated synthetic
input covariance, and a centered Gaussian disorder direction. The requested
interval is Q=0.2..2.5. Two chains use 120 discovery draws and 600 production
draws per window, with 80 warmup steps each. Each phase also receives 1,200
independent positive-prior draws per chain. This is a short demonstration of
the workflow, not a precision or mixing guarantee for difficult disorder.
The measured runtime and validation are recorded below.

The example writes three figures:

- `rate_function.png`: plain histogram points with ±2 Monte Carlo SE, independent
  prior points in the bulk, and a central-EWA reference (finite N for L≤2,
  leading large N for deeper networks).
- `windows_and_normalization.png`: biased histograms, bias functions, measured
  BAR/MBAR normalizers, and adjacent-window overlap.
- `histories_and_diagnostics.png`: endpoint Q, inverse log bias and signed
  projection histories; Q and all-layer signed ESS/Rhat across every window;
  and a forward/reverse mean comparison.

All retained rejections count as observations. Histogram estimates are bin
averages; the plotted midpoint does not imply a pointwise density evaluation.
The EWA guide is never used in the empirical normalization. Rate curves share
one prior-modal bin; tail segments are not independently shifted.

A full library call is:

```python
import numpy as np
from dataclasses import replace
from deepbays.sampling import (
    KernelTailTarget, SamplerConfig, sample_adaptive_bridge, BridgeResult,
)

P = N = 12
f = np.random.default_rng(12).normal(size=P)
f -= f.mean()
target = KernelTailTarget.from_covariance(
    np.eye(P), f, width=N, layers=2, activation='relu',
)
config = SamplerConfig(sampler='hmc', chains=2, draws=240, warmup=80,
                       trajectory_length=1.1, target_accept=.9, seed=81)
result = sample_adaptive_bridge(
    target, lower=.4, upper=2.5, config=config,
    discovery_config=replace(config, draws=120), prior_draws=1200,
    output_dir='runs/my_bridge',
)
rate = result.estimate_rate(np.linspace(0, 2.5, 31), bootstrap=80)
# Reload without sampling:
saved = BridgeResult.load('runs/my_bridge')
```

`config` is the existing `SamplerConfig`; it controls fixed-ladder production.
`discovery_config` uses the same type and must share the chain count and offset.
When omitted, discovery uses production settings with 250 retained draws.
Set `lower=None` or `upper=None` for a single tail. Raw data, supplied covariance
factors and non-isotropic covariance matrices use the existing target constructors.
The example accepts `--data prepared.npz` containing `C` and `bar_f`.

The input target is unchanged. `target.with_tilt(bias, positive_only=True)`
returns a cheap copy sharing its model and data tensors; validation is retained
and compiled closures are cleared. Network values and gradients use the same
implementation as the umbrella and focused-tilt methods. No layerwise sampler,
custom integrator or additional force evaluation is introduced.

## Distributions and common support

Let p₀(z) denote the whitened Gaussian network prior conditional on Q(z)>0.
For the right tail use

\[
p_\kappa(z)=p_0(z)Q(z)^\kappa/Z_\kappa,\quad \kappa\ge0.
\]

For the left use `RootTilt(theta=-eta)`,

\[
p_\eta(z)=p_0(z)\exp[-N\eta Q(z)^{1/L}]/Z_\eta,\quad\eta\ge0.
\]

The prior is the shared anchor, with bias zero and Z₀=1. It is sampled exactly
by independent Gaussian draws, rejecting Q=0. Rejection counts are saved.
Conditioning is imposed consistently on every window, including negative-root
ones. The resulting density excludes the ReLU atom; obtaining the unconditional
positive density would additionally require multiplication by Pr_prior(Q>0).
No positive Q is clipped or softened. A safe inactive branch avoids autograd
NaNs from log(0), without modifying positive values or their gradients.

The existing potential remains

\[
U(z)=\tfrac12\|z\|^2-b(Q(z)),\qquad
\nabla U=z-b'(Q)\nabla Q.
\]

Positive powers and negative roots are normalizable for the supported network
priors. Normalizability does not ensure rapid mixing, particularly near a deep
ReLU boundary or for latent signed modes at similar Q.

## Choosing and validating the ladder

For a current window, the next parameter increment δ is the largest one whose
empirical relative importance ESS is at least `step_ess` (default 0.65):

\[
r(\delta)=\frac{(\sum_m e^{\delta s(Q_m)})^2}
                 {M\sum_m e^{2\delta s(Q_m)}}.
\]

The score is s(Q)=log Q on the power path and s(Q)=−N Q^(1/L) on the root path.
A scalar bracketed root solve determines δ. This is the established adaptive
ESS/conditional-ESS tempering rule; they coincide for equally weighted samples.
The workflow borrows the spacing rule from adaptive tempering, but is not a
particle-resampling SMC algorithm.

The proposed window receives fresh warmup from the preceding terminal chain
coordinates. Actual bidirectional Bennett overlap is then measured. Discovery
requires overlap ≥0.2, Q bulk ESS ≥50 and Q Rhat ≤1.15 by default. These modest
checks do not assert full convergence. Signed-projection and Gaussian-energy
diagnostics across every layer remain visible, without a separate hard gate.

Each step gets at most three attempts. Inadequate overlap halves δ. Inadequate
Q mixing retries with twice the original pilot draws, without unbounded growth;
failed attempts remain archived and never enter the final estimator. Each tail
has a maximum window count and parameter cap. If an endpoint cannot be reached,
the attained ladder is frozen and labeled as partial coverage. Production still
samples that ladder; `coverage_complete` is separate from `production_complete`.

Discovery continues until its observed mean passes the requested endpoint.
This is an allocation rule, not evidence of accurate density estimation in every
bin. Production histograms and diagnostics determine useful reported coverage.

## Frozen production and measured normalization

Discovery selects the schedule. A fresh production pass traverses each branch
in reverse order, using independent random streams and new warmup/adaptation at
every fixed window. Only terminal coordinates carry over, not momenta or mass
adaptation state. Production inherits discovery endpoint coordinates, so it is
not literally independent: warmup must remove their influence. Reverse traversal
provides a useful hysteresis check but cannot rule out metastable modes.

Only production observations enter `BridgeResult.estimate_rate`. The sampler
retains fixed production draw counts even when diagnostics are poor, exposing
the problem rather than adapting a stopping rule to the final estimates.

For neighboring distributions a,b, the Gaussian prior cancels:

\[
\frac{p_b(z)}{p_a(z)}=
\exp[b_b(Q(z))-b_a(Q(z))]\frac{Z_a}{Z_b}.
\]

Bennett's acceptance ratio (BAR) estimates log(Z_b/Z_a) from both windows.
Sequential BAR sums are an audit. Multistate Bennett (MBAR) estimates all log Z
jointly with log Z₀=0. A small SciPy convex optimizer solves the standard
reverse-logistic equations. It is an inference step after sampling, not a new
MCMC transition. Unequal sample counts and additive bias constants are handled
explicitly and checked against analytic distributions.

For pooled samples Q_m, the prior weight is

\[
w_m=\left[\sum_j M_j\exp\{b_j(Q_m)-\log Z_j\}\right]^{-1}.
\]

At a converged solve these weights sum to one, up to roundoff. Histogram density
in bin i is Σ_m w_m 1_i(Q_m)/Δ_i. Every retained observation, including observations
outside the displayed bins, remains in the mixture denominator. M_j is the
actual draw count, never ESS substituted for a count. Correlation affects errors
and diagnostics, not the identity of the sampled mixture.

`estimate_bridge_rate` also accepts explicitly supplied lists of Q arrays and
tilts. The first must be exact iid positive-prior observations with PowerTilt(0).
All arrays must describe the same network, input covariance, disorder and support;
these identities cannot be inferred from scalar Q arrays alone. The integrated
runner enforces them by sharing one target.

## Errors, storage and practical limits

A joint bootstrap refits all MBAR normalizers in each replicate. The iid prior
is resampled from its pooled observations. Tilted windows use circular time
blocks, with chain identities shared within each sequential branch and separate
between the two branches. This retains observed chain effects and normalization
covariances; it does not model every finite warm-start dependence exactly.
Default blocks are twice an estimated positive-sequence Q autocorrelation time,
with a minimum of five draws. Inspect slower signed modes and increase block
lengths or sampling when appropriate. Two short chains do not establish robust
coverage of unknown modes.

The returned `relative_mcse` is the absolute MCSE of log density, approximately
the fractional density error for small errors. `mcse_rate` divides it by P.
`mcse_shifted_rate` additionally includes covariance with the common modal bin.
The modal bin is selected from iid prior density counts, then treated as fixed.
Errors omit mode-selection effects, finite-bin bias, disorder variability and
unvisited modes. Empty bins are NaN with infinite MCSE. A bin empty in any
bootstrap replicate also has unresolved MCSE, rather than a deceptively small
error bar. There is no KDE interpolation between windows.

Native `SamplingResult` files stream every 64 retained draws by default. Each
completed HMC window saves one terminal coordinate vector per chain, rather than
an entire weight history. `bridge.json` is atomically updated after completed
attempts/windows and records failures. `BridgeResult.load` requires completed
production; interrupted native windows can still be inspected individually.
Checkpointing is not resumable HMC. Use a new output directory for a new run.

The full MBAR calculation uses memory proportional to windows × pooled scalar
draws; bootstrap computation can dominate extremely large analyses. Increasing
system size or requested depth also increases HMC force costs, the number of
windows or the time needed to mix within a window. Overlap alone cannot certify
mixing. Compared with an umbrella, bridges reduce each chain's Q range but add
repeated warmup and accumulated normalizer uncertainty; neither method is
universally faster.

## Short methods description

> We sample full network weights with standard Pyro HMC along overlapping power
> and negative-root tilted priors. Discarded pilot windows choose bias increments
> by an importance-ESS rule and verify neighboring overlap. Fresh chains on the
> frozen ladder estimate normalization relative to independent prior draws using
> multistate Bennett reweighting. We report unsmoothed density histograms with
> joint chain/block-bootstrap Monte Carlo errors and observable mixing diagnostics.

## Integration validation (2026-10-23)

The original smaller example (ReLU, P=N=12, Q=0.4..2.5, 240 production draws)
completed in **45.7 seconds on the local CPU**,
including discovery, production, 40 bootstrap refits and all three figures.
It used five distributions: the iid prior and two windows on each side.
Discovery endpoint means were 2.573 and 0.370; reverse-production means were
2.520 and 0.355. There were 4,320 retained production observations, including
2,400 iid prior draws. Minimum production Q bulk ESS was 216, maximum Q Rhat
1.011, and minimum neighboring overlap 0.870. Signed diagnostics across layers
had minimum ESS 99 and maximum Rhat 1.024. These short-run diagnostics are
illustrative, rather than a claim of precision across arbitrary disorder.

The tests in `tests/test_adaptive_bridge.py` cover analytic Gamma normalizers
for both bias signs, unequal counts, BAR reversal, MBAR bias-constant invariance,
normalized histograms, serial-correlation effects, positive-support gradients,
frozen production, target/RNG preservation, output reload, parameter-limited
coverage and interrupted-run reports. An explicit test checks score-equation
refinement after premature optimizer stopping. The native bridge test also
passes on the local CUDA GPU. Existing tilted-prior, umbrella, storage,
posterior orchestration and native-Pyro equivalence checks pass on CPU; a
separate existing tilted-prior CUDA smoke test passes as well.

The implementation adds one sampling module, `bridge.py`. It reuses the existing
sampler configuration, storage and diagnostics; rate inference remains in
`rate_functions.py`. No optional dependency or custom integrator was added.
The larger development-folder L=1 and L=5 experiments motivated the design;
this integration validation does not imply those archived runs were generated
by the new library wrapper.

## Primary references

- Zhou, Johansen and Aston, [Toward Automatic Model Comparison: An Adaptive
  Sequential Monte Carlo Approach](https://arxiv.org/abs/1303.3123), Algorithm 4.
- Bennett, [Efficient Estimation of Free Energy Differences from Monte Carlo
  Data](https://doi.org/10.1016/0021-9991(76)90078-4), 1976.
- Shirts and Chodera, [Statistically optimal analysis of samples from multiple
  equilibrium states](https://pmc.ncbi.nlm.nih.gov/articles/PMC2671659/), 2008.
