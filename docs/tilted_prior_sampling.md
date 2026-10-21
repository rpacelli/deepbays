# Tilted prior sampling and kernel-projection rate functions

This API samples a full finite-width FC network path using the existing
**ordinary Pyro HMC/NUTS** engine. Power and root tilts define two potentials,
not two transition algorithms. HMC is the default. There are no custom
integrators, independence proposals or layer updates.

## A complete small example

```python
import numpy as np
from deepbays.sampling import (
    KernelTailTarget, PowerTilt, RootTilt, SamplerConfig,
    sample_tilted_prior, diagnostics_and_mcse,
    central_ewa_log_normalizer, estimate_density_rate,
)

rng = np.random.default_rng(12)
P, N, L = 16, 16, 2
X = np.sqrt(P) * np.eye(P)  # Or prepared data with shape (P,input_dimension).
bar_f = rng.normal(size=P)
bar_f -= bar_f.mean()       # Optional: suppress first-layer ReLU mean overlap.
tilt = PowerTilt.for_mean(2.5, width=N, layers=L)
target = KernelTailTarget(X, bar_f, width=N, layers=L,
                          activation='relu', tilt=tilt, device='cpu')
config = SamplerConfig(sampler='hmc', chains=4, warmup=250, draws=600,
                       trajectory_length=1.5, target_accept=.8, seed=12)
result = sample_tilted_prior(target, config)
q = result.get_samples('Q')  # (chains, draws), including retained rejections.
obs = {name: result.get_samples(name) for name in result.arrays}
diagnostics, mcse = diagnostics_and_mcse(obs, chain_info=result.chain_info)
log_z = central_ewa_log_normalizer(tilt, width=N, layers=L)
rate = estimate_density_rate(q, [2.0, 2.5, 3.0], tilt=tilt, width=N, layers=L,
                             P=P, log_normalizer=log_z, bandwidth=.08)
print(diagnostics['passed'], rate['rate'], rate['mcse_rate'])
```

`PowerTilt.for_mean` matches the central-EWA mean: exact for identity, a guide
for nonlinear networks. Use `PowerTilt(kappa=...)` to set the exponent directly
or `RootTilt(theta=...)` for the second family. Means below one select negative
strengths; negative powers are supported only for identity, while negative
root tilts support all three activations. A short configuration does not
guarantee that convergence diagnostics pass.

The result is the same `SamplingResult` used for posterior sampling. Supply
`output_dir='runs/my_tilt'` to stream to disk, or use `result.save(...)` after
an in-memory run. Existing results are not overwritten.

## Runnable rate-function plot

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/sample_tilted_prior.py --output runs/tilted_prior_demo
```

Defaults: P=N=16, L=2, Erf and ReLU, EWA target means 1.5, 2 and 2.5; four
chains per window, 250 warmup and 600 retained draws. This is intended as a
few-minute laptop CPU demonstration. It also draws an independent prior
baseline, without downloading a dataset. Outputs include `rate_function.png`,
`sampling_diagnostics.png`, `report.json`, settings, the exact inputs, iid prior
draws and streamed chain observables. Sampling and analysis are separate functions in the example;
the HMC target and transition kernel do not depend on plotting choices.

```bash
# Exact normalization control:
python examples/sample_tilted_prior.py --activations id --output runs/tilt_id
# Root target, with the conservative ReLU bound enforced:
python examples/sample_tilted_prior.py --tilt root --output runs/tilt_root
# Prepared NPZ containing X and bar_f; no implicit preprocessing:
python examples/sample_tilted_prior.py --data prepared.npz --width 100 \
  --device cuda --output runs/tilt_data
# Change smoothing without rerunning any HMC chains:
python examples/sample_tilted_prior.py replot runs/tilted_prior_demo \
  --output runs/tilted_prior_demo_h006 --bandwidth .06
# Left tail for Id, Erf and ReLU, using the same standard Pyro HMC:
python examples/sample_tilted_prior.py --tilt root --means .8 .5 .3 \
  --activations id erf relu --output runs/tilt_left_root
# Exact power-tilt benchmark on the left:
python examples/sample_tilted_prior.py --tilt power --means .8 .5 .3 \
  --activations id --output runs/tilt_left_power
```

The root example matches the large-N saddle, then caps positive theta at 80%
of the actual target's sufficient bound; negative theta needs no such cap.
It labels the actual theta, without claiming
a finite-N mean match. Capping can give repeated strengths; larger studies
should select strengths explicitly through the API.

The plot uses the paper's density-rate convention **-log rho(Q)/P**, shifted by
one independent prior-modal constant per activation. All windows share that
offset: no per-window alignment is fitted. The finite-N central-EWA curve for
L=1 or 2 uses its own modal offset; at larger L the example plots the leading
large-N rate. Finite-N and large-N curves should not be conflated.

Replotting writes a new analysis directory and preserves the source run. Old
runs that did not archive the iid baseline regenerate only those cheap prior
draws from the saved inputs, device and seed; the report records this. New runs
reuse the archived baseline. The source sampling run remains the input for
further replots; analysis directories contain no copies of the HMC chains.

### Histories, log weights and ESS

Both sampling and `replot` also write `sampling_diagnostics.png`, with one
row per activation and tilt. The columns show Q histories for every chain,
inverse-bias log-weight histories, and ESS summaries across all chains. Traces
include repeated rejected states and exclude discarded warmup. Chain colors
match between the two history panels.

Writing the biased density as p_tilt proportional to p_prior exp(b(Q)), the
plotted log weights are **-b(Q)**: the correction back toward the prior, with
the constant log normalizer omitted. This constant affects neither their
fluctuations nor the displayed weight ESS.

The ESS panel distinguishes ArviZ bulk/tail ESS of Q (blue, accounting for
MCMC dependence) from **Kish weight ESS**, (sum w)^2 / sum(w^2) (orange,
measuring only the concentration of pooled inverse-bias weights). Weight ESS
does not correct for autocorrelation and is not the ESS of the local density
estimator. A low global weight ESS can occur even when the chain samples its
tilted window well. The dotted line marks the total retained draw count;
MCMC ESS can exceed that count for antithetic samples. Constant-Q MCMC ESS
and nonfinite-weight ESS are marked unavailable. Numeric weight ESS is also
saved under each window's `trace_diagnostics` in `report.json`.

### Bandwidth and colored ranges

`--bandwidth h` is the **standard deviation of a Gaussian kernel in log Q**.
The density estimator averages

\[
K_h(\log x-\log Q_j)=\frac{1}{\sqrt{2\pi}h}
\exp\left[-\frac{(\log x-\log Q_j)^2}{2h^2}\right].
\]

At Q=x, one bandwidth spans x exp(-h) to x exp(h). Thus h=0.08 corresponds
to roughly -7.7% / +8.3% around x: an absolute width near 0.08 at x=1, and
0.4 at x=5. This is an absolute log-space bandwidth, not a multiplier of an
automatic bandwidth rule. Smaller h resolves finer structure but increases
Monte Carlo variance; larger h reduces variance while increasing smoothing.
The report compares h with h sqrt(2) as a sensitivity check, not a bias bound.

Each color is a **separate tilt strength**, evaluated at seven points between
the 15th and 85th percentiles of its positive retained Q values. These ranges
cover the central 70% of the continuous component and are independent of h.
Any zero draws remain in the density normalization and MCSE time series;
they are not silently conditioned away. When any requested mean is below
one, evaluation points and the horizontal plot axis use logarithmic spacing.
Bandwidth does enter the local support
check: at least ten draws within two bandwidths in every chain, and a kernel
weight concentration ESS of at least 100. This latter count is not the
autocorrelation ESS. Red crosses flag failed local checks; open markers mean
the full chain diagnostic gate is inconclusive. The black prior curve stops
where fewer than twenty iid draws lie within two bandwidths.

### What the uncertainty bands include

All shaded bands, including the black prior curve, show **pointwise ±2 Monte
Carlo standard errors (MCSE)**. They are first-order delta-method estimates,
conditional on the chosen bandwidth, supplied EWA normalizer and selected
grid mode. They are not simultaneous confidence bands. Sparse tails can be
non-Gaussian even before the support cutoff.

For the independent baseline, the error of each kernel average follows from
the sample variance divided by the number of draws. For tilted chains, the
estimator uses autocorrelation-aware MCSE of the kernel contributions, with
batch-means and between-chain safeguards. It does not substitute the ESS of Q
for the ESS of a local density estimate. In either case,

\[
\operatorname{SE}(\hat I(x))\simeq
\frac{\operatorname{SE}(\hat\rho_{b,Y}(\log x))}
     {P\,\hat\rho_{b,Y}(\log x)}.
\]

The colored errors are computed separately at every point. They may look
nearly constant because only each tilt's well-populated central region is
displayed and the rate measures *relative* density error divided by P.
The prior reaches its sparsely sampled tail, where its errors grow.

The plotted rates share an estimated offset c=log(rho_prior,max)/P. The
black curve's MCSE includes covariance between its local density and this
modal estimate: it is zero at the selected grid mode by construction. For
the independent tilted chains, Var(c) is added to their rate variances.
This common offset correlates errors across all colored points and can
also make their widths look similar. It must not be counted as independent
noise when comparing two tilted curves. The report keeps unshifted and
shifted errors separately (`rate.mcse_rate` and `mcse_shifted_rate`).

These bands exclude EWA normalization error, smoothing bias, missed modes,
and higher-order effects of selecting the mode from the data/grid. In
particular, a narrow band is not a bound on total error. Window disagreements
can reflect EWA error even with good mixing; the known bias and the supplied
normalizer contribute no Monte Carlo variance of their own.

## Network and target

For prepared data X with d columns and layer precision lambda_l,

\[
h_1=XW_1^T/\sqrt d,\quad \phi_l=\sigma(h_l),\quad
h_l=\phi_{l-1}W_l^T/\sqrt N\;(l>1),\qquad W_l=z_l/\sqrt{\lambda_l}.
\]

All hidden matrices z_l are sampled jointly with independent standard-normal
priors. There is no readout. The observable is

\[
\mathcal Q=\frac{\|\bar f^T\phi_L\|^2}{ND},\qquad
D=\bar f^T K_{\infty,L}\bar f.
\]

K_inf,0=XX^T/d. Each next reference kernel is the Gaussian activation second
moment applied to K_inf,l-1/lambda_l. In particular, the ReLU kernel is
uncentered: it includes the mean contribution. D is a fixed NNGP reference,
not a fitted finite-width mean. The potential passed to Pyro is exactly

\[
U(z)=\tfrac12\|z\|^2-b(\mathcal Q(z)),\qquad
b(Q)=\kappa\log Q\quad\text{or}\quad N\theta Q^{1/L}.
\]

Supported: Id, Erf, ReLU; equal hidden widths; zero biases; `precisions` as a
positive scalar or L positive hidden-layer precisions. The target does not
assume Gaussian Q or use an EWA approximation.

Initialization uses independent prior draws from private RNG streams, separate
from transition streams. Invalid Q=0 positive-power starts are retried. Warmup
must remove initialization effects; difficult targets may need user-supplied
dispersed `initial` states. These starts are not importance proposals.

## Inputs and device

Raw `X` may be NumPy or Torch, with arbitrary non-isotropic covariance.
`KernelTailTarget.from_factor(B, bar_f, ...)` supplies C=BB^T, including
rectangular factors. `from_covariance(C, ...)` uses the existing support
factorization: it removes eigenvalues at spectral roundoff, records the
tolerance and adds no jitter. Use a supplied factor for a different policy.
A zero reference projection is rejected. Covariance compression preserves
the law on the fixed input set, not predictions on arbitrary new raw inputs.

Inputs and direction are copied; hashes are recorded. Archive the arrays as
the example does. Keep the target fixed while sampling.

`device='cuda'` uses the same target and Pyro kernel. Float64 is the default;
inputs, weights and gradients remain on-device. Only requested observables
are transferred to CPU. `target.compile()` is optional and off by default.

## Root integrability

Nonnegative finite powers are normalizable for these models. Negative powers
are permitted only for identity, with kappa>-N/2. Every finite negative root
strength is normalizable because its weight is bounded by one. Identity root
tilts require theta<L/2. Erf is bounded, so its actual target permits every
finite theta. For a positive ReLU root tilt, put

\[
c=\frac{\max(\|\bar f_+\|^2,\|\bar f_-\|^2)\|X/\sqrt d\|_{op}^2}
         {D\prod_l\lambda_l}.
\]

The inequality Q <= c product_l(||z_l||_F^2/N) and AM-GM give the sufficient
condition theta < L/(2 c^(1/L)). The constructor enforces it. This ReLU bound
can be conservative: exceeding it does not prove impropriety.

## Separate EWA normalization and rate reconstruction

The central EWA takes Q=product_l q_l with independent q_l~chi2_N/N. Hence

\[
\log Z_{power}=L[\kappa\log(2/N)+\log\Gamma(N/2+\kappa)-\log\Gamma(N/2)].
\]

For positive root bias the helper sums

\[
Z_{root}=\sum_{k\ge0}\frac{(N\theta)^k}{k!}
\left[(2/N)^{k/L}\frac{\Gamma(N/2+k/L)}{\Gamma(N/2)}\right]^L.
\]

A geometric remainder bound controls truncation; nonconvergence raises an
error. This EWA series requires 0<=theta<L/2 independently of the actual
network bound, including for bounded Erf targets. Negative theta uses the
one-dimensional integral described below, not an alternating version of this
series. For powers, the gamma formula requires kappa>-N/2.

For Y=log Q, the exact relation at x>0 is

\[
\rho_{0,Q}(x)=Z_b e^{-b(x)}\rho_{b,Y}(\log x)/x.
\]

The estimator smooths rho_b,Y with a fixed Gaussian bandwidth and then applies
this relation, including the 1/x Jacobian. It neither self-normalizes inverse
weights nor drops rejections. MCSE uses autocorrelation, batch and
between-chain safeguards. Check local hits in every chain and bandwidth
sensitivity; positive Gaussian-KDE tails alone do not establish overlap.

Using Z_EWA multiplies the true density by Z_EWA/Z_true. MCSE excludes that
model error, KDE bias and missed modes. The API returns unshifted rates;
the example propagates its separately fitted modal offset as described above.
Identity supplies an exact-normalization control. Both log-Q estimators
handle a zero atom by counting all draws in the density normalization while
assigning a zero positive-density kernel contribution to Q=0. The tilted
estimator keeps zeros in their original time positions for MCSE. At least
one positive draw is needed to estimate a continuous density. The zero mass
is reported separately using `estimate_left_probability` at threshold zero.

## Left tails: applicability and limitations

The target remains U(z)=||z||^2/2-b(Q(z)), with the same Pyro HMC, warmup,
storage and diagnostics. Only the sign of the bias and its normalization
change. This does not imply that a strong left bias has the same mixing cost
as a right bias.

### Negative powers: an exact identity benchmark

For identity, Q is a product of L independent Gamma(a, rate=a) variables,
with a=N/2. Multiplication by Q^kappa changes each factor's shape to a+kappa,
so

\[
\kappa>-a,\qquad E_\kappa Q=(1+\kappa/a)^L,
\qquad \kappa=a(m^{1/L}-1).
\]

These formulas apply on either side of m=1. The small-Q density is
proportional to Q^(a-1) times a power of log(1/Q), giving the same strict
integrability boundary kappa>-a. The code checks it in both the target and
the EWA normalizer. Approaching the boundary concentrates factors near zero;
the log-Q bias also creates a singular force, so strong left power bias can
be difficult for HMC even though it is normalizable.

For ReLU, inactive regions can give a positive atom p0 at Q=0. Then every
negative power has infinite normalization. Even when a particular data set
has no atom, near-zero behavior can depend on its rank, activation pattern
and depth; the central-EWA inverse-moment condition is not a proof for the
actual network. Negative powers are therefore rejected for both ReLU and
Erf until their inverse moments are separately justified. This restriction
does not assert that every negative Erf power is improper.

### Negative roots: the general extension

For theta<0,

\[
0<e^{N\theta Q^{1/L}}\le1,
\]

so normalization exists for every nonnegative Q law, with or without an atom.
The example uses theta=L/2(1-m^(-1/L)) as a large-N central-EWA guide. This
is not an exact finite-N or nonlinear mean calibration.

The boundary geometry still matters. In a locally nondegenerate transverse
direction, Q is of order r^2, so the root contribution to the force scales
as r^(2/L-1). At L=2 there is a cusp; at L>2 it can diverge near zero. Deep
ReLU nets additionally have inactive regions and activation boundaries. The
implementation assigns zero bias force inside a flat Q=0 region, without
clipping positive Q or changing the target. Strong tilts may require smaller
steps and more warmup; inspect divergences, ESS, layer observables and
transitions into/out of zero regions. An observed zero indicator is included
in the example's chain diagnostics when both zero and positive states occur.
These curvature concerns are standard HMC limitations; see the
[Stan diagnostic discussion](https://mc-stan.org/docs/cmdstan-guide/diagnose_utility.html).

If boundary stiffness becomes the bottleneck, a useful next option is the
bounded weight exp(-N*tau*Q), tau>0: it avoids the extra fractional-power
cusp and factorizes over last-layer units conditional on previous layers.
A softened negative power (Q+epsilon)^kappa is also normalizable, but adds
a scale epsilon and needs its own normalizer. Neither additional target is
implemented here; the API retains the two power/root families.

### Stable negative-root EWA normalization

Let a=N/2, Y=Q^(1/L), and theta<0. Mellin inversion of exp(-x), together with
E[Y^(-s)]=a^s[Gamma(a-s/L)/Gamma(a)]^L, gives

\[
Z(\theta)=\frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty}
\Gamma(s)\left[\frac{\Gamma(a-s/L)}{\Gamma(a)}\right]^L
(2|\theta|)^{-s}\,ds,\qquad 0<c<La.
\]

Choose c at the real saddle, solving
psi(c)-psi(a-c/L)=log(2|theta|). Integrate the real part for s=c+it, t>=0,
with factor 1/pi, scaling out the real log-integrand at c. This gives a
one-dimensional calculation for any depth and avoids underflow when Z is
tiny. It uses SciPy's `brentq`, complex `loggamma` and
[`quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).
At L=1, log Z=-(N/2)log(1-2theta) directly. The integration error estimate is
checked against rtol; failure raises an error. This numerical tolerance is
distinct from floating-point roundoff and EWA approximation error.

An alternating moment expansion is unsuitable as a general left-tail
normalizer: it can have severe cancellation and does not converge at all
negative strengths, although the actual Laplace transform always exists.

### Tail probabilities and zero mass

For a left-oriented ensemble the direct estimate is

\[
\Pr_0(Q\le x)=Z_b E_b[e^{-b(Q)}1\{Q\le x\}],\quad x\ge0.
\]

`estimate_left_probability(q, thresholds, tilt=..., width=N, layers=L,
log_normalizer=...)` returns log probabilities, their chain-aware MCSE and
local support checks. It uses no KDE, inverse-weight self-normalization or
modal shift. The example saves estimates at zero and the requested mean
guides in `report.json`. A threshold with no observed events remains
unresolved: its MCSE is infinite (JSON null), not a claimed zero error.
Positive-power right-tail ensembles can have poor or infinite-variance
inverse weights for this purpose; use left-oriented windows for left tails.

For a root tilt, b(0)=0 and Pr_b(Q=0)=p0/Z_b. As theta tends to -infinity,
Z_b tends to p0 if an atom exists. The central EWA has **no zero atom**, so
its normalizer can become especially inaccurate in this regime. The method
samples the actual tilted target, but EWA-normalized densities/probabilities
remain approximations; a resolved atom or disagreement between overlapping
windows calls for true-normalizer estimation before claiming quantitative
tail probabilities. The plotted positive density and the atom are different
objects and cannot be represented by a single density curve at zero.

The example retains a common bulk reference: the highest prior KDE value on
its finite evaluation grid. This is a plotting offset, not a claim that the
true positive density has a finite global maximum. A density can diverge
near zero, especially in mixtures of ReLU activation patterns. For probability
calculations use the unshifted reweighted estimator above; it never uses this
plotting offset.

## Methods paragraph

We sampled all hidden weights jointly in whitened Gaussian coordinates from
the tilted prior using ordinary Pyro HMC and PyTorch automatic differentiation.
Step size and diagonal momentum mass adapted during discarded warmup and were
fixed during production. Retained states included repeated rejections.
Independent chains were assessed using rank-normalized split Rhat, bulk/tail
ESS and divergences, including intermediate-layer observables. We reconstructed
prior density rates by removing the known bias from log-projection density
estimates; the central EWA supplied an explicitly approximate normalizer for
nonlinear networks.

See [the machinery review](sampling_review.md) for implementation boundaries
and proposed improvements.
The [validation record](validation/tilted_prior_sampling.md) lists the measured
example runtime, diagnostic outcomes, automated checks and existing unrelated
test failures.
