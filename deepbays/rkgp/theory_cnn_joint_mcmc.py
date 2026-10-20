"""Gaussian-factor MCMC for the finite-width joint/path CNN integral.

pCN proposals preserve the independent Gaussian innovation measure exactly.
Only the conditional evidence enters their Metropolis ratio; an SPD rate or
Wishart determinant must not be added. Singular channel Grams are supported.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import chi, norm

from ..conv_geometry import positive_int
from ..kernels.conv_kernels import image_batch
from ..sampling.diagnostics import diagnostics_and_mcse
from ._softmax_laplace import gaussian_softmax_statistics
from .theory_cnn_joint_finite_width import CNNJointFiniteWidth


def _diagnostics(observables, *, min_ess, max_rhat, min_chains):
    """Exclude exactly deterministic coordinates, recording that exclusion."""
    variable, constants, masks = {}, {}, {}
    for name, samples in observables.items():
        samples = np.asarray(samples, dtype=float)
        flattened = samples.reshape(*samples.shape[:2], -1)
        # Unlike a variance around a rounded mean, peak-to-peak is exactly
        # zero for repeated non-binary floating-point constants (e.g. 1/3).
        keep = np.ptp(flattened, axis=(0, 1)) > 0
        masks[name] = keep
        constants[name] = int((~keep).sum())
        if keep.any():
            variable[name] = flattened[..., keep]
    if not variable:
        return dict(passed=False, reason='all monitored coordinates are constant',
                    excluded_constant_coordinates=constants), {}
    report, errors = diagnostics_and_mcse(variable, min_ess=min_ess,
        max_rhat=max_rhat, min_chains=min_chains)
    report['excluded_constant_coordinates'] = constants
    restored = {}
    for name, samples in observables.items():
        values = np.zeros(len(masks[name]))
        if name in errors:
            values[masks[name]] = np.asarray(errors[name]).reshape(-1)
        restored[name] = values.reshape(np.asarray(samples).shape[2:])
    return report, restored


class CNNJointMCMC(CNNJointFiniteWidth):
    """Integrate the same finite-width hierarchy as ``CNNJointFiniteWidth``.

    Independent chains use layer-block or full-factor pCN updates. Step-size
    adaptation stops before retained draws. The retained first-layer Grams are
    sufficient for predictions; only the final Gaussian factors are retained
    for continuation, avoiding a channel-factor history at every draw.

    Fit diagnostics monitor evidence, first-Gram scale, and training predictive
    directions. Prediction diagnostics separately monitor each selected mean,
    marginal second moment and predictive-variance influence. Classification
    additionally monitors centered class logits, rather than only contrast
    coordinates. Neither diagnostic set certifies all modes of the posterior.
    For CE this integrates conditional *Laplace* evidence, not the exact
    crossentropy function integral.
    """

    def __init__(self, model, *, factorization='auto',
                 max_particle_bytes=512*1024**2, max_evaluations=100000,
                 max_prediction_bytes=512*1024**2, prior_coordinates='gaussian'):
        super().__init__(model, factorization=factorization,
                         max_particle_bytes=max_particle_bytes,
                         max_evaluations=max_evaluations)
        self.max_prediction_bytes = positive_int(max_prediction_bytes, 'max_prediction_bytes')
        if prior_coordinates not in ('gaussian', 'bartlett'):
            raise ValueError("prior_coordinates must be 'gaussian' or 'bartlett'")
        self.prior_coordinates = prior_coordinates
        self.gaussian_factor_shapes = self.factor_shapes
        self._bartlett = []
        if prior_coordinates == 'bartlett':
            shapes = []
            for width, columns in self.gaussian_factor_shapes:
                rows, cols = np.triu_indices(columns)
                active = rows < min(width, columns)
                rows, cols = rows[active], cols[active]
                diagonal = rows == cols
                self._bartlett.append((min(width, columns), columns, rows, cols,
                                       diagonal, width-rows[diagonal]))
                shapes.append((len(rows),))
            self.factor_shapes = tuple(shapes)
            self.factor_coordinates = sum(shape[0] for shape in shapes)
        self.grams = self.monitor_samples = self.last_state = None

    @property
    def info(self):
        return dict(super().info, prior_coordinates=self.prior_coordinates,
                    gaussian_factor_shapes=self.gaussian_factor_shapes)

    def _factor_matrices(self, factors):
        if self.prior_coordinates == 'gaussian':
            return factors
        matrices = []
        for latent, (height, columns, rows, cols, diagonal, degrees) in zip(factors, self._bartlett):
            values = np.array(latent, dtype=float, copy=True)
            z = values[diagonal]
            quantile = np.empty_like(z)
            positive = z > 0
            # Survival probabilities retain the upper tail where cdf rounds
            # to one. No tail clipping or altered chi prior is introduced.
            quantile[positive] = chi.isf(norm.sf(z[positive]), degrees[positive])
            quantile[~positive] = chi.ppf(norm.cdf(z[~positive]), degrees[~positive])
            if not np.isfinite(quantile).all() or np.any(quantile <= 0):
                raise FloatingPointError('Bartlett quantile cannot be represented; no prior-tail clipping is allowed')
            values[diagonal] = quantile
            matrix = np.zeros((height, columns))
            matrix[rows, cols] = values
            matrices.append(matrix)
        return matrices

    def _first_gram(self, factors):
        return super()._first_gram(self._factor_matrices(factors))

    def _fingerprint(self, *, legacy=False):
        digest = sha256()
        for value in (self.model.X, self.model.Y):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        if not legacy and self.model.closure == 'linear_physical':
            # Archived Grams live in this support basis. Degenerate SVD bases
            # must not silently be rotated by a different numerical build.
            for value in self.features.factors:
                digest.update(np.ascontiguousarray(value).tobytes())
        settings = self.model._settings
        if not legacy:
            # Cache sizes, batch sizes and saddle resource/rank guards do not
            # change the finite-width probability measure or its predictions.
            settings = {name: value for name, value in settings.items() if name not in
                        ('batch_size', 'max_kernel_bytes', 'kernel_backend',
                         'max_joint_coordinates', 'rank_policy')}
        source = dict(settings=settings,
            likelihood=self.model._evidence_signature(), factorization=self.factorization)
        if self.prior_coordinates != 'gaussian':
            source['prior_coordinates'] = self.prior_coordinates
        digest.update(json.dumps(source,
            sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode())
        return digest.hexdigest()

    def importance(self, *args, **kwargs):
        raise NotImplementedError('Use CNNJointFiniteWidth for importance integration')

    def sample_smc(self, *args, **kwargs):
        raise NotImplementedError('Use CNNJointFiniteWidth for SMC integration')

    def _evaluate(self, factors):
        if self._evaluations >= self.max_evaluations:
            raise RuntimeError('finite-width evidence evaluation budget exceeded')
        self._evaluations += 1
        Q = self._first_gram(factors)
        K = self.features.kernel(Q)
        nll, state = self._conditional(K, moments=False)
        if not np.isfinite(nll):
            raise FloatingPointError('non-finite finite-width log evidence')
        return -float(nll), Q, K, state

    def _monitor(self, factors, evaluated, callback):
        ll, Q, K, state = evaluated
        mu = K @ state.alpha
        targets = self.model.Y.ravel()
        if self._classification:
            targets = self.model.basis[self.model.Y].reshape(-1)
        result = dict(log_evidence=np.array(ll), gram_trace=np.array(np.trace(Q)/len(Q)),
            training_mean_square=np.array(np.mean(mu**2)),
            training_target_alignment=np.array(mu @ targets/max(1, len(mu))))
        if callback is not None:
            extra = callback(Q, K, state)
            if not isinstance(extra, dict) or not extra or set(extra) & set(result):
                raise ValueError('monitor callback must return a nonempty dict with unique names')
            result.update(extra)
        for name, value in result.items():
            result[name] = np.asarray(value, dtype=float)
            if not result[name].size or not np.isfinite(result[name]).all():
                raise ValueError(f'non-finite or empty MCMC observable: {name}')
        return result

    def sample_mcmc(self, draws=2000, *, warmup=500, chains=4, thin=1, seed=0,
                    pcn_step=.3, target_accept=.3, blocking='layer',
                    min_ess=1000., max_rhat=1.01, min_chains=4,
                    monitor=None, initial_state=None, verbose=False):
        """Sample Gaussian factors and retain ``draws`` Grams per chain.

        ``thin`` counts complete sweeps, not likelihood evaluations. Each layer
        update costs one conditional evidence evaluation. ``monitor(Q,K,state)``
        can add training-only predictive observables to the diagnostic gate.
        A failed ESS/Rhat gate is reported and predictions require an explicit
        opt-in; it does not erase chains useful for further investigation.

        ``initial_state`` accepts ``last_state`` from a previous call or saved
        archive, including Gaussian factors and RNG states. To continue with
        fixed production kernels use warmup=0; returned draws are the new batch,
        not an automatically concatenated chain history.
        """
        self._check_source()
        try:
            import arviz  # noqa: F401; fail before an expensive sampling run
        except ImportError as exc:
            raise ImportError('Install optional diagnostics: pip install "deepbays[sampling]"') from exc
        draws, chains, thin = [positive_int(x, name) for x, name in
                              ((draws, 'draws'), (chains, 'chains'), (thin, 'thin'))]
        if isinstance(warmup, (bool, np.bool_)) or int(warmup) != warmup or warmup < 0:
            raise ValueError('warmup must be a nonnegative integer')
        warmup = int(warmup)
        if draws < 4 or chains < 2:
            raise ValueError('MCMC diagnostics require draws>=4 and chains>=2')
        if blocking not in ('layer', 'all'):
            raise ValueError("blocking must be 'layer' or 'all'")
        if not 0 < pcn_step <= 1 or not np.isfinite(pcn_step):
            raise ValueError('pcn_step must be in (0,1]')
        if not 0 < target_accept < 1 or not np.isfinite(target_accept):
            raise ValueError('target_accept must be in (0,1)')
        if not np.isfinite(min_ess) or min_ess <= 0 or min_ess > chains*draws:
            raise ValueError('min_ess must be positive and <= chains*draws')
        if not np.isfinite(max_rhat) or max_rhat < 1:
            raise ValueError('max_rhat must be >=1')
        if isinstance(min_chains, bool) or not isinstance(min_chains, int) or min_chains < 2:
            raise ValueError('min_chains must be an integer >=2')
        groups = [[l] for l in range(self.model.L)] if blocking == 'layer' else [list(range(self.model.L))]
        count = len(groups)
        required = 8*(chains*draws*self.ranks[0]**2+3*chains*self.factor_coordinates)
        if required > self.max_particle_bytes:
            raise MemoryError(f'MCMC retained Grams/factors need about {required/2**20:.1f} MiB; '
                              'reduce draws or increase max_particle_bytes')
        calls = chains*(1+count*(warmup+draws*thin))
        if calls > self.max_evaluations:
            raise ValueError(f'MCMC needs {calls} evidence evaluations, exceeding budget')
        fingerprint = self._fingerprint()
        if initial_state is not None:
            if (initial_state.get('fingerprint') not in (fingerprint, self._fingerprint(legacy=True)) or
                    initial_state.get('blocking') != blocking or
                    len(initial_state.get('factors', ())) != chains):
                raise ValueError('continuation state does not match source/chains/blocking')
        self._factors = self._weights = self.result = None
        self.grams = self.monitor_samples = self.last_state = None
        self._evaluations = 0
        start = perf_counter()
        grams = np.empty((chains, draws, self.ranks[0], self.ranks[0]))
        seeds = np.random.SeedSequence(seed).spawn(chains)
        samples, final_factors, rng_states, final_steps, acceptance = {}, [], [], [], []
        for chain in range(chains):
            rng = np.random.default_rng(seeds[chain])
            if initial_state is None:
                factors = [rng.normal(size=shape) for shape in self.factor_shapes]
                steps = np.full(count, pcn_step, dtype=float)
            else:
                factors = [np.array(z, dtype=float, copy=True) for z in initial_state['factors'][chain]]
                if tuple(z.shape for z in factors) != self.factor_shapes or any(not np.isfinite(z).all() for z in factors):
                    raise ValueError('invalid continuation factors')
                steps = np.array(initial_state['steps'][chain], dtype=float, copy=True)
                if steps.shape != (count,) or np.any((steps <= 0)|(steps > 1)):
                    raise ValueError('invalid continuation pCN steps')
                rng.bit_generator.state = deepcopy(initial_state['rng_states'][chain])
            evaluated = self._evaluate(factors)
            accepted = np.zeros(count, dtype=int)
            proposed = np.zeros(count, dtype=int)
            retained = 0
            for sweep in range(warmup+draws*thin):
                for g, group in enumerate(groups):
                    proposal = list(factors)
                    for layer in group:
                        proposal[layer] = np.sqrt(1-steps[g]**2)*factors[layer] + \
                                          steps[g]*rng.normal(size=factors[layer].shape)
                    candidate = self._evaluate(proposal)
                    accept = bool(np.log(rng.random()) < candidate[0]-evaluated[0])
                    if accept:
                        factors, evaluated = proposal, candidate
                    if sweep < warmup:
                        # Diminishing warmup adaptation; frozen for production.
                        logstep = np.log(steps[g]) + .5/(sweep+10)**.6*(float(accept)-target_accept)
                        steps[g] = np.clip(np.exp(logstep), .001, 1.)
                    else:
                        proposed[g] += 1
                        accepted[g] += accept
                if sweep >= warmup and (sweep-warmup+1) % thin == 0:
                    grams[chain, retained] = evaluated[1]
                    values = self._monitor(factors, evaluated, monitor)
                    if not samples:
                        monitored_bytes = 8*chains*draws*sum(value.size for value in values.values())
                        if required+monitored_bytes > self.max_particle_bytes:
                            raise MemoryError('MCMC retained monitors exceed retained-state memory budget')
                        samples = {name: np.empty((chains, draws)+value.shape) for name, value in values.items()}
                    if set(values) != set(samples) or any(value.shape != samples[name].shape[2:] for name, value in values.items()):
                        raise ValueError('monitor callback changed names or shapes')
                    for name, value in values.items():
                        samples[name][chain, retained] = value
                    retained += 1
                    if verbose and (retained == 1 or retained % max(1, draws//10) == 0):
                        print(f'Joint integral MCMC: chain={chain+1}/{chains}, '
                              f'draw={retained}/{draws}, evaluations={self._evaluations}, '
                              f'elapsed={perf_counter()-start:.1f}s', flush=True)
            final_factors.append([z.copy() for z in factors])
            rng_states.append(deepcopy(rng.bit_generator.state))
            final_steps.append(steps.copy())
            acceptance.append((accepted/np.maximum(proposed, 1)).tolist())
        diagnostics, _ = _diagnostics(samples, min_ess=min_ess, max_rhat=max_rhat, min_chains=min_chains)
        self.grams, self.monitor_samples = grams, samples
        self._factors, self._weights = (), np.full(chains*draws, 1/(chains*draws))
        self.last_state = dict(fingerprint=fingerprint, blocking=blocking,
            factors=final_factors, steps=final_steps, rng_states=rng_states)
        elapsed = perf_counter()-start
        self.result = dict(self.info, method='mcmc', draws=draws, chains=chains,
            warmup=warmup, thin=thin, seed=seed, blocking=blocking,
            acceptance=acceptance, pcn_steps=[z.tolist() for z in final_steps],
            evaluations=self._evaluations, seconds=elapsed,
            seconds_per_evaluation=elapsed/self._evaluations,
            diagnostics=diagnostics, min_ess=min_ess, max_rhat=max_rhat,
            min_chains=min_chains, completed=True, accepted=diagnostics['passed'],
            convergence_certified=False)
        return deepcopy(self.result)

    def _draw_gram(self, index):
        return self.grams.reshape(-1, self.ranks[0], self.ranks[0])[index]

    def predict_statistics(self, Xtest, *, probability_samples=256, seed=0,
                           include_probabilities=True, diagnostic_indices=None,
                           require_diagnostics=True, monitor=None):
        """Average predictions and diagnose mean, moment and variance ESS.

        ``diagnostic_indices=None`` monitors every test example. A caller may
        explicitly select a smaller balanced subset to bound temporary memory;
        the returned diagnostic report records those indices. All examples
        still receive predictions. By default failed fit or prediction checks
        raise. ``require_diagnostics=False`` is for pilots, not publication.
        ``monitor(mu,covariance,classification_stats)`` may add named task
        observables, such as conditional expected MSE or expected accuracy,
        which enter the same ESS/Rhat gate. Classification stats are None
        unless probabilities were requested. Variance MCSE uses the
        delta-method influence of the mixture variance, evaluated as
        ``conditional_variance + (conditional_mean-mixture_mean)**2``.
        Classification also checks centered class-logit means and variances,
        including when probability quadrature is disabled.
        """
        self._check_source()
        if self.grams is None:
            raise RuntimeError('complete MCMC integration before predicting')
        if require_diagnostics and not self.result['accepted']:
            raise RuntimeError('MCMC fit ESS/Rhat gate failed; extend independent chains')
        Xtest = image_batch(Xtest)
        n, c = len(Xtest), self.model.c
        indices = np.arange(n) if diagnostic_indices is None else np.asarray(diagnostic_indices)
        if (indices.ndim != 1 or len(indices) == 0 or not np.all(indices == indices.astype(int))
                or np.any(indices < 0) or np.any(indices >= n)):
            raise ValueError('diagnostic_indices must be valid nonempty image indices')
        indices = indices.astype(int)
        # Retain three contrast traces (mean, second moment, variance
        # influence), plus two class-logit traces for classification. Reserve
        # another copy for _diagnostics' variable-coordinate workspaces and
        # the temporary arrays used to construct centered influences.
        class_coordinates = self.model.D if self._classification else 0
        trace_coordinates = 3*c+2*class_coordinates
        required = 2*8*len(self._weights)*len(indices)*trace_coordinates
        if required > self.max_prediction_bytes:
            raise MemoryError(f'predictive diagnostic traces need {required/2**20:.1f} MiB; '
                              'select diagnostic_indices or raise max_prediction_bytes')
        if self._classification and include_probabilities:
            probability_samples = positive_int(probability_samples, 'probability_samples')
            if probability_samples & (probability_samples-1):
                raise ValueError('probability_samples must be a power of two')
        prepared = self.model.builder.prepare(Xtest)
        cached = self._prediction_features(prepared)
        mean = np.zeros((n, c)); within = np.zeros((n, c, c)); between = np.zeros_like(within)
        means = np.empty((len(self._weights), len(indices), c)); seconds = np.empty_like(means)
        class_means = class_influences = None
        if self._classification:
            class_means = np.empty((len(self._weights), len(indices), self.model.D))
            class_influences = np.empty_like(class_means)
        probability_mean = probability_second = winners = None
        extra_samples = {}
        reserved_names = {'latent_mean', 'latent_second_moment', 'latent_variance_influence',
                          'class_logit_mean', 'class_logit_variance_influence'}
        if self._classification and include_probabilities:
            probability_mean = np.zeros((n, self.model.D)); probability_second = np.zeros_like(probability_mean)
            winners = np.zeros_like(probability_mean)
        for i in range(len(self._weights)):
            Q = self._draw_gram(i)
            _, state = self._conditional(self.features.kernel(Q))
            mu, covariance = self._component(prepared, Q, state, cached)
            delta = mu-mean
            mean += delta/(i+1)
            between += i/(i+1)*np.einsum('ma,mb->mab', delta, delta)
            within += covariance
            means[i] = mu[indices]
            # Initially retain conditional variances. Centered influences are
            # formed before converting this array to raw second moments, so
            # a large shared predictive mean does not cause cancellation.
            seconds[i] = np.diagonal(covariance, axis1=1, axis2=2)[indices]
            if class_means is not None:
                class_means[i] = means[i] @ self.model.basis.T
                class_influences[i] = np.einsum('ai,mij,aj->ma', self.model.basis,
                    covariance[indices], self.model.basis, optimize=True)
            stats = None
            if probability_mean is not None:
                stats = gaussian_softmax_statistics(mu, covariance, self.model.basis,
                    samples=probability_samples, seed=seed+i)
                probability_mean += stats['probabilities']
                probability_second += stats['probability_variance']+stats['probabilities']**2
                winners += stats['argmax_probabilities']
            if monitor is not None:
                extra = monitor(mu, covariance, stats)
                if (not isinstance(extra, dict) or not extra or
                        set(extra) & reserved_names):
                    raise ValueError('prediction monitor must return a nonempty dict with unique names')
                extra = {name: np.asarray(value, dtype=float) for name, value in extra.items()}
                if any(not value.size or not np.isfinite(value).all() for value in extra.values()):
                    raise ValueError('non-finite or empty prediction observable')
                if not extra_samples:
                    if required+2*8*len(self._weights)*sum(value.size for value in extra.values()) > self.max_prediction_bytes:
                        raise MemoryError('prediction monitors exceed diagnostic trace memory budget')
                    extra_samples = {name: np.empty((len(self._weights),)+value.shape)
                                     for name, value in extra.items()}
                if set(extra) != set(extra_samples) or any(value.shape != extra_samples[name].shape[1:]
                                                          for name, value in extra.items()):
                    raise ValueError('prediction monitor changed names or shapes')
                for name, value in extra.items():
                    extra_samples[name][i] = value
        count = len(self._weights)
        shape = (self.result['chains'], self.result['draws'], len(indices), c)
        # For V=E[s]-E[m]^2, the influence is s-2*mbar*m. Adding
        # mbar^2 leaves ESS and MCSE unchanged and permits this stable form.
        influences = seconds+(means-mean[indices])**2
        seconds += means**2
        observables = dict(latent_mean=means.reshape(shape), latent_second_moment=seconds.reshape(shape),
                           latent_variance_influence=influences.reshape(shape))
        if class_means is not None:
            class_shape = (self.result['chains'], self.result['draws'], len(indices), self.model.D)
            class_influences += (class_means-mean[indices] @ self.model.basis.T)**2
            observables.update(class_logit_mean=class_means.reshape(class_shape),
                               class_logit_variance_influence=class_influences.reshape(class_shape))
        observables.update({name: value.reshape(self.result['chains'], self.result['draws'], *value.shape[1:])
                            for name, value in extra_samples.items()})
        diagnostics, errors = _diagnostics(observables, min_ess=self.result['min_ess'],
            max_rhat=self.result['max_rhat'], min_chains=self.result['min_chains'])
        diagnostics['example_indices'] = indices.tolist()
        result = dict(latent_mean=mean, latent_covariance=(within+between)/count,
            conditional_covariance=within/count, between_covariance=between/count,
            prediction_diagnostics=diagnostics, accepted=bool(self.result['accepted'] and diagnostics['passed']),
            monitored_mean_mcse=errors.get('latent_mean'),
            monitored_second_moment_mcse=errors.get('latent_second_moment'),
            monitored_variance_mcse=errors.get('latent_variance_influence'),
            latent_mean_mcse=errors.get('latent_mean') if np.array_equal(indices, np.arange(n)) else None,
            latent_variance_mcse=errors.get('latent_variance_influence') if np.array_equal(indices, np.arange(n)) else None,
            predictive_observable_means={name: value.mean(axis=0) for name, value in extra_samples.items()},
            predictive_observable_mcse={name: errors[name] for name in extra_samples})
        if class_means is not None:
            result.update(monitored_class_logit_mean_mcse=errors.get('class_logit_mean'),
                monitored_class_logit_variance_mcse=errors.get('class_logit_variance_influence'),
                class_logit_mean_mcse=errors.get('class_logit_mean') if np.array_equal(indices, np.arange(n)) else None,
                class_logit_variance_mcse=errors.get('class_logit_variance_influence') if np.array_equal(indices, np.arange(n)) else None)
        if probability_mean is not None:
            probability_mean /= count
            result.update(probabilities=probability_mean,
                probability_variance=np.maximum(probability_second/count-probability_mean**2, 0.),
                argmax_probabilities=winners/count)
        if require_diagnostics and not diagnostics['passed']:
            raise RuntimeError('MCMC predictive mean/moment/variance ESS/Rhat gate failed; extend chains')
        return result

    def save(self, path):
        """Save retained Grams, monitors, and exact continuation state as NPZ.

        No pickle objects are used. The preprocessed source model itself is not
        saved; ``load`` verifies its data, geometry and likelihood fingerprint.
        """
        self._check_source()
        if self.grams is None or self.last_state is None:
            raise RuntimeError('complete MCMC integration before saving')
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('archive path must end in .npz')
        if path.exists():
            raise FileExistsError(path)
        arrays = dict(grams=self.grams)
        for name, samples in self.monitor_samples.items():
            arrays[f'monitor_{name}'] = samples
        for chain, factors in enumerate(self.last_state['factors']):
            for layer, z in enumerate(factors):
                arrays[f'factor_{chain}_{layer}'] = z
        state = {name: value for name, value in self.last_state.items() if name != 'factors'}
        metadata = dict(format=2, result=self.result, state=state,
                        monitor_names=list(self.monitor_samples))
        arrays['metadata'] = np.array(json.dumps(metadata, default=lambda x: np.asarray(x).tolist()))
        # Exclusive creation prevents accidental overwrite of another run.
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        return path

    def load(self, path):
        """Load an archive for predictions or ``initial_state`` continuation."""
        self._check_source()
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            allowed_fingerprints = (self._fingerprint(), self._fingerprint(legacy=True))
            if meta.get('format') not in (1, 2) or meta['state']['fingerprint'] not in allowed_fingerprints:
                raise ValueError('archive does not match source data/geometry/likelihood')
            result = meta['result']
            shape = (result['chains'], result['draws'], self.ranks[0], self.ranks[0])
            required = 8*(np.prod(shape)+result['chains']*self.factor_coordinates)
            if required > self.max_particle_bytes:
                raise MemoryError('archive exceeds retained-state memory budget')
            grams = np.array(archive['grams'])
            if grams.shape != shape or not np.isfinite(grams).all():
                raise ValueError('invalid archived Grams')
            monitors = {name: np.array(archive[f'monitor_{name}']) for name in meta['monitor_names']}
            if (any(value.shape[:2] != shape[:2] or not np.isfinite(value).all()
                    for value in monitors.values()) or not monitors):
                raise ValueError('invalid archived diagnostic histories')
            if required+sum(value.nbytes for value in monitors.values()) > self.max_particle_bytes:
                raise MemoryError('archive diagnostic histories exceed retained-state memory budget')
            factors = [[np.array(archive[f'factor_{chain}_{layer}']) for layer in range(self.model.L)]
                       for chain in range(result['chains'])]
            if any(tuple(z.shape for z in rows) != self.factor_shapes or
                   any(not np.isfinite(z).all() for z in rows) for rows in factors):
                raise ValueError('invalid archived continuation factors')
        self.grams, self.monitor_samples, self.result = grams, monitors, result
        self.last_state = dict(meta['state'], factors=factors)
        self._factors, self._weights = (), np.full(result['chains']*result['draws'],
            1/(result['chains']*result['draws']))
        return deepcopy(self.result)
