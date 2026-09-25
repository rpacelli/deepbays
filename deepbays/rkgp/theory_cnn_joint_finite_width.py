"""Finite-width integration of the frozen-kernel CNN hierarchy.

This opt-in reference integrates Gaussian factors, including singular
Wisharts and their orientations. It is not another SPD rate saddle.
Gaussian evidence is analytic; CE retains conditional function-space Laplace.
"""
from copy import deepcopy
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from ..conv_geometry import positive_int
from ..kernels.conv_kernels import image_batch
from ..kernels.multioutput import contract_blocks, contract_self
from ._cnn_joint_model import JointCNNModel
from ._cnn_joint_rate import RepeatedMap, sym
from ._softmax_laplace import fit_laplace, gaussian_softmax_statistics


def _psd_root(A):
    values, vectors = np.linalg.eigh(sym(A))
    tolerance = 64*np.finfo(float).eps*len(values)*float(abs(values).max())
    if values.min() < -tolerance:
        raise FloatingPointError('finite-width backward scale is not PSD')
    # Roundoff eigenvalues on an exactly singular support must not turn into
    # spurious O(sqrt(eps)) square-root directions. No positive jitter is added.
    return (vectors*np.sqrt(np.where(values > tolerance, values, 0.))) @ vectors.T


class CNNJointFiniteWidth:
    """Integrate the finite-width prior underlying a preprocessed joint model.

    Reuses its frozen training features, widths, targets, and likelihood.
    Neither its optimized state nor network MCMC results enter this method.
    The source model must stay at the same data and likelihood settings.

    ``importance`` gives independent-prior importance estimates and ESS/MCSE.
    ``sample_smc`` tempers the evidence using resampling and prior-preserving
    Gaussian pCN moves. Completion does not certify SMC convergence; compare
    independent seeds and larger populations/move counts.

    ``thin_path`` is an exact rectangular-factor construction for repeated
    path maps, with no covariance eigenvalue truncation. General physical
    gathers use ``full_innovation`` with PSD square roots. Neither requires
    a sampled innovation to have full rank. No determinants of singular
    empirical Grams, pseudo-determinants, or activation updates are used.
    """

    def __init__(self, model, *, factorization='auto',
                 max_particle_bytes=512*1024**2, max_evaluations=100000):
        if not isinstance(model, JointCNNModel):
            raise TypeError('model must be a scalar or multi-output joint CNN theory')
        model._require_ready()
        if factorization not in ('auto', 'thin_path', 'full_innovation'):
            raise ValueError('invalid factorization')
        repeated = all(isinstance(b, RepeatedMap) for b in model.rate.maps)
        if factorization == 'auto':
            factorization = 'thin_path' if repeated else 'full_innovation'
        if factorization == 'thin_path' and not repeated:
            raise ValueError('thin_path requires repeated path maps')
        self.model, self.features = model, model.features
        self.factorization = factorization
        self.widths, self.ranks = tuple(model.widths), tuple(model.rate.ranks)
        self.max_particle_bytes = positive_int(max_particle_bytes, 'max_particle_bytes')
        self.max_evaluations = positive_int(max_evaluations, 'max_evaluations')
        self._signature = model._evidence_signature()
        self._targets = model.Y
        self._classification = hasattr(model, 'basis')
        if not self._classification and model.T <= 0:
            raise ValueError('finite-width Gaussian integration requires positive T')
        columns = list(self.ranks)
        if factorization == 'thin_path':
            for layer in range(model.L-2, -1, -1):
                columns[layer] = model.rate.maps[layer].m * min(
                    self.widths[layer+1], columns[layer+1])
        self.factor_shapes = tuple(zip(self.widths, columns))
        self.factor_coordinates = sum(n*q for n, q in self.factor_shapes)
        self._factors = self._weights = None
        self.result = None

    def _check_source(self):
        m = self.model
        if (not m._ready or m.features is not self.features or m.Y is not self._targets
                or tuple(m.widths) != self.widths or m._evidence_signature() != self._signature):
            raise RuntimeError('source data/likelihood changed; construct a new finite-width model')

    @property
    def info(self):
        self._check_source()
        return dict(inference='finite_width_mixture', factorization=self.factorization,
                    factor_shapes=self.factor_shapes, factor_coordinates=self.factor_coordinates,
                    joint_dimensions=self.ranks, widths=self.widths,
                    closure=self.model.closure,
                    conditional_evidence='Laplace' if self._classification else 'Gaussian',
                    nonlinear_closure=self.model.joint_info['nonlinear_closure'])

    def _first_gram(self, factors):
        if self.factorization == 'thin_path':
            B = None
            for layer in range(self.model.L-1, -1, -1):
                E = factors[layer]
                # QR removes row rotations while preserving E.T @ E exactly.
                R = np.linalg.qr(E, mode='r') if len(E) > E.shape[1] else E
                if B is None:
                    B = R/np.sqrt(self.widths[layer])
                else:
                    mapping = self.model.rate.maps[layer]
                    B = (R.reshape(len(R), mapping.m, len(B)) @ B).reshape(
                        len(R), self.ranks[layer])/np.sqrt(self.widths[layer])
            return sym(B.T @ B)
        Q = None
        for layer in range(self.model.L-1, -1, -1):
            Z = factors[layer]
            if Q is None:
                dressed = Z
            else:
                mapping = self.model.rate.maps[layer]
                if isinstance(mapping, RepeatedMap):
                    dressed = (Z.reshape(len(Z), mapping.m, mapping.upper) @
                               _psd_root(Q)).reshape(Z.shape)
                else:
                    dressed = Z @ _psd_root(mapping.apply(Q))
            Q = dressed.T @ dressed/self.widths[layer]
        return sym(Q)

    def _conditional(self, K, *, moments=True):
        m = self.model
        if self._classification:
            state = fit_laplace(K, m.Y, m.basis, beta=m.beta,
                mode_tol=m.mode_tol, maxiter=m.mode_maxiter, max_dense_size=m.max_dense_size)
            return state.nll, state
        factor = cho_factor(K+m.T*np.eye(len(K)), lower=True, check_finite=False)
        alpha = cho_solve(factor, m.Y.ravel(), check_finite=False)
        nll = np.log(np.diag(factor[0])).sum()+.5*m.Y.ravel() @ alpha
        reduction = cho_solve(factor, np.eye(len(K)), check_finite=False) if moments else None
        return float(nll), SimpleNamespace(alpha=alpha, reduction=reduction)

    def _log_likelihood(self, factors):
        if self._evaluations >= self.max_evaluations:
            raise RuntimeError('finite-width evidence evaluation budget exceeded')
        self._evaluations += 1
        value = -self._conditional(self.features.kernel(self._first_gram(factors)), moments=False)[0]
        if not np.isfinite(value):
            raise FloatingPointError('non-finite finite-width log evidence')
        return value

    def _initialize(self, count, seed, copies):
        self._check_source()
        count = positive_int(count, 'particle count')
        if count > self.max_evaluations:
            raise ValueError('particle count exceeds evidence evaluation budget')
        required = copies*8*count*self.factor_coordinates
        if required > self.max_particle_bytes:
            raise MemoryError(f'particle arrays/workspace need about {required/2**20:.1f} MiB; '
                              'reduce particle count or increase max_particle_bytes')
        self._factors = self._weights = self.result = None
        self._evaluations = 0
        rng = np.random.default_rng(seed)
        factors = [rng.normal(size=(count, n, q)) for n, q in self.factor_shapes]
        ll = np.array([self._log_likelihood([z[i] for z in factors]) for i in range(count)])
        return rng, factors, ll

    def importance(self, draws=2048, *, seed=0, min_ess=100.):
        """Independent-prior importance integration; refuses low-ESS predictions.

        The ESS threshold is a guard, not a guarantee against unobserved tails.
        Gaussian log evidence omits the common -P*c*log(2*pi)/2 constant,
        matching the existing Gaussian joint model.
        """
        draws = positive_int(draws, 'draws')
        min_ess = float(min_ess)
        if not np.isfinite(min_ess) or min_ess <= 0 or min_ess > draws:
            raise ValueError('min_ess must be positive and no greater than draws')
        start = perf_counter()
        _, factors, ll = self._initialize(draws, seed, copies=1)
        logsum = logsumexp(ll)
        weights = np.exp(ll-logsum)
        ess = float(1/(weights @ weights))
        self.result = dict(self.info, method='importance', draws=len(ll), seed=seed,
            ess=ess, maximum_weight=float(weights.max()), log_evidence=float(logsum-np.log(len(ll))),
            log_evidence_mcse=float(np.sqrt(max(1/ess-1/len(ll), 0.))),
            evaluations=self._evaluations, seconds=perf_counter()-start,
            completed=True, accepted=bool(ess >= min_ess), convergence_certified=False)
        if ess < min_ess:
            raise RuntimeError(f'importance ESS {ess:.1f} < {min_ess:g}; predictions disabled; '
                               'increase draws or use SMC with independent validation runs')
        self._factors, self._weights = factors, weights
        return deepcopy(self.result)

    def sample_smc(self, particles=256, *, moves=8, seed=0, pcn_step=.25,
                   ess_fraction=.8, max_stages=100, verbose=False):
        """Annealed SMC with Gaussian-prior pCN moves and adaptive temperatures.

        Each stage resamples, then applies ``moves`` full-factor pCN proposals.
        Evidence failures abort rather than being silently treated as rejections.
        Reported ESS is BEFORE resampling. No independent-particle MCSE is
        reported for correlated SMC output; compare independent runs instead.
        """
        moves, max_stages = positive_int(moves, 'moves'), positive_int(max_stages, 'max_stages')
        if not np.isfinite(pcn_step) or not 0 < pcn_step <= 1:
            raise ValueError('pcn_step must be in (0,1]')
        if not np.isfinite(ess_fraction) or not 0 < ess_fraction < 1:
            raise ValueError('ess_fraction must be in (0,1)')
        start = perf_counter()
        rng, factors, ll = self._initialize(particles, seed, copies=3)
        count, temperature, logZ = len(ll), 0., 0.
        ancestors, stages = np.arange(count), []
        step = float(pcn_step)
        while temperature < 1:
            if len(stages) >= max_stages:
                raise RuntimeError('SMC stage limit reached before the full likelihood; predictions disabled')
            def ess(delta):
                w = np.exp(delta*(ll-ll.max())-logsumexp(delta*(ll-ll.max())))
                return float(1/(w @ w))
            remaining = 1-temperature
            if ess(remaining) >= ess_fraction*count:
                delta = remaining
            else:
                lo, hi = 0., remaining
                for _ in range(45):
                    middle = (lo+hi)/2
                    if ess(middle) < ess_fraction*count:
                        hi = middle
                    else:
                        lo = middle
                delta = lo
            if delta < 1e-12:
                raise RuntimeError('SMC temperature stalled; predictions disabled')
            normalizer = logsumexp(delta*ll)
            logZ += normalizer-np.log(count)
            weights = np.exp(delta*ll-normalizer)
            temperature = min(1., temperature+delta)
            positions = (rng.random()+np.arange(count))/count
            indices = np.minimum(np.searchsorted(np.cumsum(weights), positions), count-1)
            factors = [z[indices].copy() for z in factors]
            ll, ancestors = ll[indices], ancestors[indices]
            accepted = 0
            for _ in range(moves):
                for i in range(count):
                    proposals = [rng.normal(size=z.shape[1:]) for z in factors]
                    for z, p in zip(factors, proposals):
                        p *= step
                        p += np.sqrt(1-step**2)*z[i]
                    candidate = self._log_likelihood(proposals)
                    if np.log(rng.random()) < temperature*(candidate-ll[i]):
                        for z, p in zip(factors, proposals):
                            z[i] = p
                        ll[i] = candidate
                        accepted += 1
            acceptance = accepted/(moves*count)
            stages.append(dict(temperature=temperature, pre_resampling_ess=float(1/(weights @ weights)),
                acceptance=acceptance, pcn_step=step,
                surviving_prior_ancestors=int(len(np.unique(ancestors)))))
            if verbose:
                print(f'Finite-width SMC: {stages[-1]}', flush=True)
            step = float(np.clip(step*np.exp(acceptance-.3), .01, .5))
        self._factors, self._weights = factors, np.full(count, 1/count)
        self.result = dict(self.info, method='smc', particles=count, moves=moves, seed=seed,
            log_evidence=float(logZ), stages=stages, evaluations=self._evaluations,
            seconds=perf_counter()-start, completed=True, accepted=True,
            convergence_certified=False, mcse=None)
        return deepcopy(self.result)

    def _prediction_features(self, prepared):
        """Cache small fixed cross features once across the particle ensemble."""
        f = getattr(self.features, 'spatial', self.features)
        n, r = len(prepared[0]), f.ranks[0]
        if 8*n*(f.n+1)*r*r > f.budget:
            return None
        cross = np.empty((n, f.n, r, r))
        for a, b in f._pairs(n, f.n):
            cross[a, b] = f.block(f._subset(prepared, a), f._subset(f.prepared, b))
        diagonal = np.empty((n, r, r))
        for start in range(0, n, f.batch_size):
            a = slice(start, min(start+f.batch_size, n))
            diagonal[a] = f.block(f._subset(prepared, a), paired=True)
        return cross, diagonal

    def _component(self, prepared, Q, state, cached=None):
        c, n = self.model.c, len(prepared[0])
        if cached is None:
            cross = self.features.cross(prepared, Q)
            prior = self.features.diagonal(prepared, Q)
        else:
            cross = contract_blocks(cached[0], Q, c, 1.)
            prior = contract_self(cached[1], Q, c, 1.)
            if c == 1:
                prior = prior[:, 0, 0]
        mean = (cross @ state.alpha).reshape(n, c)
        if c == 1:
            covariance = (prior-np.sum((cross @ state.reduction)*cross, axis=1))[:, None, None]
        else:
            covariance = prior-np.einsum('mai,mbi->mab',
                (cross @ state.reduction).reshape(n, c, -1), cross.reshape(n, c, -1))
        covariance = (covariance+covariance.swapaxes(-1, -2))/2
        values, vectors = np.linalg.eigh(covariance)
        if values.min() < -1e-9*max(1., float(abs(prior).max())):
            raise FloatingPointError('negative conditional predictive covariance')
        covariance = (vectors*np.maximum(values, 0.)[:, None, :]) @ vectors.swapaxes(-1, -2)
        return mean, covariance

    def predict_statistics(self, Xtest, *, probability_samples=256, seed=0,
                           include_probabilities=True):
        """Mixture moments, including uncertainty of conditional means.

        Classification probabilities are averaged component by component;
        the finite-width posterior is NOT replaced by one moment-matched GP.
        ``latent_mean_mcse`` is a self-normalized importance estimate only;
        it is None for SMC and excludes conditional CE Laplace error.
        """
        self._check_source()
        if self._factors is None:
            raise RuntimeError('complete accepted finite-width integration before predicting')
        Xtest = image_batch(Xtest)
        prepared = self.model.builder.prepare(Xtest)
        cached = self._prediction_features(prepared)
        n, c = len(Xtest), self.model.c
        mean = np.zeros((n, c))
        within = np.zeros((n, c, c)); between = np.zeros_like(within)
        square_mean = np.zeros_like(mean); square_second = np.zeros_like(mean)
        total = 0.
        probability_mean = probability_second = winners = None
        if self._classification and include_probabilities:
            # Validate before any expensive evidence evaluations.
            probability_samples = positive_int(probability_samples, 'probability_samples')
            if probability_samples & (probability_samples-1):
                raise ValueError('probability_samples must be a power of two')
            probability_mean = np.zeros((n, self.model.D))
            probability_second = np.zeros_like(probability_mean)
            winners = np.zeros_like(probability_mean)
        for i, weight in enumerate(self._weights):
            if weight == 0:
                continue
            Q = self._first_gram([z[i] for z in self._factors])
            _, state = self._conditional(self.features.kernel(Q))
            mu, covariance = self._component(prepared, Q, state, cached)
            delta = mu-mean
            updated = total+weight
            mean += weight/updated*delta
            between += weight*total/updated*np.einsum('ma,mb->mab', delta, delta)
            within += weight*covariance
            square_mean += weight**2*mu
            square_second += weight**2*mu**2
            total = updated
            if self._classification and include_probabilities:
                stats = gaussian_softmax_statistics(mu, covariance, self.model.basis,
                    samples=probability_samples, seed=seed+i)
                probability_mean += weight*stats['probabilities']
                probability_second += weight*(stats['probability_variance']+stats['probabilities']**2)
                winners += weight*stats['argmax_probabilities']
        result = dict(latent_mean=mean, latent_covariance=(within+between)/total,
                      conditional_covariance=within/total, between_covariance=between/total)
        if self.result['method'] == 'importance':
            variance = square_second-2*mean*square_mean+mean**2*np.sum(self._weights**2)
            result['latent_mean_mcse'] = np.sqrt(np.maximum(variance, 0.))/total
        else:
            result['latent_mean_mcse'] = None
        if self._classification and include_probabilities:
            probability_mean /= total
            result.update(probabilities=probability_mean,
                probability_variance=np.maximum(probability_second/total-probability_mean**2, 0.),
                argmax_probabilities=winners/total)
        return result

    def predict_latent(self, Xtest):
        """Return the finite-width mixture's mean and per-example covariance."""
        stats = self.predict_statistics(Xtest, include_probabilities=False)
        return stats['latent_mean'], stats['latent_covariance']

    def predict_proba(self, Xtest, *, samples=256, seed=0):
        if not self._classification:
            raise ValueError('probabilities require a classification source model')
        return self.predict_statistics(Xtest, probability_samples=samples, seed=seed)['probabilities']
