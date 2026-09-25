"""Reduced product approximation for CNN predictive means.

The terminal spatial readout is fully included in K_IW. A fixed output
basis, selected from training IW duals, leaves at most c scalar products.
No spatial eigenvectors or per-layer SPD matrices are fitted here.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from ..conv_geometry import positive_int
from ..kernels.conv_kernels import image_batch
from ._cnn_joint_model import JointCNNModel
from ._cnn_joint_rate import RepeatedMap, sym
from ._softmax_laplace import fit_laplace, gaussian_softmax_statistics


def _spatial_adjoint(mapping, blocks):
    """Apply a spatial adjoint to possibly nonsymmetric output-pair blocks."""
    if isinstance(mapping, RepeatedMap):
        shape = blocks.shape[:-2]+(mapping.m, mapping.upper, mapping.m, mapping.upper)
        return np.einsum('...aiaj->...ij', blocks.reshape(shape))
    return np.einsum('tui,...ij,tvj->...uv', mapping.maps, blocks, mapping.maps,
                     optimize=True)


class CNNReducedMean:
    """Product-rate mean theory, optionally paired with joint variances.

    ``model`` is a preprocessed scalar/multi-output Gaussian or CE joint CNN.
    It need not be optimized to fit this reduced model. With the default
    ``covariance='joint'``, prediction requires a selected/converged source
    state; ``predict_mean`` and ``covariance='reduced'`` do not.

    ``finite_width_averaging=False`` selects deterministic product-rate
    optimization. With True, ``fit``/``optimize`` perform independent-prior
    importance integration of reduced gamma products. This is not integration
    of the original joint CNN hierarchy. All basis/rate choices use training
    IW quantities, never test inputs or sampled networks.

    One or many final patches, flattening and average pooling, both joint
    closures, unequal widths, and widths below joint matrix dimensions are
    supported. Source feature/geometry resource limits still apply.
    """

    def __init__(self, model, *, finite_width_averaging=False, covariance='joint',
                 basis_rtol=1e-7, max_projection_bytes=256*1024**2,
                 max_draw_bytes=64*1024**2, max_evaluations=100000):
        if not isinstance(model, JointCNNModel):
            raise TypeError('model must be a preprocessed joint CNN theory')
        model._require_ready()
        if not isinstance(finite_width_averaging, (bool, np.bool_)):
            raise ValueError('finite_width_averaging must be Boolean')
        self._validate_covariance(covariance)
        if not np.isfinite(basis_rtol) or not 0 <= basis_rtol < 1:
            raise ValueError('basis_rtol must be finite and in [0,1)')
        self.model = model
        self.features = model.features
        self.spatial = getattr(model.features, 'spatial', model.features)
        self.widths, self._targets = tuple(model.widths), model.Y
        self._signature = model._evidence_signature()
        self._classification = hasattr(model, 'basis')
        self.c, self.L = model.c, model.L
        self.finite_width_averaging = bool(finite_width_averaging)
        self.covariance = covariance
        self.basis_rtol = float(basis_rtol)
        self.max_projection_bytes = positive_int(max_projection_bytes, 'max_projection_bytes')
        self.max_draw_bytes = positive_int(max_draw_bytes, 'max_draw_bytes')
        self.max_evaluations = positive_int(max_evaluations, 'max_evaluations')
        f = self.spatial
        self.K_IW = f.kernel(np.eye(f.ranks[0]))
        self.K_IW.setflags(write=False)
        _, _, iw = self._conditional(np.kron(self.K_IW, np.eye(self.c)))
        alpha = iw.alpha.reshape(model.P, self.c)
        scale = np.max(abs(alpha), initial=0.)
        normalized = alpha/scale if scale else alpha.copy()
        gram = sym(normalized.T @ self.K_IW @ normalized)
        eigenvalues, basis = np.linalg.eigh(gram)
        tolerance = 64*np.finfo(float).eps*self.c*np.max(abs(eigenvalues), initial=0.)
        if eigenvalues.min() < -tolerance:
            raise FloatingPointError('IW dual Gram is not positive semidefinite')
        self.basis_eigenvalues = np.maximum(eigenvalues, 0.)
        self.output_basis = basis
        # A shared scale in an unresolved eigenspace is rotation invariant.
        threshold = max(self.basis_rtol, 64*np.finfo(float).eps*self.c)*self.basis_eigenvalues.max()
        groups, start = [], 0
        for stop in range(1, self.c+1):
            if stop == self.c or self.basis_eigenvalues[stop]-self.basis_eigenvalues[start] > threshold:
                groups.append(np.arange(start, stop))
                start = stop
        self.groups = tuple(groups)
        self.projectors = np.array([basis[:, g] @ basis[:, g].T for g in groups])
        self.coordinates = len(groups)
        self.eta, self.projection_sources = self._projected_widths(normalized)
        self.effective_widths = self.L/np.sum(self.eta/np.asarray(self.widths), axis=1)
        self.gamma_shapes = np.asarray(self.widths)[None, :]/(2*self.eta)
        for array in (self.output_basis, self.basis_eigenvalues, self.projectors,
                      self.eta, self.effective_widths, self.gamma_shapes):
            array.setflags(write=False)
        self._reset()

    @staticmethod
    def _validate_covariance(covariance):
        if covariance not in ('joint', 'reduced'):
            raise ValueError("covariance must be 'joint' or 'reduced'")

    def _check_source(self):
        m = self.model
        if (not m._ready or m.features is not self.features or m.Y is not self._targets
                or tuple(m.widths) != self.widths or m._evidence_signature() != self._signature):
            raise RuntimeError('source data/likelihood changed; construct a new reduced mean model')

    def _reset(self):
        self._scales = self._weights = None
        self.optQ = self.result = None
        self.converged = False
        self._evaluations = 0

    @property
    def info(self):
        self._check_source()
        return dict(theory='reduced_path_product', finite_width_averaging=self.finite_width_averaging,
            covariance=self.covariance, coordinates=self.coordinates, outputs=self.c,
            group_sizes=[len(g) for g in self.groups], basis_rtol=self.basis_rtol,
            basis_eigenvalues=self.basis_eigenvalues.tolist(),
            eta=self.eta.tolist(), effective_widths=self.effective_widths.tolist(),
            gamma_shapes=self.gamma_shapes.tolist(), projection_sources=self.projection_sources,
            widths=self.widths, spatial_ranks=self.spatial.ranks,
            final_patches=self.spatial.builder.d, pooling=self.spatial.pooling,
            closure=self.model.closure,
            conditional_evidence='Laplace' if self._classification else 'Gaussian',
            rate_normalization='F = negative log evidence + width-weighted product penalty',
            approximation='training-IW output reduction; not an exact marginal of joint CNNs')

    def _conditional(self, K):
        m = self.model
        if self._classification:
            state = fit_laplace(K, m.Y, m.basis, beta=m.beta, mode_tol=m.mode_tol,
                maxiter=m.mode_maxiter, max_dense_size=m.max_dense_size)
            return state.nll, state.gradient, state
        factor = cho_factor(sym(K)+m.T*np.eye(len(K)), lower=True, check_finite=False)
        alpha = cho_solve(factor, m.Y.ravel(), check_finite=False)
        inverse = cho_solve(factor, np.eye(len(K)), check_finite=False)
        nll = np.log(np.diag(factor[0])).sum()+.5*m.Y.ravel() @ alpha
        state = OptimizeResult(alpha=alpha, reduction=inverse)
        return float(nll), .5*(inverse-np.outer(alpha, alpha)), state

    def _projected_widths(self, alpha):
        f = self.spatial
        required = 3*8*f.ranks[0]**2*sum(len(g)**2 for g in self.groups)
        if required > self.max_projection_bytes:
            raise MemoryError(f'projected IW workspaces need about {required/2**20:.1f} MiB; '
                              'increase max_projection_bytes or reduce geometry/output count')
        probes = [alpha @ self.output_basis[:, g] for g in self.groups]
        trace_scale = max(np.trace(alpha.T @ self.K_IW @ alpha), 0.)
        fallback = [np.trace(a.T @ self.K_IW @ a) <= 128*np.finfo(float).eps*trace_scale
                    for a in probes]
        projected = [np.zeros((len(g), len(g), f.ranks[0], f.ranks[0])) for g in self.groups]
        trace = np.zeros((f.ranks[0], f.ranks[0])) if any(fallback) else None
        for a, b in f._pairs(f.n, f.n, symmetric=True):
            H = f.H[a, b] if f.H is not None else f.block(
                f._subset(f.prepared, a), f._subset(f.prepared, b))
            for j, probe in enumerate(probes):
                if fallback[j]:
                    continue
                value = np.einsum('ma,mnij,nb->abij', probe[a], H, probe[b], optimize=True)
                projected[j] += value if a.start == b.start else value+value.transpose(1, 0, 3, 2)
            if trace is not None and a.start == b.start:
                trace += np.einsum('mmij->ij', H)/f.n
        eta, sources = [], []
        for j, (g, A) in enumerate(zip(self.groups, projected)):
            if fallback[j]:
                A = np.einsum('ab,ij->abij', np.eye(len(g))/len(g), trace)
            terms = []
            first_trace = np.einsum('aaii->', A)
            if not np.isfinite(first_trace) or first_trace <= 0:
                raise ValueError('reduced rate requires nonzero training IW kernel in each projection')
            for layer, rank in enumerate(f.ranks):
                if layer:
                    A = _spatial_adjoint(f.maps[layer-1], A)
                z = np.einsum('aaii->', A)
                if not np.isclose(z, first_trace, rtol=1e-8, atol=0.):
                    raise FloatingPointError('IW projection trace changed through normalized gathers')
                value = float(np.sum(A*A)/z**2)
                lower = 1/(rank*len(g))
                if not np.isfinite(value) or not lower-1e-8 <= value <= 1+1e-8:
                    raise FloatingPointError('projected IW width violates PSD bounds')
                terms.append(float(np.clip(value, lower, 1.)))
            eta.append(terms)
            sources.append('iw_trace: zero dual group' if fallback[j] else 'iw_dual')
        return np.array(eta), sources

    def _Q(self, scales):
        return sym(np.einsum('g,gab->ab', scales, self.projectors))

    def objective_and_gradient(self, log_scales):
        """Unnormalized negative-log-evidence objective, at fixed IW basis."""
        self._check_source()
        x = np.asarray(log_scales, dtype=float)
        if x.shape != (self.coordinates,) or not np.all(np.isfinite(x)):
            raise ValueError('log_scales must be a finite vector with one entry per group')
        with np.errstate(over='raise', invalid='raise', under='raise'):
            scales = np.exp(x)
            prior_slope = np.expm1(x/self.L)
        Q = self._Q(scales)
        nll, gradient, _ = self._conditional(np.kron(self.K_IW, Q))
        output_gradient = np.einsum('mn,manb->ab', self.K_IW,
            gradient.reshape(self.model.P, self.c, self.model.P, self.c))
        prior = .5*np.sum(self.effective_widths*(self.L*prior_slope-x))
        derivative = scales*np.einsum('ab,gab->g', output_gradient, self.projectors)
        derivative += .5*self.effective_widths*prior_slope
        return float(nll+prior), derivative

    def fit(self, *, maxiter=300, gtol=1e-6, draws=4096, seed=0, min_ess=100.):
        """Fit the default saddle, or average products when explicitly enabled."""
        if self.finite_width_averaging:
            return self.average(draws=draws, seed=seed, min_ess=min_ess)
        self._check_source()
        maxiter = positive_int(maxiter, 'maxiter')
        if not np.isfinite(gtol) or gtol <= 0:
            raise ValueError('gtol must be positive and finite')
        self._reset()
        started = perf_counter()
        def objective(x):
            if self._evaluations >= self.max_evaluations:
                raise RuntimeError('reduced objective evaluation budget exceeded; predictions disabled')
            self._evaluations += 1
            return self.objective_and_gradient(x)
        result = minimize(objective, np.zeros(self.coordinates), jac=True, method='L-BFGS-B',
            options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-14, maxls=40))
        result.fun, result.jac = objective(result.x)
        if np.max(abs(result.jac)) > gtol and self.coordinates <= 128:
            refined = minimize(objective, result.x, jac=True, method='BFGS',
                               options=dict(maxiter=maxiter, gtol=gtol))
            refined.fun, refined.jac = objective(refined.x)
            if refined.fun <= result.fun+1e-12 and np.max(abs(refined.jac)) < np.max(abs(result.jac)):
                result = refined
        result.gradient_norm = float(np.max(abs(result.jac)))
        result.converged = bool(np.isfinite(result.fun) and result.gradient_norm <= gtol)
        result.method, result.seconds = 'product_saddle', perf_counter()-started
        result.evaluations, result.info = self._evaluations, self.info
        self.result = result
        if result.converged:
            self._scales, self._weights = np.exp(result.x)[None, :], np.ones(1)
            self.optQ = self._Q(self._scales[0])
            self.converged = True
            result.scales, result.Q = self._scales[0].copy(), self.optQ.copy()
        return deepcopy(result)

    optimize = fit

    def average(self, draws=4096, *, seed=0, min_ess=100.):
        """Optional independent-prior importance integration of gamma products.

        Retains scalar products only. No importance proposal is fitted to the
        joint saddle. Low ESS disables prediction; high ESS does not exclude
        unobserved importance tails. Compare independent seeds and precision.
        """
        self._check_source()
        draws = positive_int(draws, 'draws')
        if not np.isfinite(min_ess) or not 0 < min_ess <= draws:
            raise ValueError('min_ess must be positive and no greater than draws')
        self._reset()
        if draws > self.max_evaluations:
            raise ValueError('draw count exceeds evidence evaluation budget')
        required = 8*draws*(2*self.coordinates+4)
        if required > self.max_draw_bytes:
            raise MemoryError(f'reduced draw workspace needs about {required/2**20:.1f} MiB')
        started = perf_counter()
        rng = np.random.default_rng(seed)
        scales = np.ones((draws, self.coordinates))
        for layer in range(self.L):
            shape = self.gamma_shapes[:, layer]
            scales *= rng.gamma(shape, 1/shape, size=scales.shape)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise FloatingPointError('gamma product underflow/overflow; predictions disabled')
        ll = np.empty(draws)
        for i, s in enumerate(scales):
            ll[i] = -self._conditional(np.kron(self.K_IW, self._Q(s)))[0]
        if not np.all(np.isfinite(ll)):
            raise FloatingPointError('non-finite product evidence; predictions disabled')
        weights = np.exp(ll-logsumexp(ll))
        ess = float(1/(weights @ weights))
        self.result = OptimizeResult(method='product_average', completed=True,
            accepted=bool(ess >= min_ess), convergence_certified=False, draws=draws, seed=seed,
            ess=ess, maximum_weight=float(weights.max()), min_ess=float(min_ess),
            log_evidence=float(logsumexp(ll)-np.log(draws)),
            log_evidence_mcse=float(np.sqrt(max(1/ess-1/draws, 0.))),
            evaluations=draws, seconds=perf_counter()-started, info=self.info)
        if ess < min_ess:
            raise RuntimeError(f'product importance ESS {ess:.1f} < {min_ess:g}; predictions disabled')
        self._scales, self._weights = scales, weights
        return deepcopy(self.result)

    def _require_fit(self):
        self._check_source()
        if self._scales is None:
            raise RuntimeError('complete a converged reduced saddle or accepted averaging run first')

    def _component(self, cross, diagonal, Q):
        _, _, state = self._conditional(np.kron(self.K_IW, Q))
        Kcross = np.kron(cross, Q)
        n = len(cross)
        mean = (Kcross @ state.alpha).reshape(n, self.c)
        covariance = diagonal[:, None, None]*Q-np.einsum('mai,mbi->mab',
            (Kcross @ state.reduction).reshape(n, self.c, -1),
            Kcross.reshape(n, self.c, -1))
        covariance = (covariance+covariance.swapaxes(-1,-2))/2
        values, vectors = np.linalg.eigh(covariance)
        if values.min() < -1e-9*max(1., float(np.max(abs(diagonal)))*np.linalg.norm(Q, 2)):
            raise FloatingPointError('negative reduced predictive covariance')
        covariance = (vectors*np.maximum(values, 0.)[:, None, :]) @ vectors.swapaxes(-1, -2)
        return mean, covariance

    def predict_statistics(self, Xtest, *, covariance=None, include_probabilities=True,
                           probability_samples=256, seed=0):
        """Reduced means and either source-joint or reduced latent covariance.

        Joint covariance gives an explicit hybrid Gaussian prediction. Reduced
        averaging integrates class probabilities component by component, and
        includes both within-component and between-component covariance.
        """
        self._require_fit()
        covariance = self.covariance if covariance is None else covariance
        self._validate_covariance(covariance)
        if covariance == 'joint':
            self.model._require_solution()
        if self._classification and include_probabilities:
            probability_samples = positive_int(probability_samples, 'probability_samples')
            if probability_samples & (probability_samples-1):
                raise ValueError('probability_samples must be a power of two')
        Xtest = image_batch(Xtest)
        if not len(Xtest):
            raise ValueError('provide at least one test input')
        prepared = self.model.builder.prepare(Xtest)
        f = self.spatial
        identity = np.eye(f.ranks[0])
        cross, diagonal = f.cross(prepared, identity), f.diagonal(prepared, identity)
        n = len(Xtest)
        mean = np.zeros((n, self.c)); within = np.zeros((n, self.c, self.c))
        between = np.zeros_like(within)
        square_mean = np.zeros_like(mean); square_second = np.zeros_like(mean)
        total = 0.
        probabilities = probability_second = winners = None
        mix_probabilities = self._classification and include_probabilities and covariance == 'reduced'
        if mix_probabilities:
            probabilities = np.zeros((n, self.model.D))
            probability_second = np.zeros_like(probabilities); winners = np.zeros_like(probabilities)
        for i, (s, w) in enumerate(zip(self._scales, self._weights)):
            if w == 0:
                continue
            mu, cov = self._component(cross, diagonal, self._Q(s))
            delta = mu-mean; updated = total+w
            mean += w/updated*delta
            between += w*total/updated*np.einsum('ma,mb->mab', delta, delta)
            within += w*cov
            square_mean += w*w*mu; square_second += w*w*mu*mu
            total = updated
            if mix_probabilities:
                stats = gaussian_softmax_statistics(mu, cov, self.model.basis,
                    samples=probability_samples, seed=seed+i)
                probabilities += w*stats['probabilities']
                probability_second += w*(stats['probability_variance']+stats['probabilities']**2)
                winners += w*stats['argmax_probabilities']
        reduced_cov = (within+between)/total
        latent_cov = reduced_cov
        if covariance == 'joint':
            _, latent_cov = self.model._predict_gaussian(Xtest)
        result = dict(latent_mean=mean, latent_covariance=latent_cov,
            covariance_source=covariance, reduced_latent_covariance=reduced_cov,
            reduced_conditional_covariance=within/total, reduced_between_covariance=between/total,
            latent_mean_mcse=None)
        if covariance == 'joint':
            result['joint_solution_kind'] = self.model.solution_kind
        if self.result.method == 'product_average':
            variance = square_second-2*mean*square_mean+mean**2*np.sum(self._weights**2)
            result['latent_mean_mcse'] = np.sqrt(np.maximum(variance, 0.))/total
        if mix_probabilities:
            result.update(probabilities=probabilities/total,
                probability_variance=np.maximum(probability_second/total-(probabilities/total)**2, 0.),
                argmax_probabilities=winners/total,
                probability_model='reduced_mixture' if self.result.method == 'product_average' else 'reduced_gaussian')
        elif self._classification and include_probabilities:
            result.update(gaussian_softmax_statistics(mean, latent_cov, self.model.basis,
                samples=probability_samples, seed=seed))
            result['probability_model'] = 'hybrid_gaussian'
        return result

    def predict_mean(self, Xtest):
        return self.predict_statistics(Xtest, covariance='reduced', include_probabilities=False)['latent_mean']

    def predict_latent(self, Xtest, *, covariance=None):
        stats = self.predict_statistics(Xtest, covariance=covariance, include_probabilities=False)
        return stats['latent_mean'], stats['latent_covariance']

    def predict_proba(self, Xtest, *, covariance=None, samples=256, seed=0):
        if not self._classification:
            raise ValueError('probabilities require a classification source model')
        return self.predict_statistics(Xtest, covariance=covariance,
            probability_samples=samples, seed=seed)['probabilities']

    def predict(self, Xtest):
        return self.predict_proba(Xtest).argmax(axis=1) if self._classification else self.predict_mean(Xtest)
