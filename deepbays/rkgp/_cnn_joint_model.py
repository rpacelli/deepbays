"""Shared scalar and multi-output inference for the opt-in joint CNN theories."""

from copy import deepcopy
import warnings
import numpy as np
from ..conv_geometry import positive_int
from ..kernels.conv_kernels import as_numpy, image_batch
from ._cnn_joint_kernel import JointCNNFeatures
from ._cnn_joint_multioutput import MultioutputJointFeatures
from ._cnn_joint_rate import JointRate, BlockCoordinates, minimize_joint


class JointCNNModel:
    def _initialize(self, L, Nc, *, priors, act, mask, stride, padding, gamma,
                    pooling, closure, parameterization, batch_size,
                    max_kernel_bytes, kernel_backend, max_dense_size,
                    max_joint_coordinates, rank_policy='full_rank',
                    allow_experimental_relu=False):
        self.L = positive_int(L, 'L')
        if np.ndim(Nc) == 0:
            widths = [positive_int(Nc, 'Nc')]*self.L
        else:
            if len(Nc) != self.L:
                raise ValueError("Nc must be an integer or one width per hidden layer")
            widths = [positive_int(n, 'Nc') for n in Nc]
        self.widths, self.N1 = tuple(widths), widths[0]
        self.Nc = widths[0] if len(set(widths)) == 1 else self.widths
        self.parameterization = JointRate.validate_parameterization(parameterization)
        self.rank_policy = JointRate.validate_rank_policy(rank_policy)
        self.batch_size = positive_int(batch_size, 'batch_size')
        self.max_dense_size = positive_int(max_dense_size, 'max_dense_size')
        self._settings = deepcopy(dict(L=self.L, widths=self.widths, priors=priors,
            act=act, mask=mask, stride=stride, padding=padding, gamma=gamma,
            pooling=pooling, closure=closure, batch_size=batch_size,
            max_kernel_bytes=max_kernel_bytes, kernel_backend=kernel_backend,
            max_joint_coordinates=max_joint_coordinates, outputs=self.c,
            rank_policy=self.rank_policy, allow_experimental_relu=allow_experimental_relu))
        self._ready = False
        self._reset()

    def _reset(self):
        self._grams = self._innovations = None
        self.result, self.solution_kind = None, None
        self.converged = False
        self.optimization_results = []
        self._posterior = self._posterior_signature = None
        self._prediction = self._warm_alpha = None

    def preprocess(self, X, Y):
        """Freeze training data and IW kernels; no test data enters fitting."""
        self._ready = False
        self._reset()
        X = image_batch(X).copy()
        if len(X)*self.c > self.max_dense_size:
            raise ValueError("dense likelihood size limit exceeded; increase max_dense_size explicitly")
        y = self._targets(Y, len(X))
        features = JointCNNFeatures(X, **self._settings)
        if self.c > 1:
            features = MultioutputJointFeatures(features, self.c)
        rate = JointRate(features.ranks, features.maps, self.widths, rank_policy=self.rank_policy)
        X.setflags(write=False)
        y.setflags(write=False)
        self.X, self.Y, self.P = X, y, len(X)
        self.features, self.rate = features, rate
        self.builder, self._coordinates = features.builder, rate.coordinates
        self.closure = features.closure
        self._ready = True
        return self

    def _require_ready(self):
        if not self._ready:
            raise RuntimeError("call preprocess(X, Y) first")

    @property
    def joint_info(self):
        self._require_ready()
        return dict(self.features.info, widths=self.widths, parameterization=self.parameterization,
                    spatial_ranks=tuple(r//self.c for r in self.rate.ranks), outputs=self.c,
                    rank_policy=self.rank_policy, inference='leading_rate_saddle',
                    empirical_singular=tuple(n < r for n, r in zip(self.widths, self.rate.ranks)),
                    dimension_width_ratios=tuple(r/n for n, r in zip(self.widths, self.rate.ranks)),
                    likelihood_dimension=self.P*self.c,
                    nonlinear_closure=(self.closure == 'iw_path_gain' and self.builder.act != 'id'))

    def fluctuation_diagnostics(self, weighting='label_free'):
        """Training-only, linearized IW fluctuations; not posterior error bounds.

        label_free projects onto the average kernel diagonal. iw_dual uses
        the outer product of the normalized IW Gaussian/Laplace alpha vector.
        Neither option changes the action or fits a correction to predictions.
        """
        self._require_ready()
        if weighting not in ('label_free', 'iw_dual'):
            raise ValueError("weighting must be 'label_free' or 'iw_dual'")
        identity = tuple(np.eye(r) for r in self.rate.ranks)
        state = self.rate.evaluate(identity, 'innovation')
        K = self.features.kernel(state.grams[0])
        if weighting == 'label_free':
            score = np.eye(len(K))/len(K)
        else:
            warm = self._warm_alpha
            try:
                _, _, posterior = self._evidence(K)
            finally:
                self._warm_alpha = warm
            v = posterior.alpha.copy()
            norm = np.linalg.norm(v)
            if norm:
                v /= norm
            score = np.outer(v, v)
        projections = self.rate.gradient(state, self.features.adjoint(score))
        squares = [float(np.sum(A*A)) for A in projections]
        ranks = tuple(float(np.trace(A)**2/s) if s > 0 else None
                      for A, s in zip(projections, squares))
        components = tuple(2*s/n for s, n in zip(squares, self.widths))
        value = float(np.sum(score*K))
        variance = sum(components)
        return dict(weighting=weighting, labels_used=(weighting == 'iw_dual'),
                    iw_value=value, effective_ranks=ranks,
                    layer_variances=components, linearized_variance=variance,
                    linearized_relative_variance=(variance/value**2 if value > 0 else None))

    @property
    def optGrams(self):
        """Normalized supported/path Grams, first layer to last; copies."""
        return None if self._grams is None else tuple(Q.copy() for Q in self._grams)

    @property
    def optInnovations(self):
        """Independent innovations in symmetric-root coordinates; copies."""
        return None if self._innovations is None else tuple(U.copy() for U in self._innovations)

    def _state(self, matrices, parameterization):
        self._require_ready()
        matrices = self._coordinates.validate(matrices)
        return self.rate.evaluate(matrices, parameterization)

    def _objective(self, state):
        K = self.features.kernel(state.grams[0])
        nll, kernel_gradient, _ = self._evidence(K)
        first_gradient = (2/self.N1)*self.features.adjoint(kernel_gradient)
        gradients = self.rate.gradient(state, first_gradient)
        return state.rate+2*nll/self.N1, gradients

    def _log_action_gradient(self, x, parameterization=None):
        self._require_ready()
        mode = self.parameterization if parameterization is None else parameterization
        matrices, spectra = self._coordinates.decode(x)
        state = self.rate.evaluate(matrices, mode)
        value, gradients = self._objective(state)
        return value, self._coordinates.pullback(gradients, spectra)

    def priorRate(self, matrices=None, *, parameterization=None):
        mode = self.parameterization if parameterization is None else parameterization
        if matrices is None:
            self._require_solution()
            matrices = self._grams if mode == 'gram' else self._innovations
        return self._state(matrices, mode).rate

    def effectiveAction(self, matrices=None, *, parameterization=None):
        mode = self.parameterization if parameterization is None else parameterization
        if matrices is None:
            self._require_solution()
            matrices = self._grams if mode == 'gram' else self._innovations
        return self._objective(self._state(matrices, mode))[0]

    def computeActionGrad(self, matrices, *, parameterization=None):
        """Full Frobenius derivatives, one matrix per layer."""
        mode = self.parameterization if parameterization is None else parameterization
        return self._objective(self._state(matrices, mode))[1]

    def _select(self, state, kind, converged):
        self._grams = tuple(Q.copy() for Q in state.grams)
        self._innovations = (tuple(U.copy() for U in state.matrices)
                             if state.parameterization == 'innovation'
                             else self.rate.innovations(self._grams))
        for Q in (*self._grams, *self._innovations):
            Q.setflags(write=False)
        self.solution_kind, self.converged = kind, bool(converged)
        self._posterior = self._posterior_signature = self._prediction = None

    def setInnovations(self, matrices=1.):
        state = self._state(matrices, 'innovation')
        self._reset()
        self._select(state, 'fixed_innovations', True)
        return self

    def setGrams(self, matrices=1.):
        state = self._state(matrices, 'gram')
        self._reset()
        self._select(state, 'fixed_grams', True)
        return self

    def setIW(self):
        self.setInnovations(1.)
        self.solution_kind = 'infinite_width'
        return self

    def optimize(self, state0=1., *, parameterization=None, maxiter=500, gtol=1e-6,
                 n_restarts=0, random_state=0, verbose=False, profile_last=False):
        """Minimize the same joint action in Gram or innovation coordinates.

        state0 is a scalar times every identity, or a sequence of SPD matrices
        in the selected parameterization. Saved optGrams/optInnovations permit
        explicit continuation between widths or parameterizations.
        With two layers and repeated path maps, profile_last=True analytically
        eliminates the terminal Gram; requires parameterization='gram'.
        The supplied terminal starting matrix is then replaced by its optimum.
        """
        self._require_ready()
        mode = self.parameterization if parameterization is None else parameterization
        JointRate.validate_parameterization(mode)
        state0 = self._coordinates.validate(state0)
        if not isinstance(profile_last, (bool, np.bool_)):
            raise ValueError('profile_last must be Boolean')
        coordinates = self._coordinates
        objective = lambda x: self._log_action_gradient(x, mode)
        if profile_last:
            if mode != 'gram':
                raise ValueError("profile_last requires parameterization='gram'")
            # Validate geometry before resetting an existing solution.
            self.rate.profile_last_gram(state0[0])
            coordinates = BlockCoordinates(self.rate.ranks[:1])
            state0 = state0[:1]

            def objective(x):
                (first,), spectra = coordinates.decode(x)
                grams = (first, self.rate.profile_last_gram(first))
                value, gradients = self._objective(self.rate.evaluate(grams, 'gram'))
                # Envelope theorem: the eliminated matrix has zero gradient.
                return value, coordinates.pullback(gradients[:1], spectra)
        self._reset()
        self.parameterization = mode
        result, attempts = minimize_joint(
            objective, coordinates, state0,
            maxiter=maxiter, gtol=gtol, n_restarts=n_restarts,
            random_state=random_state, verbose=verbose)
        matrices, _ = coordinates.decode(result.x)
        if profile_last:
            matrices = (matrices[0], self.rate.profile_last_gram(matrices[0]))
            _, full_gradient = self._log_action_gradient(self._coordinates.encode(matrices), 'gram')
            result.full_gradient_norm = float(np.max(abs(full_gradient)))
            result.converged = result.converged and result.full_gradient_norm <= gtol
        state = self.rate.evaluate(matrices, mode)
        self._select(state, 'saddle', result.converged)
        self.result, self.optimization_results = result, attempts
        result.parameterization, result.closure = mode, self.closure
        result.ranks, result.coordinates = self.rate.ranks, coordinates.size
        result.joint_coordinates = self._coordinates.size
        result.profile_last = bool(profile_last)
        result.rank_policy = self.rank_policy
        result.joint_info = self.joint_info
        result.grams, result.innovations = self.optGrams, self.optInnovations
        if not self.converged:
            warnings.warn(f"joint saddle did not converge: gradient {result.gradient_norm:.3g}; "
                          "predictions disabled", RuntimeWarning)
        return result

    def _require_solution(self):
        self._require_ready()
        if not self.converged or self._grams is None:
            raise RuntimeError("optimize successfully or select setIW/setGrams/setInnovations first")

    def effectiveKernel(self):
        self._require_solution()
        return self.features.kernel(self._grams[0])

    def _solution_posterior(self):
        self._require_solution()
        signature = self._evidence_signature()
        if self._posterior is None or signature != self._posterior_signature:
            _, _, self._posterior = self._evidence(self.effectiveKernel())
            self._posterior_signature = signature
        return self._posterior

    def _predict_gaussian(self, Xtest, batch_size=None):
        posterior = self._solution_posterior()
        Xtest = image_batch(Xtest)
        bs = self.batch_size if batch_size is None else positive_int(batch_size, 'batch_size')
        means, variances = [], []
        for start in range(0, len(Xtest), bs):
            prepared = self.builder.prepare(Xtest[start:start+bs])
            cross = self.features.cross(prepared, self._grams[0])
            prior = self.features.diagonal(prepared, self._grams[0])
            mean = cross @ posterior.alpha
            if self.c == 1:
                variance = prior-np.sum((cross @ posterior.reduction)*cross, axis=1)
                if variance.min() < -1e-9*max(1., float(np.max(np.abs(prior)))):
                    raise FloatingPointError("negative joint predictive variance beyond roundoff")
                covariance = np.maximum(variance, 0.)[:, None, None]
            else:
                left = (cross @ posterior.reduction).reshape(-1, self.c, self.P*self.c)
                right = cross.reshape(left.shape)
                covariance = prior-np.einsum('mai,mbi->mab', left, right)
                covariance = (covariance+covariance.transpose(0, 2, 1))/2
                values, vectors = np.linalg.eigh(covariance)
                if values.min() < -1e-9*max(1., float(np.max(np.abs(prior)))):
                    raise FloatingPointError("negative joint predictive covariance beyond roundoff")
                covariance = (vectors*np.maximum(values, 0.)[:, None, :]) @ vectors.transpose(0, 2, 1)
            means.append(mean.reshape(-1, self.c))
            variances.append(covariance)
        return np.concatenate(means), np.concatenate(variances)
