"""Supported linear CNN joint rates and block-SPD optimization.

The state is normalized at IW, ordered from first hidden layer to last.
Gram and independent-innovation coordinates describe the same rate.
No finite-width density or coordinate Jacobians are included.
"""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from ..conv_geometry import positive_int
from ._matrix_order_parameter import SymmetricCoordinates, exp_divided_differences


def sym(A):
    return (A + A.T) / 2


class SPD:
    def __init__(self, Q):
        self.values, self.vectors = np.linalg.eigh(sym(Q))
        if not np.all(np.isfinite(self.values)) or self.values[0] <= 0:
            raise np.linalg.LinAlgError("joint state lost positive definiteness")
        self.sqrt_values = np.sqrt(self.values)
        self.root = (self.vectors * self.sqrt_values) @ self.vectors.T
        self.inverse = (self.vectors / self.values) @ self.vectors.T
        self.inverse_root = (self.vectors / self.sqrt_values) @ self.vectors.T
        self.logdet = float(np.log(self.values).sum())

    def root_adjoint(self, G):
        V = self.vectors
        return V @ ((V.T @ sym(G) @ V) /
                    (self.sqrt_values[:, None] + self.sqrt_values[None, :])) @ V.T


class GatherMap:
    def __init__(self, maps):
        self.maps = np.asarray(maps)
        self.upper, self.lower = self.maps.shape[1:]

    def apply(self, Q):
        return sym(sum(A.T @ Q @ A for A in self.maps))

    def adjoint(self, G):
        return sym(sum(A @ G @ A.T for A in self.maps))

    def algebra(self, Q):
        return SPD(self.apply(Q))


class OutputGatherMap:
    """Lift a spatial gather by I_c without storing Kronecker gather tensors."""
    def __init__(self, spatial, outputs):
        self.spatial, self.c = spatial, outputs
        self.upper, self.lower = spatial.upper*outputs, spatial.lower*outputs

    def apply(self, Q):
        A = self.spatial.maps
        blocks = Q.reshape(self.spatial.upper, self.c, self.spatial.upper, self.c)
        return sym(np.einsum('tui,uavb,tvj->iajb', A, blocks, A,
                             optimize=True).reshape(self.lower, self.lower))

    def adjoint(self, G):
        A = self.spatial.maps
        blocks = G.reshape(self.spatial.lower, self.c, self.spatial.lower, self.c)
        return sym(np.einsum('tui,iajb,tvj->uavb', A, blocks, A,
                             optimize=True).reshape(self.upper, self.upper))

    def algebra(self, Q):
        return SPD(self.apply(Q))


class RepeatedMap:
    """I_m tensor Q, without a dense rank-three gather tensor."""
    def __init__(self, m, upper):
        self.m, self.upper, self.lower = m, upper, m * upper

    def apply(self, Q):
        return np.kron(np.eye(self.m), Q)

    def adjoint(self, G):
        r = self.upper
        return sym(sum(G[t*r:(t+1)*r, t*r:(t+1)*r] for t in range(self.m)))

    def algebra(self, Q):
        return RepeatedSPD(Q, self.m)


class RepeatedSPD:
    def __init__(self, Q, m):
        self.base, self.m = SPD(Q), m
        self.root = np.kron(np.eye(m), self.base.root)
        self.inverse = np.kron(np.eye(m), self.base.inverse)
        self.inverse_root = np.kron(np.eye(m), self.base.inverse_root)
        self.logdet = m * self.base.logdet

    def root_adjoint(self, G):
        r = len(self.base.values)
        V, s = self.base.vectors, self.base.sqrt_values
        result = np.empty_like(G)
        denominator = s[:, None] + s[None, :]
        # Off-diagonal blocks need not be symmetric individually.
        for i in range(self.m):
            for j in range(self.m):
                a, b = slice(i*r, (i+1)*r), slice(j*r, (j+1)*r)
                result[a, b] = V @ ((V.T @ G[a, b] @ V) / denominator) @ V.T
        return sym(result)


class BlockCoordinates:
    def __init__(self, ranks):
        self.blocks = tuple(SymmetricCoordinates(r) for r in ranks)
        self.sizes = tuple(len(c.scale) for c in self.blocks)
        self.size = sum(self.sizes)
        self.origin = np.zeros(self.size)

    def validate(self, matrices):
        if np.isscalar(matrices) or (isinstance(matrices, np.ndarray) and matrices.ndim == 0):
            matrices = [matrices] * len(self.blocks)
        if len(matrices) != len(self.blocks):
            raise ValueError("provide one SPD matrix per hidden layer, first to last")
        return tuple(c.validate_q(Q)[0] for c, Q in zip(self.blocks, matrices))

    def encode(self, matrices):
        result = []
        for c, Q in zip(self.blocks, self.validate(matrices)):
            _, e, V = c.validate_q(Q)
            result.append(c.pack((V * np.log(e)) @ V.T))
        return np.concatenate(result)

    def decode(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.size,) or not np.all(np.isfinite(x)):
            raise ValueError("invalid block log-matrix coordinates")
        matrices, spectra = [], []
        for c, z in zip(self.blocks, np.split(x, np.cumsum(self.sizes)[:-1])):
            s, V = np.linalg.eigh(c.unpack(z))
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                Q = (V * np.exp(s)) @ V.T
            if np.linalg.eigvalsh(Q)[0] <= 0:
                raise FloatingPointError("joint matrix cannot be represented as SPD")
            matrices.append(sym(Q))
            spectra.append((s, V))
        return tuple(matrices), spectra

    def pullback(self, gradients, spectra):
        return np.concatenate([
            c.pack(sym(V @ (exp_divided_differences(s) * (V.T @ G @ V)) @ V.T))
            for c, G, (s, V) in zip(self.blocks, gradients, spectra)])


@dataclass
class JointState:
    grams: tuple
    matrices: tuple
    scales: tuple
    prior_gradient: tuple
    rate: float
    parameterization: str


class JointRate:
    def __init__(self, ranks, maps, widths, *, rank_policy='full_rank'):
        self.ranks = tuple(positive_int(r, 'rank') for r in ranks)
        self.maps = tuple(maps)
        self.widths = tuple(positive_int(n, 'width') for n in widths)
        if not self.ranks or len(self.widths) != len(self.ranks) or len(self.maps) != len(self.ranks)-1:
            raise ValueError('provide one positive width per rank and one map between adjacent layers')
        for l, mapping in enumerate(self.maps):
            if (mapping.lower, mapping.upper) != self.ranks[l:l+2]:
                raise ValueError('gather dimensions must match adjacent joint ranks')
        self.rank_policy = self.validate_rank_policy(rank_policy)
        self.nu = np.asarray(self.widths, dtype=float) / self.widths[0]
        for l, (width, rank) in enumerate(zip(self.widths, self.ranks)):
            if width < rank and self.rank_policy == 'full_rank':
                raise ValueError(f"layer {l+1}: width {width} < joint support/path rank {rank}; "
                                 "use rank_policy='leading_rate' to explicitly continue "
                                 "the SPD covariance-tilt saddle below the empirical rank threshold")
        self.coordinates = BlockCoordinates(self.ranks)

    @staticmethod
    def validate_rank_policy(value):
        if value not in ('full_rank', 'leading_rate'):
            raise ValueError("rank_policy must be 'full_rank' or 'leading_rate'")
        return value

    @staticmethod
    def validate_parameterization(value):
        if value not in ('gram', 'innovation'):
            raise ValueError("parameterization must be 'gram' or 'innovation'")
        return value

    def evaluate(self, matrices, parameterization):
        self.validate_parameterization(parameterization)
        grams, scales = [None]*len(self.ranks), [None]*len(self.ranks)
        grads = [np.zeros((r, r)) for r in self.ranks]
        value = 0.
        for l in range(len(self.ranks)-1, -1, -1):
            r, M = self.ranks[l], matrices[l]
            S = SPD(np.eye(r)) if l == len(self.ranks)-1 else self.maps[l].algebra(grams[l+1])
            scales[l] = S
            E = SPD(M)
            if parameterization == 'innovation':
                grams[l] = sym(S.root @ M @ S.root)
                z = np.log(E.values)
                value += self.nu[l] * np.sum(np.expm1(z)-z)
                grads[l] = self.nu[l] * (np.eye(r)-E.inverse)
            else:
                grams[l] = M
                value += self.nu[l] * (np.trace(S.inverse @ M)-E.logdet+S.logdet-r)
                grads[l] += self.nu[l] * (S.inverse-E.inverse)
                if l < len(self.ranks)-1:
                    grads[l+1] += self.nu[l] * self.maps[l].adjoint(
                        S.inverse-S.inverse @ M @ S.inverse)
        return JointState(tuple(grams), tuple(matrices), tuple(scales),
                          tuple(grads), float(value), parameterization)

    def gradient(self, state, first_gradient):
        gradients = [G.copy() for G in state.prior_gradient]
        if state.parameterization == 'gram':
            gradients[0] += first_gradient
        else:
            G = first_gradient
            for l, (M, S) in enumerate(zip(state.matrices, state.scales)):
                gradients[l] += S.root @ G @ S.root
                if l < len(self.ranks)-1:
                    root_gradient = G @ S.root @ M + M @ S.root @ G
                    G = self.maps[l].adjoint(S.root_adjoint(root_gradient))
        return tuple(sym(G) for G in gradients)

    def innovations(self, grams):
        return tuple(sym(S.inverse_root @ Q @ S.inverse_root) for S, Q in zip(
            self.evaluate(grams, 'gram').scales, grams))

    def profile_last_gram(self, first):
        """Exact terminal minimizer of a two-layer repeated-path Gram action.

        This profiles the leading saddle, not finite-width fluctuations.
        """
        if len(self.ranks) != 2 or not isinstance(self.maps[0], RepeatedMap):
            raise ValueError('profile_last requires two layers with a repeated path map')
        T = self.maps[0].adjoint(first)
        values, vectors = np.linalg.eigh(T)
        if values[0] <= 0:
            raise np.linalg.LinAlgError('profiled Gram scale must be positive definite')
        a = self.nu[1]
        b = self.maps[0].m*self.nu[0]-a
        rad = np.sqrt(b*b+4*a*self.nu[0]*values)
        roots = (2*self.nu[0]*values/(rad+b) if b >= 0 else (rad-b)/(2*a))
        return sym((vectors*roots) @ vectors.T)


def minimize_joint(objective, coordinates, start, *, maxiter=500, gtol=1e-6,
                   n_restarts=0, random_state=0, verbose=False):
    """L-BFGS with checked gradients; small BFGS refinement on failure."""
    maxiter = positive_int(maxiter, 'maxiter')
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    if (isinstance(n_restarts, (bool, np.bool_)) or
            not isinstance(n_restarts, (int, np.integer)) or n_restarts < 0):
        raise ValueError("n_restarts must be a nonnegative integer")
    started = perf_counter()
    evaluations, cache_x, cache = 0, None, None

    def safe(x, initial=False):
        nonlocal evaluations, cache_x, cache
        if cache_x is not None and np.array_equal(x, cache_x):
            return cache[0], cache[1].copy()
        evaluations += 1
        try:
            value, grad = objective(x)
            if not np.isfinite(value) or not np.all(np.isfinite(grad)):
                raise FloatingPointError("non-finite joint action or gradient")
        except (np.linalg.LinAlgError, FloatingPointError):
            if initial:
                raise
            value, grad = np.inf, np.zeros_like(x)
        cache_x, cache = x.copy(), (float(value), np.array(grad, copy=True))
        return value, grad

    rng = np.random.default_rng(random_state)
    starts = [coordinates.encode(start)]
    for k in range(n_restarts):
        # Perturbations include a shared expansion/contraction of all layers.
        scalar = (-1 if k % 2 == 0 else 1) * .35 * (1+k//2)
        diagonal = np.concatenate([c.pack(np.eye(c.d)) for c in coordinates.blocks])
        starts.append(starts[0] + scalar*diagonal + rng.normal(0, .035, coordinates.size))
    attempts = []
    for i, x in enumerate(starts):
        safe(x, initial=True)
        for method in ('L-BFGS-B', 'BFGS'):
            options = dict(gtol=gtol, maxiter=maxiter)
            if method == 'L-BFGS-B':
                options.update(ftol=1e-14, maxls=40, maxcor=20)
            def progress(z):
                if verbose:
                    val, grad = safe(z)
                    print(f"Joint {method}, start {i+1}: action={val:.9g}, "
                          f"gradient={np.max(abs(grad)):.3g}, "
                          f"elapsed={perf_counter()-started:.1f}s", flush=True)
            f = minimize(safe, x, jac=True, method=method, options=options, callback=progress)
            f.fun, f.jac = safe(f.x)
            f.gradient_norm = float(np.max(abs(f.jac))) if np.isfinite(f.fun) else np.inf
            f.converged = bool(np.isfinite(f.fun) and f.gradient_norm <= gtol)
            f.start_index, f.method = i, method
            attempts.append(f)
            if f.converged or coordinates.size > 512 or not np.isfinite(f.fun):
                break
            x = f.x
    finite = [a for a in attempts if np.isfinite(a.fun)]
    if not finite:
        raise RuntimeError("all joint saddle attempts failed")
    eligible = [a for a in finite if a.converged] or finite
    minimum = min(a.fun for a in eligible)
    best = min([a for a in eligible if a.fun <= minimum+1e-12*(1+abs(minimum))],
               key=lambda a: a.gradient_norm)
    best.optimization_seconds = perf_counter()-started
    best.total_objective_evaluations = evaluations
    best.lower_unconverged_action = any(
        not a.converged and a.fun < best.fun-1e-12*(1+abs(best.fun)) for a in finite)
    best.converged_starts = len({a.start_index for a in attempts if a.converged})
    return best, attempts
