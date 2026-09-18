"""Deterministic posterior potentials in whitened Gaussian coordinates.

The prior is N(0, I) in theta, so U(theta) = ||theta||²/2 + summed_loss/T.
These objects know nothing about HMC, NUTS, chain storage, or diagnostics.
"""
from collections.abc import Mapping
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _positive(value, name):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _device_dtype(device, dtype):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('supported sampling devices are cpu and cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch installation')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('sampling requires torch.float32 or torch.float64')
    return device, dtype


def negative_log_likelihood(outputs, targets, task, temperature):
    """Summed loss / T. Classification expects raw C logits and integer labels.

    This low-level differentiable function assumes shapes were validated when
    constructing the posterior. The Gaussian loss includes the factor 1/2.
    Normalizing constants independent of the sampled coordinates are omitted.
    """
    temperature = _positive(temperature, 'temperature')
    if task == 'regression':
        return .5 * (outputs-targets).square().sum() / temperature
    if task == 'classification':
        return F.cross_entropy(outputs, targets, reduction='sum') / temperature
    raise ValueError("task must be 'regression' or 'classification'")


def potential(theta, outputs, targets, task, temperature):
    """Standard-normal coordinate prior plus tempered likelihood."""
    return .5 * theta.square().sum() + negative_log_likelihood(outputs, targets, task, temperature)


class WhitenedNetwork:
    """Flatten every named parameter of a deterministic torch module.

    ``priors`` is a scalar precision, an exact parameter-name mapping, or one
    precision per named parameter tensor (in named_parameters() order). Biases
    are sampled too; for biased networks a mapping is usually clearest.
    The original module is untouched: a private copy is moved to device/dtype
    and put in eval mode. Buffers remain fixed. All named parameters are sampled,
    regardless of their original requires_grad flag; no trainable layer is skipped.
    """

    def __init__(self, network, priors=1., device='cpu', dtype=torch.float64):
        self.device, self.dtype = _device_dtype(device, dtype)
        self.network = copy.deepcopy(network).to(device=self.device, dtype=dtype).eval()
        named = list(self.network.named_parameters())
        if not named:
            raise ValueError('network has no parameters to sample')
        names = [name for name, _ in named]
        if isinstance(priors, Mapping):
            if set(priors) != set(names):
                raise ValueError('prior mapping must contain exactly all named parameters')
            precisions = [priors[name] for name in names]
        elif np.isscalar(priors) or (torch.is_tensor(priors) and priors.ndim == 0):
            precisions = [priors] * len(named)
        else:
            precisions = list(priors)
            if len(precisions) != len(named):
                raise ValueError('provide one prior precision per named parameter tensor')
        self.specification = [(name, p.shape, p.numel(), _positive(lam, 'prior precision'))
                              for (name, p), lam in zip(named, precisions)]
        self._sizes = tuple(size for _, _, size, _ in self.specification)
        self._parameter_specs = tuple((name, shape, 1. / math.sqrt(precision))
                                      for name, shape, _, precision in self.specification)
        self.dimension = sum(self._sizes)
        self._buffers = dict(self.network.named_buffers())
        # Inference uses only theta's gradient, not gradients of the template.
        self.network.requires_grad_(False)

    def parameters(self, theta):
        """Differentiable unwhitening; returns physical named parameter tensors."""
        if theta.ndim != 1 or theta.numel() != self.dimension:
            raise ValueError(f'theta must have shape ({self.dimension},)')
        # One split joins parameter gradients once; independent slices each
        # scatter their gradient into a full-sized theta in eager autograd.
        parameters = {}
        for (name, shape, scale), flat in zip(self._parameter_specs, theta.split(self._sizes)):
            value = flat.reshape(shape)
            parameters[name] = value if scale == 1. else value * scale
        return parameters

    def _forward(self, parameters, X):
        return torch.func.functional_call(self.network, (parameters, self._buffers),
                                          (X,), strict=True)

    def __call__(self, theta, X):
        return self._forward(self.parameters(theta), X)

    def predict(self, theta, X, batch_size=None):
        """One draw's outputs, optionally chunked over inputs; returns a tensor.

        Input arrays may remain on CPU even for a CUDA network. Only the current
        input chunk is transferred. The output stays on the network's device.
        Use no_grad for predictive evaluation; gradients are supported too.
        """
        X = torch.as_tensor(X, dtype=self.dtype)
        if X.ndim < 2 or len(X) == 0:
            raise ValueError('X must be a nonempty batch')
        size = _batch_size(batch_size, len(X))
        theta = theta.to(device=self.device, dtype=self.dtype)
        parameters = self.parameters(theta)
        if size == len(X):
            return self._forward(parameters, X.to(self.device))
        return torch.cat([self._forward(parameters, x.to(self.device)) for x in X.split(size)], dim=0)

    def metadata(self):
        return dict(kind='network', dimension=self.dimension,
                    parameters=[dict(name=n, shape=list(shape), precision=lam)
                                for n, shape, _, lam in self.specification])


def _batch_size(size, count):
    if size is None:
        return count
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError('batch_size must be a positive integer or None')
    return min(size, count)


def _targets(y, task, count, outputs, device, dtype):
    y = torch.as_tensor(y, device=device).detach().clone()
    if task == 'classification':
        if outputs < 2 or y.shape != (count,) or y.is_floating_point() or y.is_complex() or y.dtype == torch.bool:
            raise ValueError('classification needs >=2 logits and integer labels of shape (P,)')
        if torch.any((y < 0) | (y >= outputs)):
            raise ValueError('class labels must be in [0, number of logits)')
        return y.to(dtype=torch.long)
    if task != 'regression':
        raise ValueError("task must be 'regression' or 'classification'")
    if y.shape == (count,) and outputs == 1:
        y = y[:, None]
    if y.shape != (count, outputs) or not torch.isfinite(y).all():
        raise ValueError('regression targets must match outputs (P,D); (P,) is accepted for D=1')
    return y.to(dtype=dtype)


class NetworkPosterior:
    """Square-loss or softmax-CE weight posterior, independent of the sampler.

    ``batch_size`` chunks the *entire* summed likelihood; it never subsamples
    data. With ``checkpoint_batches=True`` (default), activation checkpointing
    bounds backward-pass activation memory by a chunk, at extra compute cost.
    ``data_device='cpu'`` also keeps the training data off the GPU.
    The network must return (batch, outputs) in deterministic eval mode.
    """

    def __init__(self, network, X, y, *, priors=1., task='regression', temperature=1.,
                 device='cpu', dtype=torch.float64, batch_size=None,
                 data_device=None, checkpoint_batches=True):
        self.weights = WhitenedNetwork(network, priors, device, dtype)
        self.dimension, self.device, self.dtype = self.weights.dimension, self.weights.device, dtype
        self.task, self.temperature = task, _positive(temperature, 'temperature')
        location = self.device if data_device is None else torch.device(data_device)
        self.X = torch.as_tensor(X, dtype=dtype, device=location).detach().clone()
        if self.X.ndim < 2 or len(self.X) == 0 or not torch.isfinite(self.X).all():
            raise ValueError('X must be a finite nonempty input batch')
        self.batch_size = _batch_size(batch_size, len(self.X))
        self.checkpoint_batches = bool(checkpoint_batches)
        with torch.no_grad():
            probe = self.weights(torch.zeros(self.dimension, device=self.device, dtype=dtype),
                                 self.X[:1].to(self.device))
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise ValueError('network outputs must have shape (batch, outputs)')
        self.y = _targets(y, task, len(self.X), probe.shape[1], location, dtype)
        self._compiled = None

    def _chunk_loss(self, parameters, x, y):
        outputs = self.weights._forward(parameters, x.to(self.device))
        return negative_log_likelihood(outputs, y.to(self.device), self.task, self.temperature)

    def _potential(self, theta):
        total = .5 * theta.square().sum()
        parameters = self.weights.parameters(theta)
        if self.batch_size == len(self.X):
            return total + self._chunk_loss(parameters, self.X, self.y)
        chunks = zip(self.X.split(self.batch_size), self.y.split(self.batch_size))
        for x, y in chunks:
            if self.checkpoint_batches and torch.is_grad_enabled():
                loss = checkpoint(self._chunk_loss, parameters, x, y, use_reentrant=False)
            else:
                loss = self._chunk_loss(parameters, x, y)
            total = total + loss
        return total

    def potential(self, theta):
        return self._potential(theta) if self._compiled is None else self._compiled(theta)

    def compile(self, *, backend='inductor', **kwargs):
        """Opt in to torch.compile of the potential, then return self.

        Compilation has startup cost and backend/architecture constraints. It
        does not compile Pyro's trajectory-control logic or vectorize chains.
        No silent fallback is added: compilation errors propagate to the caller.
        """
        self._compiled = torch.compile(self._potential, backend=backend, **kwargs)
        return self

    def predict(self, theta, X, batch_size=None):
        return self.weights.predict(theta, X, batch_size)

    def metadata(self):
        return dict(self.weights.metadata(), task=self.task, temperature=self.temperature,
                    observations=len(self.X), input_shape=list(self.X.shape[1:]),
                    batch_size=self.batch_size, checkpoint_batches=self.checkpoint_batches,
                    data_device=str(self.X.device), compiled=self._compiled is not None)


def covariance_root(K):
    """NumPy PSD support factor; truncate spectral roundoff, never add jitter."""
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or len(K) == 0 or not np.isfinite(K).all():
        raise ValueError('K must be a finite nonempty square matrix')
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
        raise ValueError('K must be symmetric')
    values, vectors = np.linalg.eigh((K+K.T)/2)
    tolerance = 64 * np.finfo(float).eps * len(K) * max(1., np.max(np.abs(values)))
    if values.min() < -tolerance:
        raise ValueError('prior covariance has negative eigenvalues beyond roundoff')
    keep = values > tolerance
    return vectors[:, keep] * np.sqrt(values[keep]), tolerance


class GaussianFunctionPosterior:
    """Fixed-kernel control on a joint train/test batch in whitened coordinates.

    root has shape (P_total * latent_outputs, rank), example-major. The first
    len(y) examples are observed. Optional basis has shape (outputs, latent_outputs)
    and lifts orthonormal contrasts to logits. It is identity for regression.
    root is fixed: this samples the conditional GP, not a distribution over Q.
    """

    def __init__(self, root, y, *, latent_outputs, basis=None, task='classification',
                 temperature=1., device='cpu', dtype=torch.float64):
        self.device, self.dtype = _device_dtype(device, dtype)
        self.root = torch.as_tensor(root, dtype=dtype, device=self.device).detach().clone()
        if (not isinstance(latent_outputs, int) or latent_outputs < 1 or self.root.ndim != 2
                or min(self.root.shape) < 1 or self.root.shape[0] % latent_outputs
                or not torch.isfinite(self.root).all()):
            raise ValueError('invalid nonzero-rank root or latent output dimension')
        self.dimension = self.root.shape[1]
        self.latent_outputs = latent_outputs
        self.count = self.root.shape[0] // latent_outputs
        self.basis = (torch.eye(latent_outputs, device=self.device, dtype=dtype) if basis is None
                      else torch.as_tensor(basis, device=self.device, dtype=dtype).detach().clone())
        if self.basis.ndim != 2 or self.basis.shape[1] != latent_outputs or not torch.isfinite(self.basis).all():
            raise ValueError('basis must have shape (outputs, latent_outputs)')
        if not 0 < len(y) <= self.count:
            raise ValueError('observed targets must be a nonempty prefix of the joint batch')
        self.task, self.temperature = task, _positive(temperature, 'temperature')
        self.y = _targets(y, task, len(y), self.basis.shape[0], self.device, dtype)

    def predict(self, theta):
        return (self.root @ theta).reshape(self.count, self.latent_outputs) @ self.basis.T

    def potential(self, theta):
        # The test rows need not be multiplied in every leapfrog evaluation.
        train_root = self.root[:len(self.y)*self.latent_outputs]
        outputs = (train_root @ theta).reshape(len(self.y), self.latent_outputs) @ self.basis.T
        return potential(theta, outputs, self.y, self.task, self.temperature)

    def metadata(self):
        return dict(kind='fixed_gaussian_function', dimension=self.dimension, task=self.task,
                    temperature=self.temperature, observations=len(self.y), joint_count=self.count,
                    latent_outputs=self.latent_outputs)


class PotentialPosterior:
    """Adapter for a custom scalar potential(theta), with unconstrained theta.

    Unlike NetworkPosterior, your callable must include *all* prior/likelihood
    terms itself. Initialization defaults to independent standard normals.
    """

    def __init__(self, potential_fn, dimension, *, device='cpu', dtype=torch.float64):
        self.device, self.dtype = _device_dtype(device, dtype)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError('dimension must be a positive integer')
        self.potential = potential_fn
        self.dimension = dimension

    def metadata(self):
        return dict(kind='custom_potential', dimension=self.dimension)
