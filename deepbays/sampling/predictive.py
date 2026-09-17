"""Post-processing posterior function draws, separate from posterior sampling.

Softmax moments, argmax frequencies, and categorical-label draws are distinct.
None of these functions alters the HMC target or applies its temperature to logits.
"""
import numpy as np
from scipy.linalg import helmert
from scipy.special import softmax
import torch


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def logit_contrasts(logits):
    """Project C logits to C-1 orthonormal contrasts, dropping the common mode."""
    if logits.shape[-1] < 2:
        raise ValueError('at least two logits are required')
    basis = helmert(logits.shape[-1], full=False).T.copy()
    if torch.is_tensor(logits):
        return logits @ torch.as_tensor(basis, device=logits.device, dtype=logits.dtype)
    return np.asarray(logits) @ basis


def classification_statistics(logits, labels=None, *, batch_size=128):
    """Summarize (chains, draws, examples, classes) raw logit draws on CPU.

    Only batch_size draws are converted to probabilities at a time. Returned
    variances use the posterior empirical measure (ddof=0). Winner frequencies
    estimate the accuracy of a sampled *network*, not sampled categorical labels.
    If labels are supplied, accuracy_draws retains the chain/draw axes for MCSE.
    No independence assumption is made between examples in a network draw.
    """
    if len(logits.shape) != 4 or min(logits.shape) < 1 or logits.shape[-1] < 2:
        raise ValueError('logits must have shape (chains, draws, examples, classes>=2)')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    chains, draws, points, classes = logits.shape
    if labels is not None:
        labels = _numpy(labels)
        if labels.shape != (points,) or labels.dtype.kind not in 'iu' or np.any((labels<0)|(labels>=classes)):
            raise ValueError('labels must be integer class indices of shape (examples,)')
    mean = np.zeros((points, classes), dtype=np.float64)
    m2, winners = np.zeros_like(mean), np.zeros_like(mean)
    accuracies = np.empty((chains, draws)) if labels is not None else None
    count = 0
    for chain in range(chains):
        for start in range(0, draws, batch_size):
            values = _numpy(logits[chain, start:start+batch_size]).astype(np.float64, copy=False)
            if not np.isfinite(values).all():
                raise ValueError('logit draws must be finite')
            probabilities = softmax(values, axis=-1)
            batch_mean = probabilities.mean(axis=0)
            n = len(values)
            delta = batch_mean-mean
            m2 += np.square(probabilities-batch_mean).sum(axis=0) + delta**2*count*n/(count+n)
            mean += delta*n/(count+n)
            count += n
            prediction = values.argmax(axis=-1)
            for c in range(classes):
                winners[:, c] += (prediction==c).sum(axis=0)
            if labels is not None:
                accuracies[chain, start:start+n] = (prediction==labels).mean(axis=-1)
    result = dict(probabilities=mean, probability_variance=m2/count,
                  argmax_probabilities=winners/count)
    if labels is not None:
        result.update(mean_predictor_accuracy=float(np.mean(mean.argmax(-1)==labels)),
                      posterior_sample_accuracy=float(accuracies.mean()), accuracy_draws=accuracies)
    return result


def sample_categorical_labels(logits, *, seed=0):
    """Draw one categorical observation per logit row, using a local NumPy RNG.

    This includes observation randomness; it is not a posterior network's argmax
    prediction. The output has logits.shape[:-1], retaining any chain/draw axes.
    """
    values = _numpy(logits)
    if values.ndim < 1 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError('finite logits with >=2 classes are required')
    probabilities = softmax(values, axis=-1)
    u = np.random.default_rng(seed).random(values.shape[:-1])
    return np.minimum((u[...,None] > probabilities.cumsum(-1)).sum(-1), values.shape[-1]-1)


def iter_predictions(network, theta_draws, X, *, batch_size=None):
    """Evaluate saved whitened weights without loading the whole trajectory.

    network is a WhitenedNetwork or NetworkPosterior. theta_draws has shape
    (chains, draws, dimension), possibly a memory map. Yields (chain, draw,
    CPU_numpy_outputs), suitable for a streaming analysis or an output file.
    Prediction batches use the same weight draw across all test examples.
    """
    if len(theta_draws.shape) != 3 or theta_draws.shape[-1] != network.dimension:
        raise ValueError('theta_draws must have shape (chains, draws, dimension)')
    for chain in range(theta_draws.shape[0]):
        for draw in range(theta_draws.shape[1]):
            with torch.no_grad():
                value = theta_draws[chain, draw]
                # copy() avoids unsafe writable views of read-only memory maps.
                theta = torch.as_tensor(_numpy(value).copy(), device=network.device, dtype=network.dtype)
                output = network.predict(theta, X, batch_size).cpu().numpy()
            # Never leave the caller's thread in no_grad while the generator pauses.
            yield chain, draw, output
