"""Optional Pyro posterior sampling for CPU and CUDA.

Install ``deepbays[sampling]``. Pyro and ArviZ are loaded only when used.
See docs/sampling.md for priors, exact likelihood chunking, storage and diagnostics.
"""
from .posteriors import (WhitenedNetwork, NetworkPosterior, GaussianFunctionPosterior,
                         PotentialPosterior, negative_log_likelihood, covariance_root)
from .mcmc import SamplerConfig, sample_posterior
from .storage import SamplingResult
from .predictive import (classification_statistics, logit_contrasts,
                         sample_categorical_labels, iter_predictions)
from .diagnostics import diagnostics_and_mcse

__all__ = ['WhitenedNetwork', 'NetworkPosterior', 'GaussianFunctionPosterior',
           'PotentialPosterior', 'negative_log_likelihood', 'covariance_root',
           'SamplerConfig', 'sample_posterior', 'SamplingResult',
           'classification_statistics', 'logit_contrasts', 'sample_categorical_labels',
           'iter_predictions', 'diagnostics_and_mcse']
