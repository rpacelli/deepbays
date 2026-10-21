"""Optional Pyro posterior and tilted-prior sampling for CPU and CUDA.

Install ``deepbays[sampling]``. Pyro and ArviZ are loaded only when used.
See docs/sampling.md for priors, exact likelihood chunking, storage and diagnostics.
See docs/tilted_prior_sampling.md for kernel-projection tails and rate estimates.
"""
from .posteriors import (WhitenedNetwork, NetworkPosterior, GaussianFunctionPosterior,
                         PotentialPosterior, negative_log_likelihood, covariance_root)
from .mcmc import SamplerConfig, sample_posterior
from .storage import SamplingResult
from .predictive import (classification_statistics, logit_contrasts,
                         sample_categorical_labels, iter_predictions)
from .diagnostics import diagnostics_and_mcse
from .tilted_prior import KernelTailTarget, PowerTilt, RootTilt, sample_tilted_prior
from .rate_functions import (central_ewa_log_normalizer, central_ewa_rate,
                             estimate_density_rate, estimate_prior_density_rate,
                             estimate_left_probability)

__all__ = ['WhitenedNetwork', 'NetworkPosterior', 'GaussianFunctionPosterior',
           'PotentialPosterior', 'negative_log_likelihood', 'covariance_root',
           'SamplerConfig', 'sample_posterior', 'SamplingResult',
           'classification_statistics', 'logit_contrasts', 'sample_categorical_labels',
           'iter_predictions', 'diagnostics_and_mcse',
           'KernelTailTarget', 'PowerTilt', 'RootTilt', 'sample_tilted_prior',
           'central_ewa_log_normalizer', 'central_ewa_rate', 'estimate_density_rate',
           'estimate_prior_density_rate', 'estimate_left_probability']
