"""Predictive-observable diagnostics; ArviZ is optional and imported lazily."""
import numpy as np


def diagnostics_and_mcse(observables, *, chain_info=None, min_ess=400., max_rhat=1.01,
                         min_chains=4):
    """Rank/folded split Rhat, bulk/tail ESS, and autocorrelation-aware MCSE.

    Constant observed coordinates are reported, not silently certified. Four
    independent chains are required by default; min_chains=2 allows an explicit
    two-chain policy while retaining rank-normalized split-Rhat and ESS checks.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError('Install optional diagnostics: pip install "deepbays[sampling]"') from exc
    if not observables:
        raise ValueError('supply at least one predictive observable')
    observables = {name:np.asarray(value) for name,value in observables.items()}
    shape = next(iter(observables.values())).shape[:2]
    if len(shape) != 2 or min(shape) < 1 or shape[1] < 4:
        raise ValueError('observables need (chains, draws>=4, ...) axes')
    if any(value.ndim < 2 or value.shape[:2] != shape or value.size == 0
           or not np.isfinite(value).all() for value in observables.values()):
        raise ValueError('observables must be finite and have matching chain/draw axes')
    if min_ess <= 0 or not np.isfinite(min_ess) or max_rhat < 1 or not np.isfinite(max_rhat):
        raise ValueError('invalid diagnostic thresholds')
    if isinstance(min_chains, bool) or not isinstance(min_chains, int) or min_chains < 2:
        raise ValueError('min_chains must be an integer >=2')
    if chain_info is not None and len(chain_info) != shape[0]:
        raise ValueError('chain_info must correspond to the observed chains')

    posterior = az.from_dict(posterior=observables).posterior
    rhat = az.rhat(posterior, method='rank')
    bulk = az.ess(posterior, method='bulk')
    tail = az.ess(posterior, method='tail')
    mcse = az.mcse(posterior, method='mean')
    rows, errors = {}, {}
    for name, samples in observables.items():
        def finite_stat(array, operation):
            values = np.asarray(array).reshape(-1)
            return float(operation(values)) if np.all(np.isfinite(values)) else None
        rows[name] = dict(max_rhat=finite_stat(rhat[name], np.max),
                          min_bulk_ess=finite_stat(bulk[name], np.min),
                          min_tail_ess=finite_stat(tail[name], np.min),
                          constant_coordinates=int(np.sum(np.var(samples, axis=(0, 1)) == 0)))
        errors[name] = np.asarray(mcse[name])
    chains = next(iter(observables.values())).shape[0]
    passed = chains >= min_chains and all(
        row['constant_coordinates'] == 0
        and row['max_rhat'] is not None and row['max_rhat'] <= max_rhat
        and row['min_bulk_ess'] is not None and row['min_bulk_ess'] >= min_ess
        and row['min_tail_ess'] is not None and row['min_tail_ess'] >= min_ess
        for row in rows.values())
    result = dict(passed=bool(passed), chains=chains, max_rhat_threshold=max_rhat,
                  min_ess_threshold=min_ess, min_chains_threshold=min_chains, observables=rows)
    if chain_info is not None:
        result['divergences'] = sum(row.get('divergences', 0) for row in chain_info)
        result['passed'] &= (result['divergences'] == 0 and all(
            row.get('status', 'complete') == 'complete' for row in chain_info))
    return result, errors
