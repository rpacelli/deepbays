"""Coherent mean-separated, frozen-IW path features.

The rate and optimization variables are unchanged. The activation mean is
lifted coherently over prefix offsets, preserving its rank at insertion.
This is a positive covariance ansatz, not the full NNGP Jacobian or an exact
nonlinear finite-width prior. See docs/cnn_mean_path.md.
"""
import numpy as np


def centered_gain_and_mean(act, left_variance, covariance, right_variance):
    """C/Z and mu(left)*mu(right), with continuous zero-covariance limits."""
    lv, rv, z = left_variance, right_variance, covariance
    if act == 'id':
        return np.ones_like(z), np.zeros_like(z)
    if act == 'erf':
        scale = 1/np.sqrt((1+2*lv)*(1+2*rv))
        rho = np.clip(2*scale*z, -1., 1.)
        ratio = np.ones_like(rho)
        np.divide(np.arcsin(rho), rho, out=ratio, where=rho != 0)
        return 4/np.pi*scale*ratio, np.zeros_like(z)
    if act != 'relu':
        raise ValueError("mean path features support 'id', 'erf', and 'relu'")
    scale = np.sqrt(lv*rv)
    rho = np.zeros_like(z)
    np.divide(z, scale, out=rho, where=scale > 0)
    rho = np.clip(rho, -1., 1.)
    # Stable form of (sqrt(1-rho^2)-1)/rho, including rho=0.
    gain = .25+(np.arcsin(rho)-rho/(1+np.sqrt(1-rho*rho)))/(2*np.pi)
    return np.where(scale > 0, gain, 0.), np.broadcast_to(scale/(2*np.pi), z.shape)


def mean_path_block(features, left, right, *, paired=False):
    """Construct H for full raw paths, then perform any GAP contraction."""
    builder = features.builder
    xp, xv = left
    yp, yv = right
    covariance = (np.einsum('aif,ajf->aij', xp, yp) if paired else
                  np.einsum('aif,bjf->abij', xp, yp))

    def lift(matrix, layer):
        ids = features.visited[:, layer]
        valid = features.valid_at_layer[:, layer]
        return matrix[..., ids[:, None], ids[None, :]]*(valid[:, None]*valid[None, :])

    lifted, prefix_count = None, 1
    for l, layer in enumerate(builder.layers):
        if l:
            covariance = builder._convolve(covariance, l)
        lv = xv[l][:, :, None] if paired else xv[l][:, None, :, None]
        rv = yv[l][:, None, :] if paired else yv[l][None, :, None, :]
        if l == 0:
            covariance = builder._activation(lv, covariance, rv)
            lifted = lift(covariance, l)
        else:
            gain, mean = centered_gain_and_mean(builder.act, lv, covariance, rv)
            lifted *= lift(gain, l)/(layer.area*builder.priors[l])
            prefix_count *= layer.area
            if builder.act == 'relu':
                lifted += lift(mean, l)/prefix_count
            covariance = builder._activation(lv, covariance, rv)
    # No final all-layers mask: a later activation mean must survive when
    # its preceding auxiliary prefix slot lies outside the padded image.
    p = builder.d
    if features.pooling == 'avg':
        r = features.ranks[0]
        lifted = lifted.reshape(lifted.shape[:-2]+(r, p, r, p)).sum(axis=(-3, -1))
    lifted *= builder.output_scale/(p**2 if features.pooling == 'avg' else p)
    if not np.all(np.isfinite(lifted)):
        raise FloatingPointError('non-finite coherent mean-path feature kernel')
    return lifted
