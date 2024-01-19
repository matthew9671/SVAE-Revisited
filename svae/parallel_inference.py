import matplotlib.pyplot as plt
import jax
import jax.random as jr
import jax.numpy as np
import jax.scipy as scipy
from jax import lax, jit, vmap
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance
from svae.utils import psd_solve

# Generic typing stuff.
from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)
from jax._src.random import KeyArray as PRNGKey

def _make_associative_filtering_elements(params, potentials, inputs):
    """Preprocess observations to construct input for filtering assocative scan."""

    F = params["A"]
    Q = params["Q"]
    Q1 = params["Q1"]
    P0 = Q1
    P1 = Q1
    dim = Q.shape[0]
    H = np.eye(dim)

    def _first_filtering_element(mu, Sigma, m1):

        y, R = mu, Sigma

        S = H @ Q @ H.T + R
        CF, low = scipy.linalg.cho_factor(S)

        S1 = H @ P1 @ H.T + R
        K1 = psd_solve(S1, H @ P1).T

        A = np.zeros_like(F)
        b = m1 + K1 @ (y - H @ m1)
        C = P1 - K1 @ S1 @ K1.T
        eta = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), (y - H @ m1))
        J = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), H @ F)

        return A, b, C, J, eta

    def _generic_filtering_element(mu, Sigma, u):

        y, R = mu, Sigma

        S = H @ Q @ H.T + R
        CF, low = scipy.linalg.cho_factor(S)
        K = scipy.linalg.cho_solve((CF, low), H @ Q).T
        A = F - K @ H @ F
        b = u + K @ (y - H @ u)
        C = Q - K @ H @ Q

        eta = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), (y - H @ u))
        J = F.T @ H.T @ scipy.linalg.cho_solve((CF, low), H @ F)

        return A, b, C, J, eta

    mus, Sigmas = potentials["mu"], potentials["Sigma"]

    first_elems = _first_filtering_element(mus[0], Sigmas[0], inputs[0])
    generic_elems = vmap(_generic_filtering_element)(mus[1:], Sigmas[1:], inputs[1:])
    combined_elems = tuple(np.concatenate((first_elm[None ,...], gen_elm))
                           for first_elm, gen_elm in zip(first_elems, generic_elems))
    return combined_elems


def lgssm_filter(params, emissions, inputs):
    """A parallel version of the lgssm filtering algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """

    initial_elements = _make_associative_filtering_elements(params, emissions, inputs)

    @vmap
    def filtering_operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2
        dim = A1.shape[0]
        I = np.eye(dim)

        I_C1J2 = I + C1 @ J2
        temp = scipy.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = temp @ C1 @ A2.T + C2

        I_J2C1 = I + J2 @ C1
        temp = scipy.linalg.solve(I_J2C1.T, A1).T

        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = temp @ J2 @ A1 + J1

        return A, b, C, J, eta

    _, filtered_means, filtered_covs, *_ = lax.associative_scan(
        filtering_operator, initial_elements
    )

    return {
        "marginal_loglik": lgssm_log_normalizer(params, filtered_means, filtered_covs,
                                                emissions, inputs),
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs
    }


def _make_associative_smoothing_elements(params, key, filtered_means, filtered_covariances, inputs, sample_shape = ()):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    F = params["A"]
    Q = params["Q"]

    num_timesteps = len(filtered_means)
    dims = filtered_means.shape[-1]
    keys = jr.split(key, num_timesteps)

    def _last_smoothing_element(key, m, P):
        h = MVN(m, P).sample(seed=key, sample_shape=sample_shape)
        return np.zeros_like(P), m, P, h

    def _generic_smoothing_element(key, m, P, u):

        eps = 1e-6
        P += np.eye(dims) * eps
        Pp = F @ P @ F.T + Q

        E  = psd_solve(Pp, F @ P).T
        g  = m - E @ (F @ m + u)
        L  = P - E @ Pp @ E.T

        L = (L + L.T) * .5 + np.eye(dims) * eps # Add eps to the crucial covariance matrix

        h = MVN(g, L).sample(seed=key, sample_shape=sample_shape)
        return E, g, L, h

    last_elems = _last_smoothing_element(keys[-1], filtered_means[-1],
                                         filtered_covariances[-1])
    # Note the indexing of inputs here
    generic_elems = vmap(_generic_smoothing_element) \
        (keys[:-1], filtered_means[:-1], filtered_covariances[:-1], inputs[1:])
    combined_elems = tuple(np.append(gen_elm, last_elm[None,:], axis=0)
                           for gen_elm, last_elm in zip(generic_elems, last_elems))
    return combined_elems


def real_lgssm_smoother(params,
                        emissions,
                        inputs,
                        key = None,
                        sample_shape = ()):
    """A parallel version of the lgssm smoothing algorithm.
    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    Note: This function does not yet handle `inputs` to the system.
    """
    # Default key
    if key is None:
        key = jr.PRNGKey(0)

    filtered_posterior = lgssm_filter(params, emissions, inputs)
    filtered_means = filtered_posterior["filtered_means"]
    filtered_covs = filtered_posterior["filtered_covariances"]
    initial_elements = _make_associative_smoothing_elements(params, key, filtered_means,
                                                            filtered_covs, inputs,
                                                            sample_shape=sample_shape)

    @vmap
    def smoothing_operator(elem1, elem2):
        E1, g1, L1, h1 = elem1
        E2, g2, L2, h2 = elem2

        E = E2 @ E1
        g = E2 @ g1 + g2
        L = E2 @ L1 @ E2.T + L2
        h = np.einsum("ji,...i->...j", E2, h1) + h2

        return E, g, L, h

    _, smoothed_means, smoothed_covs, samples = lax.associative_scan(
        smoothing_operator, initial_elements, reverse=True)
    return {
        "samples": samples,
        "marginal_loglik": filtered_posterior["marginal_loglik"],
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "smoothed_means": smoothed_means,
        "smoothed_covariances": smoothed_covs
    }


def lgssm_log_normalizer(dynamics_params,
                         mu_filtered,
                         Sigma_filtered,
                         potentials,
                         inputs):
    p = dynamics_params
    Q, A = p["Q"][None], p["A"][None]
    AT = (p["A"].T)[None]

    I = np.eye(Q.shape[-1])

    Sigma_filtered, mu_filtered = Sigma_filtered[:-1], mu_filtered[:-1]
    Sigma = Q + A @ Sigma_filtered @ AT
    mu = (A[0] @ mu_filtered.T).T + inputs[1:]
    # Append the first element
    Sigma_pred = np.concatenate([p["Q1"][None], Sigma])
    mu_pred = np.concatenate([inputs[:1], mu])
    mu_rec, Sigma_rec = potentials["mu"], potentials["Sigma"]

    def log_Z_single(mu_pred, Sigma_pred, mu_rec, Sigma_rec):
        return MVN(loc=mu_pred, covariance_matrix=Sigma_pred+Sigma_rec).log_prob(mu_rec)

    log_Z = vmap(log_Z_single)(mu_pred, Sigma_pred, mu_rec, Sigma_rec)
    return np.sum(log_Z)
