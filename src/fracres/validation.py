"""Analytic references for validating the fractional kernels.

The fractional derivative of a power law has a closed form (Podlubny 1999,
Eq. 2.117); for ``beta > -1`` the Riemann-Liouville and Caputo derivatives of
``t^beta`` (with ``t^beta -> 0`` as ``t -> 0`` for ``beta > 0``) coincide and
equal

    D^alpha t^beta = Gamma(beta + 1) / Gamma(beta + 1 - alpha) * t^(beta - alpha).

This is the ground truth that :meth:`fracres.kernels.AbstractFractionalKernel.apply`
is checked against (knowledge base v2 Section 7.1). Empirically the package
kernels reproduce it at the textbook orders: Grünwald-Letnikov is ``O(h)`` and
the L1 scheme is ``O(h^{2-alpha})``.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gamma


def analytic_power_law_derivative(t, alpha: float, beta: float):
    """Closed-form ``D^alpha t^beta = Gamma(b+1)/Gamma(b+1-a) * t^(b-a)``.

    Parameters
    ----------
    t : array or float
        Evaluation times (``> 0``).
    alpha : float
        Fractional-derivative order ``alpha_D``.
    beta : float
        Power-law exponent (``> -1``).
    """
    coeff = gamma(beta + 1.0) / gamma(beta + 1.0 - alpha)
    return coeff * jnp.asarray(t) ** (beta - alpha)


def convergence_order(err_coarse: float, err_fine: float, refinement: float = 2.0) -> float:
    """Empirical order ``p`` from two errors at grid spacings differing by ``refinement``.

    ``err ~ h^p`` implies ``p = log(err_coarse / err_fine) / log(refinement)``.
    """
    return float(jnp.log(err_coarse / err_fine) / jnp.log(refinement))
