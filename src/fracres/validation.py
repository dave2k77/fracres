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


def mittag_leffler(z, alpha: float, terms: int = 128):
    """One-parameter Mittag-Leffler ``E_alpha(z) = sum_k z^k / Gamma(a k + 1)``.

    The eigenfunction of the Caputo derivative: ``x(t) = E_alpha(lambda t^alpha)``
    solves the fractional relaxation equation ``D^alpha_C x = lambda x`` with
    ``x(0) = 1``. Used to validate the kernels on a signal with non-zero initial
    value (unlike the power law), which exercises the Caputo-vs-Riemann-Liouville
    distinction.

    Truncated power series (``terms`` terms). Accurate for moderate ``|z|`` in
    float64; for large ``|z|`` the alternating series loses precision -- keep
    ``lambda t^alpha`` within a few units (or raise ``terms``). For ``alpha = 0.5``
    there is the closed form ``E_{1/2}(z) = exp(z^2) erfc(-z)`` (used in tests).

    Parameters
    ----------
    z : array or float
        Argument(s).
    alpha : float
        Order (``> 0``).
    terms : int
        Number of series terms.
    """
    z = jnp.asarray(z)
    k = jnp.arange(terms)  # integer exponents preserve negative-base sign
    return jnp.sum(z[..., None] ** k / gamma(alpha * k + 1.0), axis=-1)


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


def convergence_order(
    err_coarse: float, err_fine: float, refinement: float = 2.0
) -> float:
    """Empirical order ``p`` from two errors at grids differing by ``refinement``.

    ``err ~ h^p`` implies ``p = log(err_coarse / err_fine) / log(refinement)``.
    """
    return float(jnp.log(err_coarse / err_fine) / jnp.log(refinement))
