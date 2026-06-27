"""Stochastic driving processes.

Critical brain dynamics are driven here by *fractional Gaussian noise* (fGn),
the increment process of fractional Brownian motion (fBm). The Hurst exponent
``H`` controls long-range dependence: ``H > 1/2`` is persistent (long memory),
``H = 1/2`` recovers white noise, ``H < 1/2`` is anti-persistent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def generate_fbm_increments(time_steps: int, H: float, key: jax.Array) -> jnp.ndarray:
    """Generate fractional Gaussian noise (fBm increments) via the Davies-Harte method.

    Exact synthesis by circulant embedding of the fGn autocovariance, diagonalised
    with the FFT in ``O(n log n)``.

    Parameters
    ----------
    time_steps : int
        Number of increments to return.
    H : float in (0, 1)
        Hurst exponent of the driving process.
    key : jax.Array
        PRNG key.

    Returns
    -------
    array, shape ``(time_steps,)``
        Unit-variance fGn increments.
    """
    # 1. Autocovariance sequence of fGn.
    idx = jnp.arange(time_steps)
    r = jnp.zeros(time_steps)
    r = r.at[0].set(1.0)
    r = r.at[1:].set(
        0.5
        * (
            (idx[1:] + 1) ** (2 * H)
            - 2 * idx[1:] ** (2 * H)
            + (idx[1:] - 1) ** (2 * H)
        )
    )

    # 2. Circulant embedding; eigenvalues via FFT (clipped for numerical safety).
    c = jnp.concatenate([r, jnp.array([0.0]), r[:0:-1]])
    eigenvalues = jnp.maximum(jnp.fft.fft(c).real, 0.0)

    # 3. Complex Gaussian spectrum.
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (len(eigenvalues),)) + 1j * jax.random.normal(
        k2, (len(eigenvalues),)
    )

    # 4. Back to the time domain; standardise to unit variance.
    fgn = jnp.fft.ifft(z * jnp.sqrt(eigenvalues)).real[:time_steps]
    return fgn / jnp.std(fgn)
