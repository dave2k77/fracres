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
    with the FFT in ``O(n log n)``. The output is scaled by the deterministic
    Davies-Harte factor :math:`\\sqrt{m}` (``m`` the embedding length), so the
    realised process has the exact fGn autocovariance
    :math:`r(k)=\\tfrac12(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H})` and **unit variance in
    expectation** (:math:`r(0)=1`). It deliberately does *not* divide by the
    per-realisation empirical std: that rescales every draw by a random factor,
    which biases the variance (the empirical std is itself correlated with the
    sample, badly so at large ``H`` where effective d.o.f. are few) and flattens
    the very long-range dependence the drive exists to inject. Individual draws
    therefore show a genuine variance spread about 1 -- this is correct.

    Parameters
    ----------
    time_steps : int
        Number of increments to return.
    H : float in (0, 1)
        Hurst exponent of the driving process. ``H = 1/2`` recovers white noise;
        ``H > 1/2`` is persistent (long memory), ``H < 1/2`` anti-persistent.
    key : jax.Array
        PRNG key.

    Returns
    -------
    array, shape ``(time_steps,)``
        fGn increments with the exact fGn autocovariance (unit variance in
        expectation).
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

    # 2. Circulant embedding (length m = 2 * time_steps); eigenvalues via FFT.
    #    Clipped to >= 0 for numerical safety: the fGn embedding is non-negative
    #    definite, so any negative eigenvalue is float round-off, not signal.
    c = jnp.concatenate([r, jnp.array([0.0]), r[:0:-1]])
    eigenvalues = jnp.maximum(jnp.fft.fft(c).real, 0.0)
    m = eigenvalues.shape[0]

    # 3. Complex Gaussian spectrum (each component has unit-variance real & imag
    #    parts). Re(ifft(.)) of a length-m independent complex normal weighted by
    #    sqrt(eigenvalues) reproduces the circulant covariance exactly.
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (m,)) + 1j * jax.random.normal(k2, (m,))

    # 4. Back to the time domain with the exact Davies-Harte scaling sqrt(m)
    #    (ifft carries 1/m): Var(fgn_j) = (1/m) * sum_k eigenvalues_k = r(0) = 1.
    fgn = jnp.sqrt(m) * jnp.fft.ifft(z * jnp.sqrt(eigenvalues)).real[:time_steps]
    return fgn
