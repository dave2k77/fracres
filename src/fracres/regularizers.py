"""Besov-space regularisation via the Littlewood-Paley decomposition.

To keep generated trajectories on the correct scale-free regularity manifold we
penalise their Besov norm ``||y||_{B^s_{p,q}}``. Rather than evaluate the
quadratic Gagliardo double-integral (``O(n^2)``), we use the equivalent
Littlewood-Paley characterisation: dyadic frequency-band filtering, computable in
``O(n log n)`` with FFTs (knowledge base Section 4).

Regularity bounds (enforced by the caller, see :mod:`fracres.training`):
    integrability  p < alpha_S        (stability index of the heavy-tailed drive;
                                        = 2 for Gaussian fGn -- NOT the derivative order)
    smoothness     s < min{H, 1/p}    (fBm Hurst intersected with the Besov
                                        embedding scale 1/p)

See knowledge base v2 §4.2 for the derivation of these bounds.
"""
from __future__ import annotations

import jax.numpy as jnp


def make_dyadic_masks(time_steps: int, n_bands: int | None = None) -> jnp.ndarray:
    """Build dyadic frequency-band masks for the Littlewood-Paley decomposition.

    Band ``j`` (1-indexed) selects FFT bins with ``2^{j-1} <= |k| < 2^j``.

    Parameters
    ----------
    time_steps : int
        Length of the time axis (number of FFT bins).
    n_bands : int, optional
        Number of dyadic bands ``J``. Defaults to ``floor(log2(time_steps))``.

    Returns
    -------
    array, shape ``(n_bands, time_steps)``
        Float (0/1) band-pass masks aligned with ``jnp.fft.fft`` bin ordering.
    """
    if n_bands is None:
        n_bands = max(1, int(jnp.floor(jnp.log2(time_steps))))
    k = jnp.abs(jnp.fft.fftfreq(time_steps) * time_steps)  # |bin index|, 0..n/2
    j = jnp.arange(1, n_bands + 1)[:, None]
    lo = 2.0 ** (j - 1)
    hi = 2.0**j
    return ((k[None, :] >= lo) & (k[None, :] < hi)).astype(jnp.float32)


def dyadic_band_energies(
    Y_hat: jnp.ndarray, masks: jnp.ndarray, p: float = 2.0
) -> jnp.ndarray:
    """``L^p`` norm ``||Delta_j Y||_p`` of each dyadic frequency block ``Delta_j Y``.

    The building block of the Besov norm: the signal is split into dyadic
    frequency bands (via :func:`make_dyadic_masks`) and the ``L^p`` size of each is
    returned. For a signal of Besov smoothness ``r`` these decay like
    ``2^{-j r}``, so ``-d log2(||Delta_j Y||_p) / dj`` estimates the regularity --
    which is what makes this a direct diagnostic for the regulariser's effect.

    Parameters
    ----------
    Y_hat : array, shape ``(time_steps, out_features)``
    masks : array, shape ``(n_bands, time_steps)``
    p : float, default 2.0
        Integrability index of the band norm.

    Returns
    -------
    array, shape ``(n_bands,)``
        The per-band ``L^p`` norms (summed over the output features).
    """
    Y_fft = jnp.fft.fft(Y_hat, axis=0)  # (T, F)
    filtered_fft = masks[..., None] * Y_fft[None, ...]  # (J, T, F)
    delta_j = jnp.real(jnp.fft.ifft(filtered_fft, axis=1))  # (J, T, F)
    # Stabiliser is added to the base (|delta| + eps)^p rather than to the
    # summand, which keeps the gradient well-behaved near zero when p < 1.
    return jnp.sum((jnp.abs(delta_j) + 1e-12) ** p, axis=(1, 2)) ** (1.0 / p)


def littlewood_paley_penalty(
    Y_hat: jnp.ndarray, masks: jnp.ndarray, s: float, p: float, q: float
) -> jnp.ndarray:
    """Compute the ``B^s_{p,q}`` Besov-norm regulariser of a trajectory.

    The discrete Littlewood-Paley norm ``( sum_j (2^{js} ||Delta_j Y||_p)^q )^{1/q}``:
    each dyadic band energy (:func:`dyadic_band_energies`) is weighted by ``2^{js}``
    -- which grows with the band index ``j`` for ``s > 0``, so high-frequency
    content is penalised more heavily -- then aggregated in ``l^q``.

    Parameters
    ----------
    Y_hat : array, shape ``(time_steps, out_features)``
        Predicted/generated trajectory.
    masks : array, shape ``(n_bands, time_steps)``
        Dyadic frequency masks from :func:`make_dyadic_masks`.
    s, p, q : float
        Besov smoothness, integrability, and summability indices.

    Returns
    -------
    scalar
        The aggregated Besov-norm penalty.
    """
    Lp_norms = dyadic_band_energies(Y_hat, masks, p)
    j_indices = jnp.arange(1, masks.shape[0] + 1)
    scaled_norms = (2.0 ** (j_indices * s)) * Lp_norms
    return jnp.sum(scaled_norms**q) ** (1.0 / q)
