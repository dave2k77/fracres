"""Signal metrics: long-range dependence and criticality diagnostics.

Tools to *characterise* the trajectories a reservoir generates and compare them to
the driving process -- the empirical side of the framework (knowledge base
Sections 4 and on criticality). Two families:

Long-range dependence (LRD)
    :func:`hurst_dfa` (detrended fluctuation analysis) and :func:`spectral_exponent`
    (log-periodogram slope) estimate the self-similarity of a 1-D signal. For
    *fractional Gaussian noise* -- what :func:`fracres.drivers.generate_fbm_increments`
    produces -- the two are tied to the Hurst exponent by

        DFA exponent = H,        PSD ~ f^{-beta} with beta = 2H - 1,

    so ``H = 1/2`` (white noise) gives ``DFA = 0.5``, ``beta = 0``; a persistent
    ``H = 0.7`` gives ``DFA = 0.7``, ``beta = 0.4``. :func:`signal_metrics` bundles
    both and reports the spectral estimate back on the ``H`` scale for comparison.

Criticality
    Near a critical point, event sizes and durations are power-law distributed
    (Beggs & Plenz 2003, neuronal avalanches). :func:`detect_avalanches` segments a
    non-negative activity trace into supra-threshold excursions; :func:`power_law_exponent`
    is the Clauset-Shalizi-Newman maximum-likelihood exponent; :func:`avalanche_exponents`
    combines them into the size/duration exponents ``(tau, alpha)`` (critical
    cortex sits near ``tau ~ 1.5``, ``alpha ~ 2.0``).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


# --- long-range dependence ----------------------------------------------------

def _log_spaced_scales(n: int, s_min: int = 16, s_max: int | None = None, num: int = 20):
    """Unique, log-spaced integer window sizes in ``[s_min, s_max]`` (default ``n//8``).

    The bounds avoid DFA-1's two finite-sample artefacts: a small-scale crossover
    that biases the slope low (hence ``s_min = 16``, not 4) and the high-variance
    largest scales where only a handful of windows fit (hence ``s_max = n//8``,
    keeping >= 8 windows). Resolving very high ``H`` (~0.9) still needs long series.
    """
    if s_max is None:
        s_max = max(2 * s_min, n // 8)
    raw = np.logspace(math.log10(s_min), math.log10(s_max), num)
    return np.unique(np.round(raw).astype(int))


def dfa_fluctuation(x, scales=None, order: int = 1):
    """Detrended-fluctuation curve ``(scales, F(s))`` for a 1-D signal.

    The integrated, mean-removed profile is split into non-overlapping windows of
    length ``s``; within each a degree-``order`` polynomial trend is removed and
    the root-mean-square residual accumulated, giving ``F(s)``. For a self-similar
    signal ``F(s) ~ s^H``. ``order=1`` (DFA-1) removes linear trends.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    profile = np.cumsum(x - x.mean())
    if scales is None:
        scales = _log_spaced_scales(n)
    scales = [int(s) for s in scales if 2 * (order + 1) <= int(s) <= n]

    fluct = []
    for s in scales:
        nseg = n // s
        segs = profile[: nseg * s].reshape(nseg, s)  # (nseg, s)
        t = np.arange(s)
        V = np.vander(t, order + 1)  # (s, order+1)
        coeffs, *_ = np.linalg.lstsq(V, segs.T, rcond=None)  # (order+1, nseg)
        resid = segs.T - V @ coeffs  # (s, nseg)
        fluct.append(float(np.sqrt(np.mean(resid**2))))
    return np.asarray(scales), np.asarray(fluct)


def hurst_dfa(x, scales=None, order: int = 1) -> float:
    """DFA self-similarity exponent (= Hurst ``H`` for fGn).

    Slope of ``log F(s)`` vs ``log s`` from :func:`dfa_fluctuation`. White noise
    gives ``~0.5``; persistent long memory gives ``> 0.5``.
    """
    scales, fluct = dfa_fluctuation(x, scales, order)
    slope, _ = np.polyfit(np.log(scales), np.log(fluct), 1)
    return float(slope)


def power_spectral_density(x, n_segments: int = 8):
    """Welch power spectral density ``(freqs, psd)`` (Hann window, no overlap, DC dropped).

    Averaging over ``n_segments`` segments tames the periodogram's variance so the
    log-log slope fit is stable. Frequencies are in cycles/sample.
    """
    x = np.asarray(x, dtype=float).ravel()
    seg_len = x.shape[0] // n_segments
    if seg_len < 4:
        raise ValueError("signal too short for n_segments; reduce n_segments.")
    window = np.hanning(seg_len)
    scale = np.sum(window**2)
    freqs = np.fft.rfftfreq(seg_len)
    psd = np.zeros(freqs.shape[0])
    for i in range(n_segments):
        seg = x[i * seg_len : (i + 1) * seg_len]
        seg = (seg - seg.mean()) * window
        psd += np.abs(np.fft.rfft(seg)) ** 2 / scale
    psd /= n_segments
    return freqs[1:], psd[1:]  # drop DC


def spectral_exponent(x, f_max: float = 0.1, n_bins: int = 20) -> float:
    """Spectral slope ``beta`` where ``PSD ~ f^{-beta}`` (= ``2H - 1`` for fGn).

    Log-binned log-periodogram regression. The raw periodogram is restricted to the
    low-frequency band ``f <= f_max`` -- essential, because fGn is a power law only
    as ``f -> 0`` and its spectrum flattens at high frequency, so a full-band fit
    underestimates ``beta`` (severely for strongly persistent signals) -- then
    averaged within ``n_bins`` log-spaced frequency bins before the least-squares
    fit. Binning tames the periodogram's huge per-ordinate variance (an order of
    magnitude tighter than an unbinned fit). A small residual downward bias of
    ``~0.1`` is the usual finite-sample behaviour of log-periodogram estimators.
    The full Welch spectrum is available separately via :func:`power_spectral_density`.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(n)
    f, p = f[1:], p[1:]  # drop DC
    band = f <= f_max
    f, p = f[band], p[band]
    if f.shape[0] < 2:
        raise ValueError("signal too short / f_max too small for a spectral fit.")
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), n_bins + 1)
    idx = np.digitize(f, edges)
    f_bin, p_bin = [], []
    for b in range(1, n_bins + 1):
        sel = idx == b
        if np.any(sel):
            f_bin.append(np.exp(np.mean(np.log(f[sel]))))  # geometric-mean freq
            p_bin.append(np.mean(p[sel]))
    slope, _ = np.polyfit(np.log(f_bin), np.log(p_bin), 1)
    return float(-slope)


class SignalMetrics(NamedTuple):
    """Long-range-dependence summary of a 1-D signal."""

    hurst_dfa: float  # DFA exponent (= H for fGn)
    spectral_beta: float  # PSD ~ f^{-beta}
    hurst_spectral: float  # H implied by beta, via H = (beta + 1) / 2 (fGn)


def signal_metrics(x, n_segments: int = 8) -> SignalMetrics:
    """Bundle :func:`hurst_dfa` and :func:`spectral_exponent`, with the spectral
    estimate mapped back onto the Hurst scale (``H = (beta + 1)/2`` for fGn) so the
    two estimators can be compared directly, and against the drive's known ``H``.
    """
    beta = spectral_exponent(x, n_segments)
    return SignalMetrics(
        hurst_dfa=hurst_dfa(x),
        spectral_beta=beta,
        hurst_spectral=0.5 * (beta + 1.0),
    )


# --- criticality / avalanches -------------------------------------------------

def detect_avalanches(activity, threshold=None):
    """Segment a non-negative 1-D activity trace into avalanches.

    An avalanche is a maximal run of consecutive bins whose activity exceeds
    ``threshold`` (default: the median), bracketed by sub-threshold quiescence. Its
    *size* is the summed supra-threshold excess and its *duration* the number of
    bins.

    Returns
    -------
    (sizes, durations) : tuple of arrays
        One entry per detected avalanche (empty if none).
    """
    a = np.asarray(activity, dtype=float).ravel()
    if threshold is None:
        threshold = float(np.median(a))
    active = a > threshold
    # Run boundaries via the rising/falling edges of the padded boolean mask.
    edges = np.diff(np.concatenate(([0], active.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    sizes = np.array([float(np.sum(a[s:e] - threshold)) for s, e in zip(starts, ends)])
    durations = np.array([int(e - s) for s, e in zip(starts, ends)])
    return sizes, durations


def power_law_exponent(values, xmin=None) -> float:
    """Maximum-likelihood power-law exponent (Clauset-Shalizi-Newman, continuous).

    For ``P(x) ~ x^{-mu}`` on ``x >= xmin``, the MLE is
    ``mu = 1 + n / sum_i ln(x_i / xmin)``. ``xmin`` defaults to the smallest
    positive value. Returns ``nan`` if fewer than two samples are at/above ``xmin``.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[v > 0]
    if xmin is None:
        xmin = float(v.min()) if v.size else 1.0
    v = v[v >= xmin]
    if v.size < 2:
        return float("nan")
    return float(1.0 + v.size / np.sum(np.log(v / xmin)))


class AvalancheExponents(NamedTuple):
    """Critical-exponent summary of an activity trace."""

    tau: float  # size exponent  P(S) ~ S^{-tau}      (cortex ~ 1.5)
    alpha: float  # duration exponent P(T) ~ T^{-alpha} (cortex ~ 2.0)
    n_avalanches: int


def avalanche_exponents(activity, threshold=None) -> AvalancheExponents:
    """Avalanche size/duration power-law exponents ``(tau, alpha)``.

    Detects avalanches (:func:`detect_avalanches`) and fits each distribution with
    :func:`power_law_exponent`. A signature of criticality is ``tau ~ 1.5``,
    ``alpha ~ 2.0`` (Beggs & Plenz 2003).
    """
    sizes, durations = detect_avalanches(activity, threshold)
    return AvalancheExponents(
        tau=power_law_exponent(sizes),
        alpha=power_law_exponent(durations.astype(float)),
        n_avalanches=int(sizes.size),
    )
