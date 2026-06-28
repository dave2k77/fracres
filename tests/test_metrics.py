"""Signal metrics: long-range dependence and criticality diagnostics."""
import jax
import numpy as np
import pytest

from fracres import (
    avalanche_exponents,
    detect_avalanches,
    dfa_fluctuation,
    generate_fbm_increments,
    hurst_dfa,
    power_law_exponent,
    power_spectral_density,
    signal_metrics,
    spectral_exponent,
)

N = 8000


def _fgn(H, seed=0, n=N):
    return np.asarray(generate_fbm_increments(n, H=H, key=jax.random.PRNGKey(seed)))


# --- long-range dependence ----------------------------------------------------

def test_dfa_recovers_hurst_of_fgn():
    # DFA exponent of fGn equals its Hurst exponent.
    for H in (0.3, 0.5, 0.7):
        est = np.mean([hurst_dfa(_fgn(H, s)) for s in range(4)])
        assert abs(est - H) < 0.07


def test_dfa_white_noise_is_half():
    assert abs(hurst_dfa(_fgn(0.5)) - 0.5) < 0.07


def test_dfa_increases_with_persistence():
    assert hurst_dfa(_fgn(0.3)) < hurst_dfa(_fgn(0.6)) < hurst_dfa(_fgn(0.9))


def test_dfa_fluctuation_curve_is_increasing():
    scales, fluct = dfa_fluctuation(_fgn(0.7))
    assert len(scales) == len(fluct) > 3
    assert fluct[-1] > fluct[0]  # F(s) grows with scale
    assert np.mean(np.diff(fluct) > 0) > 0.8  # near-monotone (noisy at top scales)


def test_spectral_exponent_tracks_2H_minus_1():
    # beta = 2H - 1 for fGn (with a small finite-sample downward bias).
    for H in (0.5, 0.7, 0.9):
        beta = np.mean([spectral_exponent(_fgn(H, s)) for s in range(4)])
        assert abs(beta - (2 * H - 1)) < 0.15


def test_spectral_exponent_white_noise_is_flat():
    # Average a few realisations; a single periodogram slope is noisy.
    beta = np.mean([spectral_exponent(_fgn(0.5, s)) for s in range(4)])
    assert abs(beta) < 0.12


def test_power_spectral_density_shapes_and_positive():
    f, p = power_spectral_density(_fgn(0.7), n_segments=8)
    assert f.shape == p.shape and f.shape[0] > 0
    assert np.all(p > 0) and np.all(np.diff(f) > 0)
    assert f[0] > 0  # DC dropped


def test_signal_metrics_two_estimators_agree_on_H():
    m = signal_metrics(_fgn(0.7, n=16000))
    assert m.hurst_spectral == pytest.approx(0.5 * (m.spectral_beta + 1.0))
    # DFA and the spectral H-estimate land in the same neighbourhood.
    assert abs(m.hurst_dfa - m.hurst_spectral) < 0.15


# --- criticality / avalanches -------------------------------------------------

def test_detect_avalanches_on_constructed_signal():
    a = np.array([0, 0, 3, 4, 0, 0, 5, 0, 2, 2, 2, 0], dtype=float)
    sizes, durations = detect_avalanches(a, threshold=1.0)
    assert list(durations) == [2, 1, 3]  # three supra-threshold runs
    # size = summed excess above threshold.
    assert sizes == pytest.approx([(3 - 1) + (4 - 1), (5 - 1), 3 * (2 - 1)])


def test_detect_avalanches_default_threshold_is_median():
    a = np.array([0, 0, 0, 0, 10, 0, 0, 0], dtype=float)  # median 0
    sizes, durations = detect_avalanches(a)
    assert list(durations) == [1] and sizes == pytest.approx([10.0])


def test_detect_avalanches_empty_when_all_subthreshold():
    sizes, durations = detect_avalanches(np.ones(10), threshold=5.0)
    assert sizes.size == 0 and durations.size == 0


def test_power_law_exponent_mle_recovers_exponent():
    # Inverse-transform sample x = (1-u)^{-1/(mu-1)} ~ power law on x >= 1.
    rng = np.random.default_rng(0)
    for mu_true in (2.0, 2.5, 3.0):
        x = (1.0 - rng.random(30000)) ** (-1.0 / (mu_true - 1.0))
        assert abs(power_law_exponent(x, xmin=1.0) - mu_true) < 0.06


def test_power_law_exponent_too_few_samples_is_nan():
    assert np.isnan(power_law_exponent(np.array([5.0]), xmin=1.0))


def test_avalanche_exponents_on_reservoir_activity():
    # Rectified fGn-driven activity should yield finite positive exponents.
    from fracres import GLKernel, PhantomBrain

    model = PhantomBrain(1, 100, 1, GLKernel(0.8, 50), key=jax.random.PRNGKey(0))
    drive = _fgn(0.7, n=4000)[:, None]
    X, _ = model.simulate(drive)
    activity = np.asarray(np.mean(np.abs(X), axis=1))  # population activity
    ex = avalanche_exponents(activity)
    assert ex.n_avalanches > 10
    assert ex.tau > 1.0 and ex.alpha > 1.0  # valid power-law exponents
