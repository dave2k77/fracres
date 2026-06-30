"""Drive fidelity: the Davies-Harte fGn synthesis (KB v2 §7 item 6).

The strong check is the *autocovariance*: ``generate_fbm_increments`` must
reproduce the exact fGn autocovariance
``r(k) = 1/2 (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})`` with ``r(0) = 1``. Because
the process is zero-mean by construction we estimate the autocovariance
*uncentred* and average over an ensemble of realisations (at large H the
per-draw sample mean drifts far from zero, so a centred estimate is biased).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fracres import generate_fbm_increments  # noqa: E402


def _theoretical_acov(n: int, H: float) -> np.ndarray:
    k = np.arange(n)
    r = np.zeros(n)
    r[0] = 1.0
    r[1:] = 0.5 * (
        (k[1:] + 1) ** (2 * H) - 2 * k[1:] ** (2 * H) + (k[1:] - 1) ** (2 * H)
    )
    return r


def _ensemble(n: int, H: float, n_real: int, seed: int = 0) -> np.ndarray:
    keys = jax.random.split(jax.random.PRNGKey(seed), n_real)
    draws = jax.vmap(lambda k: generate_fbm_increments(n, H, k))(keys)
    return np.asarray(draws)  # (n_real, n)


def _empirical_acov(x: np.ndarray, max_lag: int) -> np.ndarray:
    # Uncentred ensemble autocovariance: mean over realisations and time of
    # x_t * x_{t+k}. The process is zero-mean, so no mean subtraction.
    n = x.shape[1]
    return np.array([np.mean(x[:, : n - k] * x[:, k:]) for k in range(max_lag + 1)])


def test_shape_and_finite():
    fgn = generate_fbm_increments(256, H=0.7, key=jax.random.PRNGKey(1))
    assert fgn.shape == (256,)
    assert jnp.all(jnp.isfinite(fgn))


def test_deterministic_given_key():
    a = generate_fbm_increments(128, H=0.6, key=jax.random.PRNGKey(3))
    b = generate_fbm_increments(128, H=0.6, key=jax.random.PRNGKey(3))
    assert jnp.array_equal(a, b)


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7, 0.9])
def test_autocovariance_matches_theory(H):
    """The realised fGn autocovariance matches the exact target across lags."""
    n, n_real, max_lag = 512, 800, 6
    x = _ensemble(n, H, n_real, seed=11)
    emp = _empirical_acov(x, max_lag)
    theory = _theoretical_acov(n, H)[: max_lag + 1]
    # ~0.01 is comfortable for 800 realisations of length 512 (see scratch
    # verification: worst-case diff < 0.008 even at H=0.9).
    assert np.allclose(emp, theory, atol=0.02), f"H={H}: emp={emp}, theory={theory}"


def test_unit_variance_in_expectation():
    """Ensemble (uncentred) lag-0 power is 1; individual draws genuinely vary."""
    x = _ensemble(1024, H=0.7, n_real=600, seed=5)
    lag0 = float(np.mean(x * x))  # uncentred second moment over the ensemble
    assert np.isclose(lag0, 1.0, atol=0.02)
    # The exact scaling does NOT force each draw to std 1 (the old empirical-std
    # normalisation did): there must be a real spread of per-draw variances.
    per_draw_var = np.mean(x * x, axis=1)
    assert per_draw_var.std() > 0.01


def test_white_noise_at_half():
    """H = 1/2 is white: all non-zero lags vanish in the ensemble."""
    x = _ensemble(512, H=0.5, n_real=800, seed=7)
    emp = _empirical_acov(x, max_lag=5)
    assert np.isclose(emp[0], 1.0, atol=0.02)
    assert np.allclose(emp[1:], 0.0, atol=0.02)


def test_persistence_sign():
    """H > 1/2 gives positive lag-1 correlation; H < 1/2 gives negative."""
    persistent = _empirical_acov(_ensemble(512, 0.8, 600, seed=9), 1)[1]
    antipersistent = _empirical_acov(_ensemble(512, 0.2, 600, seed=9), 1)[1]
    assert persistent > 0.1
    assert antipersistent < -0.1
