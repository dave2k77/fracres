"""Validation: the Besov (Littlewood-Paley) regulariser moves trajectories toward
the target ``B^s_{p,q}`` regularity (knowledge base Section 4).

Beyond checking the penalty *computes* a Besov norm, these tests check the
operational claim: adding ``lambda * B^s_{p,q}`` to a fit makes the trajectory
smoother, and the *target* smoothness index ``s`` controls how smooth.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fracres import (
    dyadic_band_energies,
    generate_fbm_increments,
    littlewood_paley_penalty,
    make_dyadic_masks,
)

T = 256
KEY = jax.random.PRNGKey(0)
MASKS = make_dyadic_masks(T)
J = MASKS.shape[0]


def _rough_signal(seed=0):
    y = generate_fbm_increments(T, H=0.5, key=jax.random.PRNGKey(seed))[:, None]
    return y / jnp.std(y)


def _hf_fraction(y):
    """Fraction of total band energy in the upper-half (high-frequency) bands."""
    e = dyadic_band_energies(y, MASKS)
    return float(jnp.sum(e[J // 2:]) / jnp.sum(e))


def _smoothness(y):
    """Estimated regularity: ``-`` slope of ``log2 ||Delta_j y||`` vs band ``j``."""
    e = np.asarray(dyadic_band_energies(y, MASKS))
    return float(-np.polyfit(np.arange(1, J + 1), np.log2(e + 1e-20), 1)[0])


def _fit(y0, lam, s, p=2.0, q=2.0, steps=300, lr=5e-2):
    """Minimise ``MSE(y, y0) + lam * B^s_{p,q}(y)`` over a free trajectory ``y``."""
    y = y0
    opt = optax.adam(lr)
    state = opt.init(y)

    def loss(y):
        mse = jnp.mean((y - y0) ** 2)
        return mse + lam * littlewood_paley_penalty(y, MASKS, s, p, q)

    for _ in range(steps):
        grads = jax.grad(loss)(y)
        updates, state = opt.update(grads, state)
        y = optax.apply_updates(y, updates)
    return y


# --- Littlewood-Paley machinery -----------------------------------------------

def test_band_energy_isolates_a_single_band():
    # A cosine whose frequency sits in band j0 deposits its energy in band j0.
    j0 = 4
    k = int(1.5 * 2 ** (j0 - 1))  # a bin strictly inside [2^{j0-1}, 2^{j0})
    t = jnp.arange(T)
    y = jnp.cos(2 * jnp.pi * k * t / T)[:, None]
    e = dyadic_band_energies(y, MASKS)
    assert int(jnp.argmax(e)) == j0 - 1  # bands are 1-indexed; array is 0-indexed
    assert e[j0 - 1] > 50 * jnp.max(jnp.delete(np.asarray(e), j0 - 1))


def test_band_energies_reconstruct_total_power():
    # Dyadic bands partition the spectrum, so summed band power ~ signal power
    # (Parseval), up to the DC/Nyquist bins the bands omit.
    y = _rough_signal()
    band_power = float(jnp.sum(dyadic_band_energies(y, MASKS) ** 2))
    total_power = float(jnp.sum((y - jnp.mean(y)) ** 2))
    assert band_power == pytest.approx(total_power, rel=0.05)


# --- penalty ranks regularity -------------------------------------------------

def test_penalty_ranks_smoother_signal_lower():
    # A smooth low-frequency signal has a smaller B^s norm than a rough one (s>0).
    t = jnp.arange(T)
    smooth = jnp.sin(2 * jnp.pi * 2 * t / T)[:, None]
    rough = _rough_signal()
    smooth = smooth / jnp.std(smooth)
    rough = rough / jnp.std(rough)

    def pen(y):
        return littlewood_paley_penalty(y, MASKS, s=1.0, p=2.0, q=2.0)

    assert float(pen(smooth)) < float(pen(rough))


def test_penalty_grows_with_target_s_for_rough_signal():
    # Larger s weights high bands more, so a rough signal's penalty increases in s.
    rough = _rough_signal()
    pens = [
        float(littlewood_paley_penalty(rough, MASKS, s, 2.0, 2.0))
        for s in (0.5, 1.0, 2.0)
    ]
    assert pens[0] < pens[1] < pens[2]


# --- the operational claim ----------------------------------------------------

def test_regularization_reduces_high_frequency_energy():
    y0 = _rough_signal()
    hf_before = _hf_fraction(y0)
    y_reg = _fit(y0, lam=0.1, s=1.0)
    hf_after = _hf_fraction(y_reg)
    assert hf_after < 0.5 * hf_before  # high-frequency content strongly suppressed


def test_regularization_increases_measured_smoothness():
    y0 = _rough_signal()
    assert _smoothness(_fit(y0, lam=0.1, s=1.0)) > _smoothness(y0) + 0.5


def test_target_s_controls_achieved_smoothness():
    # The headline: the achieved regularity tracks the *target* index s.
    y0 = _rough_signal()
    sm = [_smoothness(_fit(y0, lam=0.1, s=s)) for s in (0.5, 1.0, 2.0)]
    assert sm[0] < sm[1] < sm[2]
    # ... and the achieved smoothness is in the neighbourhood of the target.
    assert abs(sm[1] - 1.0) < 0.4


def test_unregularized_fit_recovers_the_target():
    # Sanity: with lambda=0 the optimum is the data itself (no smoothing).
    y0 = _rough_signal()
    y = _fit(y0, lam=0.0, s=1.0)
    assert jnp.allclose(y, y0, atol=1e-3)
