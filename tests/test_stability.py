"""Matignon-wedge stability diagnostics and spectral control (KB v2 §3.4)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from fracres import (
    FractionalReservoir,
    GLKernel,
    PhantomBrain,
    matignon_diagnostics,
    matignon_edge_scale,
    set_edge_of_chaos,
    set_spectral_radius,
    system_matrix,
)

KEY = jax.random.PRNGKey(0)


def _reservoir(alpha=0.7, n=64):
    return FractionalReservoir(1, n, GLKernel(alpha, 30), KEY)


@pytest.mark.parametrize("alpha,deg", [(0.5, 45.0), (0.7, 63.0), (0.9, 81.0)])
def test_threshold_is_alpha_times_ninety_degrees(alpha, deg):
    d = matignon_diagnostics(_reservoir(alpha))
    assert d.threshold_deg == pytest.approx(deg, abs=1e-6)


def test_system_matrix_is_wres_minus_lambda():
    res = _reservoir()
    n = res.W_res.shape[0]
    assert jnp.allclose(system_matrix(res), res.W_res - res.decay * jnp.eye(n))


def test_set_spectral_radius():
    res = set_spectral_radius(_reservoir(), 0.8)
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(res.W_res)))
    assert float(rho) == pytest.approx(0.8, abs=1e-3)


def test_edge_of_chaos_brackets_stability():
    hot = set_spectral_radius(_reservoir(), 5.0)  # blow the connectome up
    assert not matignon_diagnostics(hot).stable  # unstable to start
    assert matignon_diagnostics(set_edge_of_chaos(hot, 0.9)).stable  # inside the wedge
    assert not matignon_diagnostics(set_edge_of_chaos(hot, 1.1)).stable  # past the edge


def test_edge_margin_decreases_with_safety_factor():
    hot = set_spectral_radius(_reservoir(), 5.0)
    margins = [
        matignon_diagnostics(set_edge_of_chaos(hot, sf)).margin
        for sf in (0.8, 0.9, 1.0)
    ]
    assert margins[0] > margins[1] > margins[2]  # closer to the edge => smaller margin


def test_critical_alpha_consistency():
    # stable  <=>  alpha < critical_alpha  (both derived from the same min_arg).
    hot = set_spectral_radius(_reservoir(), 5.0)
    d = matignon_diagnostics(set_edge_of_chaos(hot, 0.9))
    assert d.stable == (d.alpha < d.critical_alpha)


def test_control_on_model_freezes_readout():
    model = PhantomBrain(1, 64, 3, GLKernel(0.7, 30), key=KEY)
    W_out0 = model.readout.W_out
    tuned = set_edge_of_chaos(model, 0.9)
    # reservoir rescaled
    assert not jnp.allclose(tuned.reservoir.W_res, model.reservoir.W_res)
    assert jnp.allclose(tuned.readout.W_out, W_out0)  # readout untouched
    assert matignon_diagnostics(tuned).stable


def test_unconditionally_stable_connectome_has_infinite_edge():
    # Skew-symmetric W_res has pure-imaginary eigenvalues (arg = +-90deg); with a
    # positive leak they sit at args in (90, 180) deg, always outside the 63deg
    # cone, so no scaling can reach the edge.
    res = _reservoir(0.7)
    m = jax.random.normal(jax.random.PRNGKey(5), res.W_res.shape)
    res = eqx.tree_at(lambda r: r.W_res, res, m - m.T)
    assert not jnp.isfinite(matignon_edge_scale(res))
    with pytest.raises(ValueError):
        set_edge_of_chaos(res, 0.9)
