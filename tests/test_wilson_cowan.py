"""Fractional Wilson-Cowan E/I neural-mass reservoir (KB v2 §2.2)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from fracres import (
    GLKernel,
    WilsonCowanPhantomBrain,
    fit_readout_ridge,
    generate_fbm_increments,
    make_dyadic_masks,
    readout_filter_spec,
    train_step,
)

KEY = jax.random.PRNGKey(0)
N = 32
T = 600


def _model(alpha=0.8, tau_E=1.0, tau_I=2.0, key=KEY):
    return WilsonCowanPhantomBrain(
        1, N, 1, GLKernel(alpha, 40), key=key, tau_E=tau_E, tau_I=tau_I
    )


def _drive(seed=1, t=T):
    return generate_fbm_increments(t, H=0.7, key=jax.random.PRNGKey(seed))[:, None]


def test_stacked_state_shapes_and_split():
    model = _model()
    X, Y = model.simulate(_drive())
    assert X.shape == (T, 2 * N)  # stacked [E; I]
    assert Y.shape == (T, 1)
    E, I = model.split(X)
    assert E.shape == I.shape == (T, N)
    assert jnp.allclose(jnp.concatenate([E, I], axis=-1), X)


def test_alpha_comes_from_kernel():
    assert _model(alpha=0.6).reservoir.alpha == 0.6


def test_connectomes_are_nonnegative_signs_live_in_equations():
    res = _model().reservoir
    for W in (res.W_EE, res.W_EI, res.W_IE, res.W_II):
        assert jnp.all(W >= 0.0)


def test_dynamics_bounded_over_long_run():
    X, _ = _model().simulate(_drive(t=3000))
    assert jnp.all(jnp.isfinite(X))
    # Sigmoid drive + leak => state stays O(1); no divergence.
    assert float(jnp.max(jnp.abs(X))) < 5.0


def test_deterministic():
    X1, _ = _model().simulate(_drive())
    X2, _ = _model().simulate(_drive())
    assert jnp.array_equal(X1, X2)


def test_separate_tau_E_changes_excitatory_branch():
    # Same connectomes/key, different tau_E => different E trajectory.
    base = _model(tau_E=1.0)
    altered = _model(tau_E=0.5)
    E0, _ = base.split(base.simulate(_drive())[0])
    E1, _ = altered.split(altered.simulate(_drive())[0])
    assert not jnp.allclose(E0, E1)


def test_separate_tau_I_changes_inhibitory_branch():
    base = _model(tau_I=2.0)
    altered = _model(tau_I=4.0)
    _, I0 = base.split(base.simulate(_drive())[0])
    _, I1 = altered.split(altered.simulate(_drive())[0])
    assert not jnp.allclose(I0, I1)


def test_gradient_training_freezes_reservoir():
    model = _model()
    W_EE0, W_in0 = model.reservoir.W_EE, model.reservoir.W_in
    W_out0 = model.readout.W_out

    u = _drive()
    y = jnp.concatenate([jnp.zeros((2, 1)), u[:-2]], axis=0)  # delay-2 memory target
    masks = make_dyadic_masks(T)
    opt = optax.adam(5e-3)
    diff, _ = eqx.partition(model, readout_filter_spec(model))
    opt_state = opt.init(diff)
    for _ in range(5):
        model, opt_state, _ = train_step(model, opt, opt_state, u, y, masks, 0.7, 1e-4)

    assert jnp.allclose(model.reservoir.W_EE, W_EE0)  # connectome frozen
    assert jnp.allclose(model.reservoir.W_in, W_in0)
    assert not jnp.allclose(model.readout.W_out, W_out0)  # readout learned


def test_ridge_fit_reduces_error():
    model = _model()
    u = _drive()
    y = jnp.concatenate([jnp.zeros((2, 1)), u[:-2]], axis=0)
    mse0 = float(jnp.mean((model(u) - y) ** 2))
    fitted = fit_readout_ridge(model, u, y, beta=1e-2, washout=100)
    mse1 = float(jnp.mean((fitted(u)[100:] - y[100:]) ** 2))
    assert mse1 < mse0
    # Reservoir untouched by the closed-form fit.
    assert jnp.allclose(fitted.reservoir.W_EE, model.reservoir.W_EE)


def test_custom_firing_rate_is_used():
    # tanh firing rate => zero-centred drive, E/I can go negative (unlike sigmoid).
    model = _model()
    tanh_model = WilsonCowanPhantomBrain(
        1, N, 1, GLKernel(0.8, 40), key=KEY, firing_rate=jnp.tanh
    )
    X_sig, _ = model.simulate(_drive())
    X_tanh, _ = tanh_model.simulate(_drive())
    assert not jnp.allclose(X_sig, X_tanh)
    assert float(X_tanh.min()) < 0.0  # tanh allows negative activity
