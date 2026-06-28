"""Closed-form ridge readout (KB v2 §6.3)."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from fracres import (  # noqa: E402
    GLKernel,
    PhantomBrain,
    fit_readout_ridge,
    fit_ridge_readout,
    generate_fbm_increments,
)


def test_ridge_recovers_known_linear_map():
    key = jax.random.PRNGKey(0)
    T, N, out = 500, 30, 3
    X = jax.random.normal(key, (T, N))
    W_true = jax.random.normal(jax.random.PRNGKey(1), (out, N))
    Y = X @ W_true.T
    W_est = fit_ridge_readout(X, Y, beta=1e-8)
    assert W_est.shape == (out, N)
    assert jnp.allclose(W_est, W_true, atol=1e-4)
    assert jnp.mean((X @ W_est.T - Y) ** 2) < 1e-10


def test_washout_is_applied():
    # With T-washout < N the (unregularised) fit is underdetermined; just check
    # the rows used drop the transient (different W from no-washout on same data).
    key = jax.random.PRNGKey(3)
    X = jax.random.normal(key, (200, 10))
    Y = jax.random.normal(jax.random.PRNGKey(4), (200, 2))
    assert not jnp.allclose(
        fit_ridge_readout(X, Y, beta=1e-3, washout=0),
        fit_ridge_readout(X, Y, beta=1e-3, washout=50),
    )


def test_fit_readout_ridge_improves_and_freezes():
    key = jax.random.PRNGKey(0)
    model = PhantomBrain(1, 64, 1, GLKernel(0.8, 50), key=key)
    T, delay, washout = 600, 3, 100
    drive = generate_fbm_increments(T, H=0.7, key=jax.random.PRNGKey(2))[:, None]
    # u_{t-delay}
    target = jnp.concatenate([jnp.zeros((delay, 1)), drive[:-delay]], axis=0)

    W_res0 = model.reservoir.W_res
    mse_before = jnp.mean((model(drive)[washout:] - target[washout:]) ** 2)
    fitted = fit_readout_ridge(model, drive, target, beta=1e-2, washout=washout)
    mse_after = jnp.mean((fitted(drive)[washout:] - target[washout:]) ** 2)

    assert mse_after < mse_before  # closed-form readout beats the random one
    assert jnp.allclose(fitted.reservoir.W_res, W_res0)  # reservoir stays frozen
    assert jnp.all(jnp.isfinite(fitted(drive)))
