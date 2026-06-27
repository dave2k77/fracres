import jax
import jax.numpy as jnp

from fracres import (
    GLKernel,
    L1CaputoKernel,
    PhantomBrain,
    generate_fbm_increments,
    qSOCPhantomBrain,
)

KEY = jax.random.PRNGKey(0)


def test_phantom_brain_shapes_and_finite():
    kernel = L1CaputoKernel(alpha=0.75, history_length=20, dt=0.01)
    model = PhantomBrain(1, 32, 4, kernel, key=KEY)
    drive = generate_fbm_increments(100, H=0.65, key=KEY)[:, None]
    X, Y = model.simulate(drive)
    assert X.shape == (100, 32)
    assert Y.shape == (100, 4)
    assert jnp.all(jnp.isfinite(Y))


def test_qsoc_returns_thresholds():
    kernel = GLKernel(alpha=0.8, history_length=20)
    model = qSOCPhantomBrain(1, 32, 4, kernel, key=KEY, tau_b=5.0)
    drive = generate_fbm_increments(100, H=0.7, key=KEY)[:, None]
    X, Y, B = model.simulate(drive, dt=0.01)
    assert Y.shape == (100, 4)
    assert B.shape == (100, 32)
    assert jnp.all(jnp.isfinite(B))


def test_fbm_increments_unit_variance():
    fgn = generate_fbm_increments(1000, H=0.6, key=KEY)
    assert fgn.shape == (1000,)
    assert jnp.isclose(jnp.std(fgn), 1.0, atol=1e-5)
