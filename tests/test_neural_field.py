"""Spatial Amari neural-field reservoir (KB §2.2)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from fracres import (
    GLKernel,
    NeuralFieldPhantomBrain,
    fit_readout_ridge,
    generate_fbm_increments,
    make_dyadic_masks,
    mexican_hat_kernel,
    readout_filter_spec,
    ring_distance,
    train_step,
)

KEY = jax.random.PRNGKey(0)
N = 128
T = 600


def _model(key=KEY, in_features=1, **kw):
    return NeuralFieldPhantomBrain(in_features, N, 1, GLKernel(0.8, 40), key=key, **kw)


def _drive(seed=1, t=T):
    return generate_fbm_increments(t, H=0.7, key=jax.random.PRNGKey(seed))[:, None]


# --- geometry / connectivity --------------------------------------------------

def test_ring_distance_is_periodic_metric():
    d = ring_distance(8)
    assert jnp.array_equal(jnp.diag(d), jnp.zeros(8))
    assert jnp.allclose(d, d.T)
    assert float(d[0, 1]) == 1.0
    assert float(d[0, 7]) == 1.0  # wraps around the ring
    assert float(d.max()) == 4.0  # N // 2


def test_mexican_hat_is_symmetric_circulant_center_surround():
    W = mexican_hat_kernel(N, sigma_e=2.0, sigma_i=4.0, A_e=1.0, A_i=0.5)
    assert jnp.allclose(W, W.T)  # symmetric
    # circulant: row i is row 0 rolled by i.
    assert jnp.allclose(W[5], jnp.roll(W[0], 5), atol=1e-6)
    assert float(W[0, 0]) > 0.0  # excitatory centre
    assert float(W[0, 3]) < 0.0  # inhibitory surround


def test_default_amplitudes_give_near_zero_dc_gain():
    W = mexican_hat_kernel(N, 2.0, 4.0, 1.0, 0.5)  # A_e sigma_e == A_i sigma_i
    assert abs(float(W.sum(axis=1).mean())) < 1e-3


def test_connectivity_is_deterministic_not_random():
    a = _model(key=jax.random.PRNGKey(1)).reservoir.W_res
    b = _model(key=jax.random.PRNGKey(99)).reservoir.W_res
    assert jnp.allclose(a, b)  # independent of PRNG key


# --- dynamics -----------------------------------------------------------------

def test_simulate_shapes():
    X, Y = _model().simulate(_drive())
    assert X.shape == (T, N)
    assert Y.shape == (T, 1)


def test_dynamics_bounded_over_long_run():
    X, _ = _model().simulate(_drive(t=3000))
    assert jnp.all(jnp.isfinite(X))
    assert float(jnp.max(jnp.abs(X))) < 20.0


def test_field_self_organises_at_kernel_preferred_wavelength():
    # White-noise spatial drive; the field's spatial power spectrum should peak at
    # the connectivity kernel's own preferred (non-zero) wavenumber.
    model = _model(in_features=N)
    u = jax.random.normal(jax.random.PRNGKey(3), (1200, N)) * 0.1
    X, _ = model.simulate(u)
    field_power = jnp.mean(jnp.abs(jnp.fft.rfft(X[400:], axis=1)) ** 2, axis=0)
    kernel_power = jnp.abs(jnp.fft.rfft(model.reservoir.W_res[0])) ** 2

    k_field = int(jnp.argmax(field_power[1:]) + 1)  # skip DC
    k_kernel = int(jnp.argmax(kernel_power[1:]) + 1)
    assert 1 < k_field < N // 2  # a genuine interior wavelength, not DC/Nyquist
    assert abs(k_field - k_kernel) <= 2  # tracks the kernel's preferred band
    # The selected band stands out above the rest of the spectrum.
    assert field_power[k_field] > 1.3 * jnp.mean(field_power[1:])


def test_deterministic():
    X1, _ = _model().simulate(_drive())
    X2, _ = _model().simulate(_drive())
    assert jnp.array_equal(X1, X2)


def test_tau_changes_dynamics():
    a, _ = _model(tau=1.0).simulate(_drive())
    b, _ = _model(tau=3.0).simulate(_drive())
    assert not jnp.allclose(a, b)


# --- reservoir-computing usage ------------------------------------------------

def test_gradient_training_freezes_field_connectivity():
    model = _model()
    W_res0, W_in0 = model.reservoir.W_res, model.reservoir.W_in
    W_out0 = model.readout.W_out

    u = _drive()
    y = jnp.concatenate([jnp.zeros((2, 1)), u[:-2]], axis=0)
    masks = make_dyadic_masks(T)
    opt = optax.adam(5e-3)
    diff, _ = eqx.partition(model, readout_filter_spec(model))
    opt_state = opt.init(diff)
    for _ in range(5):
        model, opt_state, _ = train_step(model, opt, opt_state, u, y, masks, 0.7, 1e-4)

    assert jnp.allclose(model.reservoir.W_res, W_res0)  # spatial kernel frozen
    assert jnp.allclose(model.reservoir.W_in, W_in0)
    assert not jnp.allclose(model.readout.W_out, W_out0)


def test_ridge_fit_reduces_error_and_leaves_reservoir():
    model = _model()
    u = _drive()
    y = jnp.concatenate([jnp.zeros((2, 1)), u[:-2]], axis=0)
    mse0 = float(jnp.mean((model(u) - y) ** 2))
    fitted = fit_readout_ridge(model, u, y, beta=1e-2, washout=100)
    mse1 = float(jnp.mean((fitted(u)[100:] - y[100:]) ** 2))
    assert mse1 < mse0
    assert jnp.allclose(fitted.reservoir.W_res, model.reservoir.W_res)
