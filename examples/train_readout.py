"""End-to-end gradient training of the readout (KB v2 §6.2 / §4).

The companion ``fit_ridge_readout.py`` solves the readout in closed form; this
script instead drives the *full* optimisation pipeline -- ``train_step`` with an
``optax`` optimiser and the Besov (Littlewood-Paley) regulariser -- and verifies
the central design invariant: **only the readout learns.** The reservoir
connectome ``W_res``, the input map ``W_in``, and the fractional-kernel weights
are partitioned out (``readout_filter_spec``) and must come back bit-for-bit
identical after training.

Task: reconstruct a delayed copy ``u_{t-delay}`` of an fGn drive (a fading-memory
probe). We train on one noise realisation and report held-out error on an
independent one.

Run:  python examples/train_readout.py
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from fracres import (
    GLKernel,
    PhantomBrain,
    generate_fbm_increments,
    make_dyadic_masks,
    readout_filter_spec,
    train_step,
)

T = 2000
RES_SIZE = 200
DELAY = 2
H_HURST = 0.7
LAMBDA_REG = 1e-4  # Besov penalty weight
LR = 5e-3
EPOCHS = 1000


def _delayed_target(drive, delay):
    return jnp.concatenate([jnp.zeros((delay, 1)), drive[:-delay]], axis=0)


def main():
    key = jax.random.PRNGKey(0)
    k_model, k_train, k_test = jax.random.split(key, 3)
    kernel = GLKernel(alpha=0.8, history_length=100)
    model = PhantomBrain(1, RES_SIZE, 1, kernel, key=k_model)

    u_train = generate_fbm_increments(T, H=H_HURST, key=k_train)[:, None]
    u_test = generate_fbm_increments(T, H=H_HURST, key=k_test)[:, None]
    y_train = _delayed_target(u_train, DELAY)
    y_test = _delayed_target(u_test, DELAY)

    masks = make_dyadic_masks(T)

    # Optimise ONLY the trainable partition (readout.W_out).
    optimizer = optax.adam(LR)
    diff, _ = eqx.partition(model, readout_filter_spec(model))
    opt_state = optimizer.init(diff)

    # Snapshot the frozen weights so we can prove they never move.
    W_res0 = model.reservoir.W_res
    W_in0 = model.reservoir.W_in
    kernel0 = model.reservoir.fractional_operator.weights
    W_out0 = model.readout.W_out

    def test_mse(m):
        return float(jnp.mean((m(u_test) - y_test) ** 2))

    print(f"{'epoch':>6} | {'train loss':>11} {'test MSE':>10}")
    print(f"{'init':>6} | {'-':>11} {test_mse(model):>10.4f}")
    for epoch in range(1, EPOCHS + 1):
        model, opt_state, loss = train_step(
            model, optimizer, opt_state, u_train, y_train, masks, H_HURST, LAMBDA_REG
        )
        if epoch % 200 == 0 or epoch == 1:
            print(f"{epoch:>6} | {float(loss):>11.5f} {test_mse(model):>10.4f}")

    # The design invariant: reservoir/kernel frozen, readout changed.
    print("\nFrozen-weight check (readout-only training, KB v2 §6.2):")
    print(f"  W_res unchanged : {bool(jnp.allclose(model.reservoir.W_res, W_res0))}")
    print(f"  W_in  unchanged : {bool(jnp.allclose(model.reservoir.W_in, W_in0))}")
    kernel_now = model.reservoir.fractional_operator.weights
    print(f"  kernel unchanged: {bool(jnp.allclose(kernel_now, kernel0))}")
    print(f"  W_out  changed  : {bool(not jnp.allclose(model.readout.W_out, W_out0))}")

    test_corr = float(jnp.corrcoef(model(u_test)[:, 0], y_test[:, 0])[0, 1])
    print(f"\nHeld-out recall at delay {DELAY}: corr = {test_corr:.3f}")


if __name__ == "__main__":
    main()
