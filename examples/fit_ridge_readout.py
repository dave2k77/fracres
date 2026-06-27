"""Fit the readout in closed form on a reservoir-memory task.

Drives a frozen fractional reservoir with fGn, then fits the linear readout
(closed-form ridge, KB v2 §6.3) to reconstruct a delayed copy of the input
``u_{t-delay}`` -- a standard probe of the reservoir's fading memory. Reports
in-sample vs held-out error to show the readout generalises.

Run:  python examples/fit_ridge_readout.py
"""
import jax
import jax.numpy as jnp

from fracres import GLKernel, PhantomBrain, fit_ridge_readout, generate_fbm_increments

T = 3000
RES_SIZE = 300
WASHOUT = 200
BETA = 1e-2
SPLIT = int(0.7 * T)


def main():
    key = jax.random.PRNGKey(0)
    k_model, k_noise = jax.random.split(key)
    model = PhantomBrain(1, RES_SIZE, 1, GLKernel(alpha=0.8, history_length=100), key=k_model)

    drive = generate_fbm_increments(T, H=0.7, key=k_noise)[:, None]
    X_states, _ = model.simulate(drive)  # frozen reservoir states (T, N)

    print(f"{'delay':>5} | {'train MSE':>10} {'test MSE':>10} {'test corr':>10}")
    for delay in (1, 3, 5, 10):
        target = jnp.concatenate([jnp.zeros((delay, 1)), drive[:-delay]], axis=0)

        # Fit on the training segment only; evaluate on the held-out tail.
        W_out = fit_ridge_readout(X_states[:SPLIT], target[:SPLIT], BETA, washout=WASHOUT)
        Y_hat = X_states @ W_out.T

        train_mse = jnp.mean((Y_hat[WASHOUT:SPLIT] - target[WASHOUT:SPLIT]) ** 2)
        test_mse = jnp.mean((Y_hat[SPLIT:] - target[SPLIT:]) ** 2)
        test_corr = jnp.corrcoef(Y_hat[SPLIT:, 0], target[SPLIT:, 0])[0, 1]
        print(f"{delay:>5} | {float(train_mse):>10.4f} {float(test_mse):>10.4f} "
              f"{float(test_corr):>10.3f}")
    print("\n(corr -> 1 = perfect recall; it should decay as the delay grows.)")


if __name__ == "__main__":
    main()
