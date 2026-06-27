"""Drive a fractional reservoir with fractional Gaussian noise and inspect states.

Run:  python examples/run_fbm_driven.py
"""
import jax
import jax.numpy as jnp

from fracres import L1CaputoKernel, PhantomBrain, generate_fbm_increments

TIME_STEPS = 2000
RES_SIZE = 500
OUT_FEATURES = 8
ALPHA = 0.75       # fractional-derivative order
H_NOISE = 0.65     # persistent (long-memory) driving noise


def main():
    key = jax.random.PRNGKey(2026)
    k_model, k_noise = jax.random.split(key)

    # Biologically relevant power-law memory kernel.
    kernel = L1CaputoKernel(alpha=ALPHA, history_length=200)

    model = PhantomBrain(
        in_features=1,
        res_size=RES_SIZE,
        out_features=OUT_FEATURES,
        fractional_operator=kernel,
        key=k_model,
        step_size=0.1,   # integration step h; activation scaled by h^alpha
        decay=1.0,       # node leak lambda (ensures Echo State Property)
    )

    # fBm increments as the stochastic drive, shape (TIME_STEPS, 1).
    drive = generate_fbm_increments(TIME_STEPS, H=H_NOISE, key=k_noise)[:, None]

    X_states, Y_hat = model.simulate(drive)
    print(f"drive:    {drive.shape}")
    print(f"states:   {X_states.shape}")
    print(f"observed: {Y_hat.shape}")
    print(f"Y_hat range: [{Y_hat.min():.3f}, {Y_hat.max():.3f}]")
    assert jnp.all(jnp.isfinite(Y_hat)), "non-finite outputs"


if __name__ == "__main__":
    main()
