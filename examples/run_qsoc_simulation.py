"""Simulate a qSOC phantom brain and observe homeostatic threshold adaptation.

Run:  python examples/run_qsoc_simulation.py
"""
import jax
import jax.numpy as jnp

from fracres import GLKernel, generate_fbm_increments, qSOCPhantomBrain

TIME_STEPS = 1500
RES_SIZE = 400
OUT_FEATURES = 8
ALPHA = 0.8
H_NOISE = 0.7


def main():
    key = jax.random.PRNGKey(7)
    k_model, k_noise = jax.random.split(key)

    kernel = GLKernel(alpha=ALPHA, history_length=200)
    model = qSOCPhantomBrain(
        in_features=1,
        res_size=RES_SIZE,
        out_features=OUT_FEATURES,
        fractional_operator=kernel,
        key=k_model,
        E_crit=1.0,
        tau_b=5.0,
        gamma=1.0,
    )

    drive = generate_fbm_increments(TIME_STEPS, H=H_NOISE, key=k_noise)[:, None]
    X_states, Y_hat, B_thresholds = model.simulate(drive, dt=0.01)

    print(f"states:     {X_states.shape}")
    print(f"observed:   {Y_hat.shape}")
    print(f"thresholds: {B_thresholds.shape}")
    print(f"mean |b| early -> late: {jnp.abs(B_thresholds[:50]).mean():.4f} "
          f"-> {jnp.abs(B_thresholds[-50:]).mean():.4f}")
    assert jnp.all(jnp.isfinite(Y_hat)), "non-finite outputs"


if __name__ == "__main__":
    main()
