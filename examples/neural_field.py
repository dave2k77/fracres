"""Spatial Amari neural-field reservoir (KB §2.2).

A cortical sheet discretised as a ring of ``N`` sites, with **distance-dependent**
Mexican-hat connectivity (short-range excitation, longer-range inhibition) instead
of random links. The fractional Amari field equation is

    tau^a D^a u = -u + w * S(u) + W_in u_ext

with the firing-rate non-linearity *inside* the spatial convolution. This script
shows:

1. the connectivity kernel -- symmetric, center-surround, near-zero DC gain;
2. spatial self-organisation: a spatially white drive makes the field develop
   power at the kernel's preferred wavelength (a peak at a non-zero wavenumber),
   the field analogue of pattern formation;
3. that the field still works as a reservoir (closed-form ridge on a delayed-copy
   memory task), with the spatial connectivity frozen.

Run:  python examples/neural_field.py
"""
import jax
import jax.numpy as jnp

from fracres import (
    GLKernel,
    NeuralFieldPhantomBrain,
    fit_readout_ridge,
    generate_fbm_increments,
)

N = 128
KEY = jax.random.PRNGKey(0)


def main():
    print(f"Fractional Amari neural-field reservoir (ring of N={N} sites, alpha_D=0.8)\n")

    model = NeuralFieldPhantomBrain(N, N, 1, GLKernel(0.8, 60), key=KEY)
    W = model.reservoir.W_res

    # 1. Connectivity kernel.
    row = W[0]
    print("Mexican-hat connectivity w(d) (row 0, by ring distance d):")
    print(f"  d=0:{float(row[0]):+.3f}  d=1:{float(row[1]):+.3f}  d=3:{float(row[3]):+.3f}"
          f"  d=8:{float(row[8]):+.3f}  d=20:{float(row[20]):+.3f}")
    print(f"  symmetric: {bool(jnp.allclose(W, W.T))}   "
          f"DC gain (mean row sum): {float(W.sum(1).mean()):+.2e}")

    # 2. Spatial self-organisation under a white-noise drive.
    u = jax.random.normal(jax.random.PRNGKey(3), (1500, N)) * 0.1
    X, _ = model.simulate(u)
    field_power = jnp.mean(jnp.abs(jnp.fft.rfft(X[400:], axis=1)) ** 2, axis=0)
    kernel_power = jnp.abs(jnp.fft.rfft(W[0])) ** 2
    k_field = int(jnp.argmax(field_power[1:]) + 1)
    k_kernel = int(jnp.argmax(kernel_power[1:]) + 1)
    print("\nSpatial self-organisation from a white-noise drive:")
    print(f"  kernel preferred wavenumber : k={k_kernel}  (wavelength ~{N / k_kernel:.0f} sites)")
    print(f"  emergent field power peak    : k={k_field}  (wavelength ~{N / k_field:.0f} sites)")
    print(f"  peak / mean band power       : {float(field_power[k_field] / jnp.mean(field_power[1:])):.2f}x")
    print("  (the field selects a wavelength near the kernel band -- structure from")
    print("   connectivity, not from the spatially white input.)")

    # 3. Reservoir-computing usage: delayed-copy recall (connectivity frozen).
    T = 2500
    SPLIT = int(0.7 * T)
    rc = NeuralFieldPhantomBrain(1, N, 1, GLKernel(0.8, 60), key=KEY)
    drive = generate_fbm_increments(T, H=0.7, key=jax.random.PRNGKey(7))[:, None]
    print("\nFading memory (delayed-copy recall), spatial connectivity frozen:")
    print(f"{'delay':>6} | {'test corr':>10}")
    for d in (1, 2, 4, 8):
        target = jnp.concatenate([jnp.zeros((d, 1)), drive[:-d]], axis=0)
        fitted = fit_readout_ridge(rc, drive[:SPLIT], target[:SPLIT], beta=1e-2, washout=200)
        Y = fitted(drive)
        corr = float(jnp.corrcoef(Y[SPLIT:, 0], target[SPLIT:, 0])[0, 1])
        print(f"{d:>6} | {corr:>10.3f}")
    print("\n(corr -> 1 = perfect recall; decays with delay, the fading-memory property.)")


if __name__ == "__main__":
    main()
