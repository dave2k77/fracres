"""Validate the Besov (Littlewood-Paley) regulariser moves trajectories toward a
target ``B^s_{p,q}`` regularity (knowledge base Section 4).

The regulariser ``( sum_j (2^{js} ||Delta_j Y||_p)^q )^{1/q}`` weights the dyadic
frequency band ``j`` by ``2^{js}``, so for ``s > 0`` it taxes high-frequency
content. Minimising ``MSE + lambda * B^s_{p,q}`` therefore smooths a trajectory --
and the *target* index ``s`` sets how smooth. This script shows:

1. the per-band energy spectrum collapsing at high frequency once regularised;
2. that the achieved regularity tracks the target ``s``;
3. the indices :func:`fracres.besov_indices` actually derives from a drive's Hurst
   exponent.

Run:  python examples/besov_regularization.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fracres import (
    besov_indices,
    dyadic_band_energies,
    generate_fbm_increments,
    littlewood_paley_penalty,
    make_dyadic_masks,
)

T = 256
MASKS = make_dyadic_masks(T)
J = MASKS.shape[0]


def smoothness(y):
    e = np.asarray(dyadic_band_energies(y, MASKS))
    return float(-np.polyfit(np.arange(1, J + 1), np.log2(e + 1e-20), 1)[0])


def fit(y0, lam, s, p=2.0, q=2.0, steps=400, lr=5e-2):
    y = y0
    opt = optax.adam(lr)
    state = opt.init(y)

    def loss(y):
        mse = jnp.mean((y - y0) ** 2)
        return mse + lam * littlewood_paley_penalty(y, MASKS, s, p, q)

    for _ in range(steps):
        g = jax.grad(loss)(y)
        upd, state = opt.update(g, state)
        y = optax.apply_updates(y, upd)
    return y


def main():
    y0 = generate_fbm_increments(T, H=0.5, key=jax.random.PRNGKey(0))[:, None]
    y0 = y0 / jnp.std(y0)  # rough, near-white trajectory

    print(f"Besov regularisation on a rough trajectory (T={T}, {J} dyadic bands)\n")

    # 1. Band-energy spectrum before / after.
    y_reg = fit(y0, lam=0.1, s=1.0)
    e0 = np.asarray(dyadic_band_energies(y0, MASKS))
    e1 = np.asarray(dyadic_band_energies(y_reg, MASKS))
    print("Per-band energy ||Delta_j Y|| (band j = octave; higher j = higher freq):")
    print("   j:    " + "  ".join(f"{j:>5}" for j in range(1, J + 1)))
    print("  raw:   " + "  ".join(f"{v:>5.2f}" for v in e0))
    print("  reg:   " + "  ".join(f"{v:>5.2f}" for v in e1))
    print("  (the regulariser collapses the high-frequency bands.)")

    # 2. Target s controls achieved regularity.
    print("\nAchieved regularity tracks the target smoothness index s (lambda=0.1):")
    print(f"  {'target s':>9} | {'achieved smoothness':>20}")
    print(f"  {'(raw)':>9} | {smoothness(y0):>20.3f}")
    for s in (0.5, 1.0, 1.5, 2.0):
        print(f"  {s:>9.1f} | {smoothness(fit(y0, 0.1, s)):>20.3f}")

    # 3. Indices derived from the drive's Hurst exponent (realistic usage).
    print("\nBesov indices derived from the drive (besov_indices, fGn alpha_S=2):")
    print(f"  {'H':>5} | {'s':>6} {'p':>6} {'q':>6}   bound on s")
    for H in (0.3, 0.5, 0.7):
        s, p, q = besov_indices(H)
        binding = "H" if H < 1.0 / p else "1/p"
        print(f"  {H:>5.1f} | {s:>6.3f} {p:>6.3f} {q:>6.3f}   ({binding})")
    print("\n(s < min{H, 1/p}, p < alpha_S -- the strict Besov bounds of KB v2 §4.2;")
    print(" for large H the embedding scale 1/p binds instead of H.)")


if __name__ == "__main__":
    main()
