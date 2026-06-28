"""Matignon-wedge stability diagnostics and edge-of-chaos control (KB v2 §3.4).

For the fractional reservoir, stability is governed by Matignon's theorem:
``|arg(lambda_i(A))| > alpha*pi/2`` with ``A = W_res - Lambda``. The unstable cone
*shrinks* as alpha drops below 1, so a fractional reservoir tolerates a "hotter"
connectome than a classical ESN. This script shows the threshold, the diagnostics,
the edge-of-chaos control, and the key fractional advantage.

Run:  python examples/matignon_control.py
"""
import jax
import jax.numpy as jnp

from fracres import (
    FractionalReservoir,
    GLKernel,
    matignon_diagnostics,
    set_edge_of_chaos,
    set_spectral_radius,
    system_matrix,
)

KEY = jax.random.PRNGKey(0)


def main():
    print("Matignon thresholds (unstable cone half-angle = alpha * 90 deg):")
    for alpha in (0.5, 0.7, 0.9, 1.0):
        print(f"  alpha={alpha}: {alpha * 90:.0f} deg")

    # Start from an over-driven (unstable) connectome and pull it to the edge.
    hot = set_spectral_radius(FractionalReservoir(1, 200, GLKernel(0.7, 30), KEY), 4.0)
    print("\nEdge-of-chaos control (alpha=0.7), from an unstable connectome:")
    print(f"{'safety':>7} | {'rho(W)':>7} {'min_arg':>9} {'margin':>8} {'stable':>7}")
    for sf in (None, 0.8, 0.95, 1.0, 1.05):
        res = hot if sf is None else set_edge_of_chaos(hot, sf)
        d = matignon_diagnostics(res)
        tag = "start" if sf is None else f"{sf:.2f}"
        print(f"{tag:>7} | {d.spectral_radius:>7.2f} "
              f"{d.min_arg * 180 / 3.14159:>7.1f}d "
              f"{d.margin:>+8.3f} {str(d.stable):>7}")

    # The fractional advantage: one fixed A whose least-stable mode sits ~75 deg
    # (between the 0.7 and 0.9 thresholds), so stability flips with alpha alone.
    target = 75.0 * 3.14159 / 180.0
    best = None
    for sf in jnp.linspace(0.9, 1.0, 41):
        A = system_matrix(set_edge_of_chaos(hot, float(sf)))
        min_arg = float(jnp.min(jnp.abs(jnp.angle(jnp.linalg.eigvals(A)))))
        if best is None or abs(min_arg - target) < abs(best - target):
            best = min_arg
    crit = 2.0 * best / 3.14159
    print(f"\nSame matrix A (min|arg|={best * 180 / 3.14159:.1f} deg), "
          f"Matignon-stable for alpha < {crit:.2f}:")
    print(f"{'alpha':>6} {'threshold':>10} {'stable':>7}")
    for alpha in (0.5, 0.7, 0.9, 1.0):
        print(f"{alpha:>6} {alpha * 90:>8.0f}d {str(best > alpha * 3.14159 / 2):>7}")
    print("\n(A classical ESN is the alpha=1 row; the same matrix is stable for the"
          " fractional node.)")


if __name__ == "__main__":
    main()
