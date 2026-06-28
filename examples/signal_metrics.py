"""Characterise generated signals: long-range dependence and criticality.

Two questions the metrics module answers about a reservoir's output:

1. **Does the reservoir preserve / transform the long-range dependence of its
   drive?** We feed fractional Gaussian noise of known Hurst ``H`` and estimate
   ``H`` back (DFA and spectral slope) from both the drive and the reservoir's
   readout -- a fractional reservoir, carrying power-law memory, should produce a
   persistent (``H > 1/2``) output even from a near-white drive.
2. **Is the population activity critical?** We detect neuronal-avalanche-style
   events in the reservoir's activity and fit the size/duration power-law
   exponents (Beggs & Plenz: critical cortex sits near ``tau~1.5``, ``alpha~2.0``).

Run:  python examples/signal_metrics.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from fracres import (
    GLKernel,
    PhantomBrain,
    avalanche_exponents,
    generate_fbm_increments,
    signal_metrics,
)

T = 12000
KEY = jax.random.PRNGKey(0)


def main():
    print("Long-range dependence -- estimator recovery on the fGn drive (mean of 4 runs):")
    print(f"  {'(drive H)':<22} {'DFA':>9} {'beta':>10} {'2H-1':>8}")
    for H in (0.5, 0.6, 0.7, 0.8):
        runs = [signal_metrics(np.asarray(generate_fbm_increments(T, H=H, key=jax.random.PRNGKey(s))))
                for s in range(4)]
        dfa = np.mean([m.hurst_dfa for m in runs])
        beta = np.mean([m.spectral_beta for m in runs])
        print(f"  H={H:<20.1f} {dfa:>9.3f} {beta:>10.3f} {2 * H - 1:>8.2f}")
    print("  (both estimators degrade as H -> 1 at finite length -- DFA-1's known limit.)")

    # Does the reservoir impose long memory on a near-white drive? DFA is the
    # model-free LRD measure; H(spectral) assumes fGn, so it is only a cross-check
    # on the (fGn) drive -- not on the reservoir output, which is not fGn.
    print("\nReservoir transforms a near-white drive into a persistent output (DFA):")
    k_model, k_drive = jax.random.split(KEY)
    drive = np.asarray(generate_fbm_increments(T, H=0.5, key=k_drive))  # H=0.5, white
    model = PhantomBrain(1, 200, 1, GLKernel(alpha=0.8, history_length=100), key=k_model)
    X, Y = model.simulate(drive[:, None])
    print(f"  drive (H=0.5):     DFA H={signal_metrics(drive).hurst_dfa:.3f}")
    print(f"  reservoir output:  DFA H={signal_metrics(np.asarray(Y[:, 0])).hurst_dfa:.3f}"
          "   (lifted above 0.5 -> the fractional node added long memory)")

    # Criticality: avalanche statistics of the population activity.
    print("\nCriticality -- avalanche statistics of the population activity:")
    activity = np.asarray(jnp.mean(jnp.abs(X), axis=1))
    for q, label in ((0.5, "median"), (0.7, "70th pct")):
        thr = float(np.quantile(activity, q))
        ex = avalanche_exponents(activity, threshold=thr)
        print(f"  threshold={label:<9} n={ex.n_avalanches:<5} "
              f"size tau={ex.tau:.2f}   duration alpha={ex.alpha:.2f}")
    print("\n(Reference: critical cortex ~ tau 1.5, alpha 2.0; exact values depend on"
          " drive, threshold, and operating point.)")


if __name__ == "__main__":
    main()
