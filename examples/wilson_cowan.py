"""Fractional Wilson-Cowan excitatory/inhibitory neural-mass reservoir (KB v2 §2.2).

Two populations per node -- excitatory ``E`` and inhibitory ``I`` -- with separate
time constants ``tau_E``, ``tau_I`` but a shared fractional order ``alpha_D``. This
script shows three things:

1. the bounded E/I dynamics under an fGn drive (the sigmoid firing rate + leak
   keep the state O(1));
2. how the inhibitory time constant ``tau_I`` modulates the reservoir's memory
   (a knob the single-population reservoir does not have);
3. that the reservoir reads out a delayed copy of its input (closed-form ridge),
   with the reservoir frozen throughout.

Run:  python examples/wilson_cowan.py
"""
import jax
import jax.numpy as jnp

from fracres import (
    GLKernel,
    WilsonCowanPhantomBrain,
    fit_readout_ridge,
    generate_fbm_increments,
)

T = 2500
N = 150
DELAY = 3
WASHOUT = 200
BETA = 1e-2
SPLIT = int(0.7 * T)
KEY = jax.random.PRNGKey(0)


def _delayed(u, d):
    return jnp.concatenate([jnp.zeros((d, 1)), u[:-d]], axis=0)


def _test_corr(model, u, target):
    Y = model(u)
    return float(jnp.corrcoef(Y[SPLIT:, 0], target[SPLIT:, 0])[0, 1])


def main():
    k_model, k_drive = jax.random.split(KEY)
    drive = generate_fbm_increments(T, H=0.7, key=k_drive)[:, None]
    target = _delayed(drive, DELAY)

    print("Fractional Wilson-Cowan E/I reservoir (alpha_D=0.8, N=150 per population)\n")

    # 1. Bounded E/I dynamics.
    model = WilsonCowanPhantomBrain(1, N, 1, GLKernel(0.8, 80), key=k_model)
    X, _ = model.simulate(drive)
    E, I = model.split(X)
    print("Population activity over the run (sigmoid firing rate keeps it in [0,1]):")
    print(f"  E: mean {E.mean():.3f}  std {E.std():.3f}  range [{E.min():.3f}, {E.max():.3f}]")
    print(f"  I: mean {I.mean():.3f}  std {I.std():.3f}  range [{I.min():.3f}, {I.max():.3f}]")
    print(f"  state finite: {bool(jnp.all(jnp.isfinite(X)))}  max|z|: {float(jnp.abs(X).max()):.3f}")

    # 2. Inhibitory time constant vs reservoir memory (a separate-tau knob).
    print(f"\nDelayed-copy recall (delay {DELAY}) vs inhibitory time constant tau_I:")
    print(f"{'tau_I':>6} | {'test corr':>10}")
    for tau_I in (1.0, 2.0, 4.0, 8.0):
        m = WilsonCowanPhantomBrain(1, N, 1, GLKernel(0.8, 80), key=k_model, tau_I=tau_I)
        m = fit_readout_ridge(m, drive[:SPLIT], target[:SPLIT], BETA, washout=WASHOUT)
        print(f"{tau_I:>6.1f} | {_test_corr(m, drive, target):>10.3f}")

    # 3. Memory-vs-delay profile at a fixed tau_I.
    print("\nFading memory (tau_E=1, tau_I=2), held-out recall vs delay:")
    print(f"{'delay':>6} | {'test corr':>10}")
    for d in (1, 2, 4, 8):
        m = WilsonCowanPhantomBrain(1, N, 1, GLKernel(0.8, 80), key=k_model)
        tgt = _delayed(drive, d)
        m = fit_readout_ridge(m, drive[:SPLIT], tgt[:SPLIT], BETA, washout=WASHOUT)
        print(f"{d:>6} | {_test_corr(m, drive, tgt):>10.3f}")
    print("\n(corr -> 1 = perfect recall; it decays with delay, the fading-memory property.)")


if __name__ == "__main__":
    main()
