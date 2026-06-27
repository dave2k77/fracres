"""Validate GLKernel / L1CaputoKernel against the analytic D^alpha t^beta.

Reconstructs each kernel's full operator (via ``kernel.apply``) and compares it
to the closed form on a refined sequence of grids, reporting relative error and
the empirical convergence order. Expected: GL ~ O(h), L1 ~ O(h^{2-alpha}).

Run:  python examples/validate_kernels.py
"""
import jax

jax.config.update("jax_enable_x64", True)  # GL coefficients need float64 at large L

import jax.numpy as jnp

from jax.scipy.special import gamma

from fracres import (
    GLKernel,
    L1CaputoKernel,
    analytic_power_law_derivative,
    convergence_order,
    mittag_leffler,
)

BETA = 2.0
GRIDS = [(1000, 0.02), (2000, 0.01), (4000, 0.005)]
EVAL_FRAC = 0.6
LAM = -1.0  # fractional relaxation for the Mittag-Leffler test


def rel_error(kernel_cls, alpha, T, h):
    t = jnp.arange(T) * h
    est = kernel_cls(alpha=alpha, history_length=T).apply(t**BETA, h)
    k = int(EVAL_FRAC * T)
    true = analytic_power_law_derivative(t[k], alpha, BETA)
    return float(jnp.abs(est[k] - true) / jnp.abs(true))


def power_law_table():
    print(f"\n### Power law  x(t) = t^{BETA}  (x(0)=0) ###")
    for name, cls in [("GL  (expect O(h))", GLKernel), ("L1  (expect O(h^{2-a}))", L1CaputoKernel)]:
        print(f"\n=== {name} ===")
        print(f"{'alpha':>6} | {'relerr h=.02':>12} {'h=.01':>10} {'h=.005':>10} | "
              f"{'order':>6} {'2-a':>5}")
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            errs = [rel_error(cls, alpha, T, h) for T, h in GRIDS]
            order = convergence_order(errs[0], errs[-1], refinement=4.0)  # h: .02 -> .005
            print(f"{alpha:>6.1f} | {errs[0]:>12.2e} {errs[1]:>10.2e} {errs[2]:>10.2e} | "
                  f"{order:>6.2f} {2.0 - alpha:>5.2f}")


def mittag_leffler_table():
    # x(t) = E_a(lam t^a) solves D^a_C x = lam x, x(0)=1: the Caputo eigenfunction.
    # GL.apply(x) -> Riemann-Liouville = lam x + t^-a/Gamma(1-a); subtracting x(0)
    # recovers Caputo (lam x) for both kernels.
    T, h = 4000, 0.005
    t = jnp.arange(T) * h
    k = int(EVAL_FRAC * T)
    print(f"\n### Mittag-Leffler  x(t) = E_a({LAM} t^a)  (x(0)=1) @ t={float(t[k]):.1f} ###")
    print(f"{'alpha':>6} | {'GL->RL relerr':>13} | {'GL->Caputo':>11} {'L1->Caputo':>11}  "
          f"(via x - x(0))")
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        sig = mittag_leffler(LAM * t**alpha, alpha)
        caputo_true = LAM * float(sig[k])
        rl_true = caputo_true + float(t[k]) ** (-alpha) / gamma(1.0 - alpha)
        gl_rl = float(GLKernel(alpha, T).apply(sig, h)[k])
        gl_cap = float(GLKernel(alpha, T).apply(sig - sig[0], h)[k])
        l1_cap = float(L1CaputoKernel(alpha, T).apply(sig - sig[0], h)[k])
        print(f"{alpha:>6.1f} | {abs(gl_rl - rl_true) / abs(rl_true):>13.2e} | "
              f"{abs(gl_cap - caputo_true) / abs(caputo_true):>11.2e} "
              f"{abs(l1_cap - caputo_true) / abs(caputo_true):>11.2e}")


def main():
    power_law_table()
    mittag_leffler_table()


if __name__ == "__main__":
    main()
