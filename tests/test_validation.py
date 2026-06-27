"""Validate the fractional kernels against the analytic derivative of a power law.

Ground truth: ``D^alpha t^beta = Gamma(b+1)/Gamma(b+1-a) t^(b-a)`` (KB v2 §7.1).
Tolerances below are grounded in measured errors (float64); the convergence
tests pin the *order*, which is the strong check: GL is O(h), L1 is O(h^{2-a}).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from fracres import (  # noqa: E402
    GLKernel,
    L1CaputoKernel,
    analytic_power_law_derivative,
    convergence_order,
)

BETA = 2.0
EVAL_FRAC = 0.6  # interior evaluation point: t = EVAL_FRAC * T * h


def _rel_error(kernel_cls, alpha, T, h):
    t = jnp.arange(T) * h
    signal = t**BETA
    kernel = kernel_cls(alpha=alpha, history_length=T)
    est = kernel.apply(signal, h)
    k = int(EVAL_FRAC * T)
    true = analytic_power_law_derivative(t[k], alpha, BETA)
    return float(jnp.abs(est[k] - true) / jnp.abs(true))


@pytest.mark.parametrize("kernel_cls", [GLKernel, L1CaputoKernel])
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_power_law_accuracy(kernel_cls, alpha):
    # At h = 0.01 measured errors are ~4e-4 (GL) and <2e-4 (L1); 2e-3 is a safe cap.
    assert _rel_error(kernel_cls, alpha, T=2000, h=0.01) < 2e-3


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_gl_first_order_convergence(alpha):
    e_coarse = _rel_error(GLKernel, alpha, T=1000, h=0.02)
    e_fine = _rel_error(GLKernel, alpha, T=2000, h=0.01)
    assert convergence_order(e_coarse, e_fine) == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_l1_order_is_two_minus_alpha(alpha):
    e_coarse = _rel_error(L1CaputoKernel, alpha, T=1000, h=0.02)
    e_fine = _rel_error(L1CaputoKernel, alpha, T=2000, h=0.01)
    order = convergence_order(e_coarse, e_fine)
    # L1 converges at order 2 - alpha, strictly better than GL's first order.
    assert order > 1.0
    assert order == pytest.approx(2.0 - alpha, abs=0.25)
