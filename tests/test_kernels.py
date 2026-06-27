import jax.numpy as jnp
import pytest

from fracres import GLKernel, L1CaputoKernel


@pytest.mark.parametrize("kernel_cls", [GLKernel, L1CaputoKernel])
def test_weights_length_matches_strict_past(kernel_cls):
    L = 32
    kernel = kernel_cls(alpha=0.7, history_length=L)
    # Strict-past coefficients c_2..c_L act on x_history[1:], i.e. L - 1 rows.
    assert kernel.weights.shape == (L - 1,)


def test_gl_recursion_strict_past_coeffs():
    alpha = 0.5
    kernel = GLKernel(alpha=alpha, history_length=4)
    # Kernel stores c_2..c_4 (c_0, c_1 are handled by the reservoir update).
    c1 = (1.0 - (1.0 + alpha) / 1) * 1.0
    c2 = (1.0 - (1.0 + alpha) / 2) * c1
    c3 = (1.0 - (1.0 + alpha) / 3) * c2
    c4 = (1.0 - (1.0 + alpha) / 4) * c3
    assert jnp.allclose(kernel.weights, jnp.array([c2, c3, c4]))


def test_kernel_call_uses_strict_past():
    L, res = 16, 5
    kernel = GLKernel(alpha=0.6, history_length=L)
    history = jnp.ones((L, res))  # full buffer; row 0 = x_{k-1} is excluded
    out = kernel(history)
    assert out.shape == (res,)
    assert jnp.allclose(out, kernel.weights.sum())
