import jax.numpy as jnp
import pytest

from fracres import GLKernel, L1CaputoKernel


@pytest.mark.parametrize("kernel_cls", [GLKernel, L1CaputoKernel])
def test_weights_length_matches_history(kernel_cls):
    L = 32
    kernel = kernel_cls(alpha=0.7, history_length=L)
    # Buffer passed in is x_history[:-1], i.e. L - 1 rows.
    assert kernel.weights.shape == (L - 1,)


def test_gl_recursion_first_coeffs():
    alpha = 0.5
    kernel = GLKernel(alpha=alpha, history_length=4)
    # c_0 = 1 (dropped); c_1 = -(alpha) ... using c_j = (1 - (1+alpha)/j) c_{j-1}
    c1 = (1.0 - (1.0 + alpha) / 1) * 1.0
    c2 = (1.0 - (1.0 + alpha) / 2) * c1
    c3 = (1.0 - (1.0 + alpha) / 3) * c2
    assert jnp.allclose(kernel.weights, jnp.array([c1, c2, c3]))


def test_kernel_call_contracts_history():
    L, res = 16, 5
    kernel = GLKernel(alpha=0.6, history_length=L)
    history = jnp.ones((L - 1, res))
    out = kernel(history)
    assert out.shape == (res,)
    assert jnp.allclose(out, kernel.weights.sum())
