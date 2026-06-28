"""Property-based tests (Hypothesis) for fractional-kernel invariants.

These complement the deterministic kernel checks in ``test_validation.py``: rather
than verifying numerical accuracy at fixed parameters, they assert *algebraic*
invariants that must hold for **every** valid ``(alpha, history_length)`` -- the
operator contract a kernel must satisfy to plug into a reservoir. Numerical
convergence of ``apply`` (which is order- and history-length-dependent) stays in
the deterministic suite; here we exercise linearity, the strict-past structure,
and the closed forms of the GL / L1 coefficients.
"""
import jax

jax.config.update("jax_enable_x64", True)  # coefficient recursions need float64

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jax.scipy.special import gamma

from fracres import GLKernel, L1CaputoKernel

settings.register_profile("kernels", max_examples=40, deadline=None)
settings.load_profile("kernels")

alphas = st.floats(0.05, 0.95, allow_nan=False, allow_infinity=False)
lengths = st.integers(2, 64)
res_sizes = st.integers(1, 4)
kernels = st.sampled_from([GLKernel, L1CaputoKernel])


def _finite(lo, hi):
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False, width=64)


def _history(data, L, res):
    return data.draw(arrays(np.float64, (L, res), elements=_finite(-10.0, 10.0)))


# --- abstract operator contract (both kernels) --------------------------------

@given(cls=kernels, alpha=alphas, L=lengths)
def test_weights_have_length_L_minus_one(cls, alpha, L):
    assert cls(alpha, L).weights.shape == (L - 1,)


@given(cls=kernels, alpha=alphas, L=lengths)
def test_all_quantities_are_finite(cls, alpha, L):
    k = cls(alpha, L)
    assert np.isfinite(k.leading)
    assert np.isfinite(k.forcing_factor)
    assert np.all(np.isfinite(np.asarray(k.weights)))


@given(cls=kernels, alpha=alphas, L=lengths)
def test_construction_is_deterministic(cls, alpha, L):
    a, b = cls(alpha, L), cls(alpha, L)
    assert a.leading == b.leading
    assert a.forcing_factor == b.forcing_factor
    assert np.array_equal(np.asarray(a.weights), np.asarray(b.weights))


@given(data=st.data(), cls=kernels, alpha=alphas, L=lengths, res=res_sizes)
def test_call_returns_strict_past_weighted_sum(data, cls, alpha, L, res):
    k = cls(alpha, L)
    H = _history(data, L, res)
    got = np.asarray(k(H))
    # Independent reimplementation: sum_j w_j * x_{k-1-j} over the strict past.
    want = np.tensordot(np.asarray(k.weights), H[1:], axes=([0], [0]))
    assert got.shape == (res,)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


@given(data=st.data(), cls=kernels, alpha=alphas, L=lengths, res=res_sizes)
def test_call_ignores_the_current_state(data, cls, alpha, L, res):
    # Row 0 is x_{k-1} (the "leading" state); the memory term uses only the
    # strict past, so overwriting row 0 must not change the output.
    k = cls(alpha, L)
    H = _history(data, L, res)
    H2 = H.copy()
    H2[0] = data.draw(arrays(np.float64, (res,), elements=_finite(-10.0, 10.0)))
    assert np.allclose(np.asarray(k(H)), np.asarray(k(H2)), rtol=1e-12, atol=1e-12)


@given(
    data=st.data(), cls=kernels, alpha=alphas, L=lengths, res=res_sizes,
    c1=_finite(-5.0, 5.0), c2=_finite(-5.0, 5.0),
)
def test_call_is_linear(data, cls, alpha, L, res, c1, c2):
    k = cls(alpha, L)
    X = _history(data, L, res)
    Y = _history(data, L, res)
    combined = np.asarray(k(c1 * X + c2 * Y))
    separate = c1 * np.asarray(k(X)) + c2 * np.asarray(k(Y))
    assert np.allclose(combined, separate, rtol=1e-7, atol=1e-9)


@given(
    data=st.data(), cls=kernels, alpha=alphas, L=lengths,
    h=_finite(0.01, 1.0), c=_finite(-50.0, 50.0),
)
def test_apply_is_linear(data, cls, alpha, L, h, c):
    k = cls(alpha, L)
    n = data.draw(st.integers(8, 128))
    x = data.draw(arrays(np.float64, (n,), elements=_finite(-10.0, 10.0)))
    y = data.draw(arrays(np.float64, (n,), elements=_finite(-10.0, 10.0)))
    scaled = np.asarray(k.apply(c * x, h))
    assert np.allclose(scaled, c * np.asarray(k.apply(x, h)), rtol=1e-7, atol=1e-9)
    summed = np.asarray(k.apply(x + y, h))
    parts = np.asarray(k.apply(x, h)) + np.asarray(k.apply(y, h))
    assert np.allclose(summed, parts, rtol=1e-7, atol=1e-9)


# --- Grunwald-Letnikov closed form --------------------------------------------

@given(alpha=alphas, L=lengths)
def test_gl_leading_is_alpha_and_forcing_is_one(alpha, L):
    k = GLKernel(alpha, L)
    # leading = -c_1 = alpha, up to a floating-point ULP (c_1 = (1-(1+alpha))*c_0).
    assert np.isclose(k.leading, alpha, rtol=1e-12, atol=0.0)
    assert k.forcing_factor == 1.0


@given(alpha=alphas, L=lengths)
def test_gl_strict_past_weights_are_negative(alpha, L):
    # For alpha in (0,1) the GL coefficients c_j (j>=1) are all <= 0.
    assert np.all(np.asarray(GLKernel(alpha, L).weights) <= 0.0)


@given(alpha=alphas, L=lengths)
def test_gl_coefficients_satisfy_the_recursion(alpha, L):
    # Reconstructed full operator: a = [c_0, c_1, ..., c_L] = [1, -leading, w_2..w_L].
    k = GLKernel(alpha, L)
    a = np.concatenate([[1.0, -k.leading], np.asarray(k.weights)])
    j = np.arange(1, L + 1)
    expected = (1.0 - (1.0 + alpha) / j) * a[:-1]
    assert np.allclose(a[1:], expected, rtol=1e-10, atol=1e-12)


@given(alpha=alphas, L=lengths)
def test_gl_linear_memory_gain_in_unit_interval(alpha, L):
    # gain = leading - sum(weights) = 1 - sum_{j=0}^L c_j, strictly inside (0, 1):
    # the truncated GL series approaches the unit-gain limit from below.
    k = GLKernel(alpha, L)
    gain = float(k.leading - np.sum(np.asarray(k.weights)))
    assert 0.0 < gain < 1.0


@given(alpha=alphas, L=lengths, extra=st.integers(1, 64))
def test_gl_linear_memory_gain_increases_with_history(alpha, L, extra):
    # Each added strict-past term is negative, so the gain rises toward 1 with L.
    def gain(length):
        k = GLKernel(alpha, length)
        return float(k.leading - np.sum(np.asarray(k.weights)))

    assert gain(L + extra) >= gain(L) - 1e-12


# --- L1 (Caputo) closed form --------------------------------------------------

@given(alpha=alphas, L=lengths)
def test_l1_leading_and_forcing_match_closed_form(alpha, L):
    k = L1CaputoKernel(alpha, L)
    assert np.isclose(k.leading, 2.0 - 2.0 ** (1.0 - alpha), rtol=1e-10)
    assert np.isclose(k.forcing_factor, float(gamma(2.0 - alpha)), rtol=1e-10)


@given(alpha=alphas, L=lengths)
def test_l1_strict_past_weights_are_negative(alpha, L):
    # L1 weights b_j - b_{j-1} are negative (b_m is monotonically decreasing).
    assert np.all(np.asarray(L1CaputoKernel(alpha, L).weights) < 0.0)
