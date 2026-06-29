"""Cross-validate the self-contained fracres kernels against ``hpfracc.ops``.

fracres ships its *own* discrete fractional kernels (``GLKernel`` /
``L1CaputoKernel``) rather than depending on the ``hpfracc`` library, because the
reservoir needs the *decomposed recurrence* form (``leading`` / ``weights`` /
``forcing_factor`` advanced one step at a time over a rolling buffer), whereas
``hpfracc.ops`` exposes the *batch operator* form (whole signal in, ``D^alpha x``
out). The two are different operator views of the same mathematics.

This module is the "single validated source of truth" check from the roadmap
*without* coupling the two packages: it asserts that the operator reconstructed
by :meth:`AbstractFractionalKernel.apply` reproduces, to floating-point
tolerance, the full-history operator computed by ``hpfracc`` on the same signal.
It runs only where ``hpfracc`` happens to be installed; everywhere else (the
default CI) it is skipped, so fracres keeps *zero* hard dependency on hpfracc.

Equivalence being asserted (both in float64):

* **GL** -- fracres ``apply`` builds ``a = [c_0, c_1, c_2, ...]`` and returns
  ``(1/h^a) sum_j c_j x[k-j]``; ``hpfracc.ops.grunwald_letnikov`` returns
  ``(1/h^a) sum_k w_k x[i-k]`` with the *same* binomial recurrence
  ``w_0 = 1, w_k = w_{k-1} (k - 1 - a) / k``. Identical operator, so they must
  agree on *any* signal (including one with ``x(0) != 0``).
* **L1 / Caputo** -- ``hpfracc.ops.caputo`` convolves the increments
  ``x_k - x_{k-1}`` with ``b_k = (k+1)^{1-a} - k^{1-a}`` and divides by
  ``Gamma(2-a) h^a``; fracres applies the *telescoped* coefficients
  ``[1, b_1 - 1, b_2 - b_1, ...]`` directly to the signal. Telescoping makes
  these the same operator, matching when ``x(0) = 0``.

A mismatch here means one side changed a coefficient convention, a sign, or the
``h^alpha`` / ``Gamma(2-alpha)`` normalisation -- exactly the drift this guard
exists to catch.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from fracres import GLKernel, L1CaputoKernel  # noqa: E402

# Skip the whole module unless hpfracc is importable. This is what keeps the
# cross-check dependency-free: fracres never *requires* hpfracc to test or ship.
hp_ops = pytest.importorskip(
    "hpfracc.ops",
    reason="hpfracc not installed; cross-validation is opt-in (see module docstring).",
)

T = 512  # kept modest: hpfracc 'full' history materialises a T x T matrix.
H = 0.01
ALPHAS = [0.3, 0.5, 0.7, 0.9]
# Compare on the interior only: index 0 is a shared warm-up boundary (both give
# c_0 * x_0 for GL, 0 for the differenced L1), uninteresting for the equivalence.
INTERIOR = slice(1, T)


def _rel_err(est, ref):
    """Max relative error on the interior, with a small absolute floor."""
    est = jnp.asarray(est)[INTERIOR]
    ref = jnp.asarray(ref)[INTERIOR]
    return float(jnp.max(jnp.abs(est - ref) / (jnp.abs(ref) + 1e-12)))


@pytest.mark.parametrize("alpha", ALPHAS)
def test_gl_apply_matches_hpfracc_grunwald_letnikov(alpha):
    # GL is a pure causal convolution (no differencing), so it must agree even
    # on a signal with a non-zero initial value -- the case that separates GL
    # from the Caputo L1 scheme.
    t = jnp.arange(T) * H
    signal = jnp.sin(0.7 * t) + 1.5  # x(0) = 1.5 != 0

    fracres_op = GLKernel(alpha=alpha, history_length=T).apply(signal, H)
    hpfracc_op = hp_ops.grunwald_letnikov(signal, dt=H, order=alpha)

    # Same arithmetic in a different summation order (FIR convolve vs dense
    # matrix contraction) -> agreement to ~1e-9; 1e-6 is a safe, meaningful cap.
    assert _rel_err(fracres_op, hpfracc_op) < 1e-6


@pytest.mark.parametrize("alpha", ALPHAS)
def test_l1_apply_matches_hpfracc_caputo(alpha):
    # The L1/Caputo scheme differences the signal, so the operators coincide
    # when x(0) = 0; t**beta (beta > 0) gives exactly that.
    t = jnp.arange(T) * H
    signal = t**2.0  # x(0) = 0

    fracres_op = L1CaputoKernel(alpha=alpha, history_length=T).apply(signal, H)
    hpfracc_op = hp_ops.caputo(signal, dt=H, order=alpha)

    # The telescoped (fracres) and differenced (hpfracc) forms are algebraically
    # identical but accumulate slightly differently; 1e-6 still catches real drift.
    assert _rel_err(fracres_op, hpfracc_op) < 1e-6
