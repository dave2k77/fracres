"""Discrete fractional memory kernels.

A *fractional kernel* discretises a temporal fractional derivative ``D^alpha``.
Plugged into a reservoir it yields the stable one-step update (knowledge base
Section 3.2, in its standard gain-normalised form)::

    x_k = leading * x_{k-1}                       # retention of the leading state
          - sum_{j=2}^{L} w_j x_{k-j}             # strict-past fractional memory
          + forcing_factor * h^alpha * g_k        # scaled driving term g_k

where ``g_k = -lambda x_{k-1} + f(W_res x_{k-1} + W_in u_k)`` is supplied by the
reservoir, ``h`` is the reservoir step size, and ``lambda`` is the node decay.

Each kernel therefore exposes three quantities consumed by the reservoir:

* ``leading``        -- coefficient on the most recent state ``x_{k-1}``,
* ``weights``        -- strict-past coefficients ``w_2 .. w_L`` (length ``L-1``),
* ``forcing_factor`` -- dimensionless scale on the (already ``h^alpha``-scaled)
                        driving term.

Design note
-----------
The "naive" form ``x_{k-1} - memory + activation`` (leading coefficient 1, no
``h^alpha`` scaling) double-counts ``x_{k-1}``: the effective gain on the leading
state is ``1 + alpha``, which diverges. Using ``leading = -c_1 = alpha`` for the
Grünwald-Letnikov scheme makes the linear memory gain exactly 1, and the decay
``lambda`` then pulls it strictly below 1 (the Echo State Property). This is a
deliberate, documented departure from the literal coefficients written in
``docs/knowledge_base.md`` Section 3.2.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


class AbstractFractionalKernel(eqx.Module):
    """Base class for all discrete fractional memory kernels."""

    alpha: float
    history_length: int
    leading: float
    weights: jnp.ndarray
    forcing_factor: float

    def __call__(self, x_history: jnp.ndarray) -> jnp.ndarray:
        """Strict-past fractional memory ``sum_{j=2}^{L} w_j x_{k-j}``.

        Parameters
        ----------
        x_history : array, shape ``(history_length, res_size)``
            Full rolling buffer; row 0 is the current state ``x_{k-1}``.

        Returns
        -------
        array, shape ``(res_size,)``
        """
        # weights are w_2..w_L; x_history[1:] are x_{k-2}..x_{k-L}.
        return jnp.einsum("j,jk->k", self.weights, x_history[1:])

    def apply(self, signal: jnp.ndarray, step_size: float) -> jnp.ndarray:
        """Estimate the fractional derivative ``D^alpha signal`` on a uniform grid.

        Reconstructs the *full* discrete operator from the stored
        ``leading``/``weights``/``forcing_factor`` and applies it as a causal FIR
        filter::

            a = [1, -leading, w_2, ..., w_L]
            D^alpha x_k ~= (1 / (forcing_factor * h^alpha))
                           * sum_{j=0}^{min(k, L)} a_j x_{k-j}

        For Grünwald-Letnikov this recovers ``a = [c_0, c_1, ..., c_L]`` exactly
        (``c_1 = -leading = -alpha``); for the L1 scheme it recovers the
        telescoped coefficients ``[1, b_1 - 1, b_2 - b_1, ...]``. This is the
        operator view of the kernel, decoupled from the reservoir recurrence, and
        is what the validation suite checks against the analytic ``D^alpha t^beta``.

        Parameters
        ----------
        signal : array, shape ``(T,)`` or ``(T, channels)``
            Samples ``x(t_k)`` on a uniform grid ``t_k = k * step_size``.
        step_size : float
            Grid spacing ``h``.

        Returns
        -------
        array, same shape as ``signal``
            Estimate of ``D^alpha x`` at each ``t_k``. Early samples (``k < L``)
            are warm-up: a truncated history makes them inaccurate, and small-``t``
            relative error is intrinsically larger.
        """
        a = jnp.concatenate([jnp.asarray([1.0, -self.leading], dtype=self.weights.dtype), self.weights])
        scale = self.forcing_factor * step_size**self.alpha

        def conv1d(x):
            return jnp.convolve(x, a)[: x.shape[0]]

        if signal.ndim == 1:
            out = conv1d(signal)
        elif signal.ndim == 2:
            out = jax.vmap(conv1d, in_axes=1, out_axes=1)(signal)
        else:
            raise ValueError("signal must be 1-D (T,) or 2-D (T, channels)")
        return out / scale


class GLKernel(AbstractFractionalKernel):
    """Grünwald-Letnikov power-law memory operator.

    Generalised-binomial coefficients ``c_0 = 1, c_j = (1 - (1+alpha)/j) c_{j-1}``.
    The leading retention is ``leading = -c_1 = alpha`` and the strict-past
    weights are ``c_2 .. c_L`` (giving unit linear gain before decay).
    """

    def __init__(self, alpha: float, history_length: int):
        self.alpha = alpha
        self.history_length = history_length
        c = self._gl_coeffs(alpha, history_length)  # c_0 .. c_L
        self.leading = float(-c[1])  # = alpha
        self.weights = jnp.asarray(c[2:])  # c_2 .. c_L  (length L-1)
        self.forcing_factor = 1.0

    @staticmethod
    def _gl_coeffs(alpha: float, L: int) -> list[float]:
        coeffs = [1.0]
        for j in range(1, L + 1):
            coeffs.append((1.0 - (1.0 + alpha) / j) * coeffs[-1])
        return coeffs


class L1CaputoKernel(AbstractFractionalKernel):
    """L1 finite-difference scheme for the Caputo fractional derivative.

    Built from the L1 weights ``b_m = (m+1)^{1-alpha} - m^{1-alpha}`` (``b_0 = 1``,
    monotonically decreasing). Rearranging the L1 difference scheme into an
    explicit one-step update gives leading retention ``1 - b_1``, strict-past
    weights ``b_j - b_{j-1}`` (small, negative), and a forcing factor
    ``Gamma(2 - alpha)``. Unlike GL these are dimensionless: the ``h^alpha`` step
    scaling is applied once, by the reservoir.
    """

    def __init__(self, alpha: float, history_length: int):
        self.alpha = alpha
        self.history_length = history_length
        m = jnp.arange(history_length + 1)  # 0 .. L
        b = (m + 1) ** (1.0 - alpha) - m ** (1.0 - alpha)  # b_0 .. b_L
        self.leading = float(1.0 - b[1])
        j = jnp.arange(2, history_length + 1)  # 2 .. L
        self.weights = b[j] - b[j - 1]  # w_j = b_j - b_{j-1}  (length L-1)
        self.forcing_factor = float(jax.scipy.special.gamma(2.0 - alpha))
