"""Discrete fractional memory kernels.

A *fractional kernel* turns a rolling buffer of past reservoir states into a
single ``(res_size,)`` "fractional memory" vector via a power-law weighted sum.
This is the discrete realisation of a temporal fractional derivative / integral
(see ``docs/knowledge_base.md`` Section 3).

All kernels share the same interface so they can be injected interchangeably
into a reservoir:

    memory = kernel(history_buffer)   # (L-1, res_size) -> (res_size,)

where ``L == kernel.history_length`` and the buffer passed in is the *past*
states ``x_history[:-1]`` (the current state is handled by the reservoir).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


class AbstractFractionalKernel(eqx.Module):
    """Base class for all discrete fractional memory kernels.

    Subclasses populate ``weights`` (shape ``(history_length - 1,)``) at
    construction time; the call simply contracts those weights against the
    history buffer.
    """

    alpha: float
    history_length: int
    weights: jnp.ndarray

    def __call__(self, history_buffer: jnp.ndarray) -> jnp.ndarray:
        """Apply the non-Markovian memory convolution.

        Parameters
        ----------
        history_buffer : array, shape ``(history_length - 1, res_size)``
            Past reservoir states, most-recent first.

        Returns
        -------
        array, shape ``(res_size,)``
            The fractional memory vector.
        """
        return jnp.einsum("j,jk->k", self.weights, history_buffer)


class GLKernel(AbstractFractionalKernel):
    """Grünwald-Letnikov power-law memory operator.

    Uses the recursive generalised-binomial coefficients
    ``c_0 = 1, c_j = (1 - (1 + alpha) / j) c_{j-1}``. The ``c_0`` term acts on
    the current step (handled in the reservoir update), so the kernel stores
    ``c_1 .. c_{L-1}``.
    """

    def __init__(self, alpha: float, history_length: int):
        self.alpha = alpha
        self.history_length = history_length
        self.weights = self._compute_weights()

    def _compute_weights(self) -> jnp.ndarray:
        coeffs = [1.0]
        for j in range(1, self.history_length):
            coeffs.append((1.0 - (1.0 + self.alpha) / j) * coeffs[-1])
        # Drop c_0; it is applied to the current state by the reservoir.
        return jnp.asarray(coeffs[1:])


class L1CaputoKernel(AbstractFractionalKernel):
    """L1 finite-difference scheme for the Caputo fractional derivative.

    Weights follow the L1 power-law decay ``b_j = j^{1-alpha} - (j-1)^{1-alpha}``
    scaled by ``1 / (Gamma(2 - alpha) * dt^alpha)``.
    """

    dt: float

    def __init__(self, alpha: float, history_length: int, dt: float = 1.0):
        self.alpha = alpha
        self.history_length = history_length
        self.dt = dt
        self.weights = self._compute_weights()

    def _compute_weights(self) -> jnp.ndarray:
        j = jnp.arange(1, self.history_length + 1)
        b_j = j ** (1.0 - self.alpha) - (j - 1) ** (1.0 - self.alpha)
        scaling = 1.0 / (jax.scipy.special.gamma(2.0 - self.alpha) * self.dt**self.alpha)
        # Length L - 1 to match history_buffer = x_history[:-1].
        return b_j[:-1] * scaling
