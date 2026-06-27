"""Fractional reservoirs.

A reservoir is a large, fixed, randomly connected pool of non-linear nodes whose
internal dynamics carry fractional (power-law) memory supplied by an injected
:class:`~fracres.kernels.AbstractFractionalKernel`. Only the readout is trained;
the reservoir weights are frozen near the edge of chaos.

Two variants are provided:

* :class:`FractionalReservoir` -- the base parameterised reservoir. With a
  :class:`~fracres.kernels.GLKernel` and a sigmoid activation this is exactly the
  fractional neural-mass reservoir of the knowledge base (Section 2.2 / 3).
* :class:`qSOCFractionalReservoir` -- adds a quasi-self-organised-criticality
  homeostatic threshold ``b_t`` that pulls macroscopic energy toward a critical
  set-point (Section 5).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from fracres.kernels import AbstractFractionalKernel


def _roll_in(x_history: jnp.ndarray, x_next: jnp.ndarray) -> jnp.ndarray:
    """Shift the rolling history buffer down and insert ``x_next`` at index 0."""
    rolled = jnp.roll(x_history, shift=1, axis=0)
    return rolled.at[0].set(x_next)


class FractionalReservoir(eqx.Module):
    """Parameterised fractional reservoir with an injected memory kernel.

    State update (Grünwald-Letnikov / Caputo discretisation)::

        x_k = x_{k-1} - memory(x_{k-2..k-L}) + sigma(W_res x_{k-1} + W_in u_k)

    Parameters
    ----------
    in_features : int
        Dimension of the driving input ``u_t``.
    res_size : int
        Number of reservoir nodes.
    fractional_operator : AbstractFractionalKernel
        The memory kernel (also fixes ``history_length``).
    key : jax.Array
        PRNG key for weight initialisation.
    spectral_scale : float, default 0.95
        Edge-of-chaos scaling for ``W_res`` (``rho ~ spectral_scale``).
    """

    W_res: jnp.ndarray
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    spectral_scale: float

    def __init__(
        self,
        in_features: int,
        res_size: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        spectral_scale: float = 0.95,
    ):
        k1, k2 = jax.random.split(key)
        # Random connectome scaled near the edge of chaos (spectral radius ~ 1).
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * spectral_scale / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        self.spectral_scale = spectral_scale

    def __call__(self, u_t: jnp.ndarray, x_history: jnp.ndarray):
        """Advance one step.

        Parameters
        ----------
        u_t : array, shape ``(in_features,)``
        x_history : array, shape ``(history_length, res_size)``

        Returns
        -------
        (x_next, new_history)
        """
        fractional_memory = self.fractional_operator(x_history[:-1])
        current_state = x_history[0]
        activation = jax.nn.sigmoid(self.W_res @ current_state + self.W_in @ u_t)
        x_next = current_state - fractional_memory + activation
        return x_next, _roll_in(x_history, x_next)


class qSOCFractionalReservoir(eqx.Module):
    """Fractional reservoir with quasi-self-organised-criticality homeostasis.

    Extends :class:`FractionalReservoir` with an adaptive threshold ``b_t`` that
    is driven by the discrepancy between a target critical energy ``E_crit`` and
    the instantaneous macroscopic energy, integrated with time constant
    ``tau_b``::

        tau_b db/dt = -b + gamma (E_crit - ||x||^2)

    This feedback anchors the trajectory inside the tight Besov bounds and keeps
    the network from drifting into super- or sub-criticality.
    """

    W_res: jnp.ndarray
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    E_crit: float
    tau_b: float
    gamma: float

    def __init__(
        self,
        in_features: int,
        res_size: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        E_crit: float = 1.0,
        tau_b: float = 1.0,
        gamma: float = 1.0,
    ):
        k1, k2 = jax.random.split(key)
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * 0.95 / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        self.E_crit = E_crit
        self.tau_b = tau_b
        self.gamma = gamma

    def __call__(self, u_t: jnp.ndarray, x_history: jnp.ndarray, b_t: jnp.ndarray, dt: float):
        """Advance one step, carrying the dynamic homeostatic threshold ``b_t``.

        Returns ``(x_next, new_history, b_next)``.
        """
        fractional_memory = self.fractional_operator(x_history[:-1])
        current_state = x_history[0]
        activation = jax.nn.sigmoid(self.W_res @ current_state + self.W_in @ u_t + b_t)
        x_next = current_state - fractional_memory + activation

        # Homeostatic threshold update (explicit Euler step of the qSOC ODE).
        current_energy = jnp.mean(jnp.square(x_next))
        db = (-b_t + self.gamma * (self.E_crit - current_energy)) / self.tau_b
        b_next = b_t + db * dt

        return x_next, _roll_in(x_history, x_next), b_next


class NeuralFieldReservoir(eqx.Module):
    """Spatially-extended neural *field* reservoir (Amari-type). Not yet implemented.

    Placeholder for the continuum variant where ``W_res`` encodes a spatial
    connectivity kernel over a cortical sheet rather than discrete random links.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "NeuralFieldReservoir is a planned variant; see docs/knowledge_base.md "
            "Section 2.2 (Neural Field Models)."
        )
