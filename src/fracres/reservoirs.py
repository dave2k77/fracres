"""Fractional reservoirs.

A reservoir is a large, fixed, randomly connected pool of non-linear nodes whose
internal dynamics carry fractional (power-law) memory supplied by an injected
:class:`~fracres.kernels.AbstractFractionalKernel`. Only the readout is trained;
the reservoir weights are frozen near the edge of chaos.

Two variants are provided:

* :class:`FractionalReservoir` -- the base parameterised reservoir. With a
  :class:`~fracres.kernels.GLKernel` this is the single-population reduction of
  the fractional neural-mass reservoir (knowledge base v2 Section 2.2 / 3). The
  activation is ``tanh`` (zero-centred, avoids DC drift -- see :mod:`fracres.kernels`),
  not the ``sigmoid`` of the v1 pseudocode.
* :class:`qSOCFractionalReservoir` -- adds a quasi-self-organised-criticality
  homeostatic threshold ``b_t`` that pulls macroscopic energy toward a critical
  set-point (knowledge base v2 Section 5).
"""
from __future__ import annotations

from typing import Callable

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

    Stable Grünwald-Letnikov / Caputo discrete update (see :mod:`fracres.kernels`)::

        x_k = leading * x_{k-1} - sum_{j=2}^{L} w_j x_{k-j}
                      + forcing * [ -lambda x_{k-1} + tanh(W_res x_{k-1} + W_in u_k) ]

    with ``forcing = kernel.forcing_factor * h^alpha``. The kernel-supplied
    ``leading``/``w_j`` give unit linear memory gain; the ``-lambda x_{k-1}`` leak
    then pulls the gain strictly below 1 (Echo State Property). ``tanh`` (rather
    than ``sigmoid``) keeps the drive zero-centred and avoids DC drift.

    Parameters
    ----------
    in_features : int
        Dimension of the driving input ``u_t``.
    res_size : int
        Number of reservoir nodes.
    fractional_operator : AbstractFractionalKernel
        The memory kernel (also fixes ``history_length`` and ``alpha``).
    key : jax.Array
        PRNG key for weight initialisation.
    spectral_scale : float, default 0.95
        Edge-of-chaos scaling for ``W_res`` (``rho ~ spectral_scale``).
    step_size : float, default 0.1
        Integration step ``h``; the activation/leak are scaled by ``h^alpha``.
    decay : float, default 1.0
        Node decay rate ``lambda`` (the ``Lambda`` diagonal, scalar here).
    """

    W_res: jnp.ndarray
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    alpha: float
    spectral_scale: float
    step_size: float
    decay: float

    def __init__(
        self,
        in_features: int,
        res_size: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        spectral_scale: float = 0.95,
        step_size: float = 0.1,
        decay: float = 1.0,
    ):
        k1, k2 = jax.random.split(key)
        # Random connectome scaled near the edge of chaos (spectral radius ~ 1).
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * spectral_scale / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        self.alpha = fractional_operator.alpha
        self.spectral_scale = spectral_scale
        self.step_size = step_size
        self.decay = decay

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
        op = self.fractional_operator
        fractional_memory = op(x_history)
        current_state = x_history[0]
        forcing = op.forcing_factor * self.step_size**self.alpha
        activation = jnp.tanh(self.W_res @ current_state + self.W_in @ u_t)
        x_next = (
            op.leading * current_state
            - fractional_memory
            + forcing * (-self.decay * current_state + activation)
        )
        return x_next, _roll_in(x_history, x_next)


class qSOCFractionalReservoir(eqx.Module):
    """Fractional reservoir with quasi-self-organised-criticality homeostasis.

    Extends :class:`FractionalReservoir` with an adaptive threshold ``b_t`` driven
    by the discrepancy between a target critical energy ``E_crit`` and the
    *windowed* macroscopic energy ``E`` (knowledge base v2 Section 5.1):

        tau_soc dE/dt = ||x||^2 - E                 (leaky-integrator energy window)
        tau_b   db/dt = -b + gamma (E_crit - E)     (homeostatic threshold)

    Both states are carried through the scan. Unlike the v1 controller -- which
    *defined* a windowed energy but *used* the instantaneous ``||x||^2`` -- the
    written and implemented controllers now agree. The threshold uses the
    unconditionally-stable semi-implicit Euler step
    ``b <- (b + (dt/tau_b) gamma (E_crit - E)) / (1 + dt/tau_b)``, avoiding the
    explicit step's ``dt < 2 tau_b`` restriction.
    """

    W_res: jnp.ndarray
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    alpha: float
    spectral_scale: float
    step_size: float
    decay: float
    E_crit: float
    tau_b: float
    gamma: float
    tau_soc: float

    def __init__(
        self,
        in_features: int,
        res_size: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        spectral_scale: float = 0.95,
        step_size: float = 0.1,
        decay: float = 1.0,
        E_crit: float = 1.0,
        tau_b: float = 1.0,
        gamma: float = 1.0,
        tau_soc: float = 10.0,
    ):
        k1, k2 = jax.random.split(key)
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * spectral_scale / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        self.alpha = fractional_operator.alpha
        self.spectral_scale = spectral_scale
        self.step_size = step_size
        self.decay = decay
        self.E_crit = E_crit
        self.tau_b = tau_b
        self.gamma = gamma
        self.tau_soc = tau_soc

    def __call__(
        self,
        u_t: jnp.ndarray,
        x_history: jnp.ndarray,
        b_t: jnp.ndarray,
        e_t: jnp.ndarray,
        dt: float,
    ):
        """Advance one step, carrying threshold ``b_t`` and windowed energy ``e_t``.

        Returns ``(x_next, new_history, b_next, e_next)``.
        """
        op = self.fractional_operator
        fractional_memory = op(x_history)
        current_state = x_history[0]
        forcing = op.forcing_factor * self.step_size**self.alpha
        activation = jnp.tanh(self.W_res @ current_state + self.W_in @ u_t + b_t)
        x_next = (
            op.leading * current_state
            - fractional_memory
            + forcing * (-self.decay * current_state + activation)
        )

        # Low-pass the instantaneous energy into the windowed estimate E (matches
        # the written E_avg); explicit Euler is fine here (a stable 1st-order LPF).
        inst_energy = jnp.mean(jnp.square(x_next))
        e_next = e_t + dt * (inst_energy - e_t) / self.tau_soc

        # Homeostatic threshold: unconditionally-stable semi-implicit Euler step.
        b_next = (
            b_t + (dt / self.tau_b) * self.gamma * (self.E_crit - e_next)
        ) / (1.0 + dt / self.tau_b)

        return x_next, _roll_in(x_history, x_next), b_next, e_next


class WilsonCowanReservoir(eqx.Module):
    """Fractional excitatory/inhibitory neural-mass reservoir (KB v2 §2.2).

    ``N`` coupled Wilson-Cowan (1972) masses, each split into an excitatory ``E``
    and an inhibitory ``I`` subpopulation sharing the fractional order
    ``alpha_D`` but with **separate** time constants ``tau_E``, ``tau_I``::

        tau_E^a D^a E = -E + S_E(W_EE E - W_EI I + W_in u)
        tau_I^a D^a I = -I + S_I(W_IE E - W_II I)

    Writing each line as ``D^a x = g`` with the RHS divided by ``tau^a`` gives a
    population-wise driving term ``g_x = (-x_{k-1} + S(syn)) / tau_x^a`` -- so
    ``1/tau_x^a`` plays the role of the node decay ``lambda`` in
    :class:`FractionalReservoir`, scaling *both* the leak and the firing-rate
    drive. The two populations are stacked into one state ``z = [E; I]`` of width
    ``2N`` and advanced with the *same* validated Grünwald-Letnikov update as the
    single-population reservoir (shared ``leading``/``weights``/``forcing_factor``,
    since they depend only on ``alpha_D``)::

        z_k = leading * z_{k-1} - sum_{j>=2} c_j z_{k-j} + forcing * h^a * g_k

    The four connectomes hold **non-negative** synaptic magnitudes; the
    excitatory/inhibitory signs are explicit in the equations above. The
    firing-rate ``S`` is a logistic sigmoid by default (the Wilson-Cowan standard:
    rates are non-negative), *not* the zero-centred ``tanh`` of the single-pop
    reduction; the ``-x`` leak keeps the bounded positive drive from drifting.

    Parameters
    ----------
    in_features : int
        Dimension of the driving input ``u_t`` (injected into ``E`` only).
    res_size : int
        Number of masses ``N`` per population; the stacked state has width ``2N``.
    fractional_operator : AbstractFractionalKernel
        Shared memory kernel (fixes ``history_length`` and ``alpha_D``).
    key : jax.Array
        PRNG key for the four connectomes and the input map.
    tau_E, tau_I : float, default 1.0, 2.0
        Excitatory / inhibitory time constants (inhibition slower by default).
    spectral_scale : float, default 0.95
        Edge-of-chaos scaling applied to every connectome block (``~1/sqrt(N)``).
    step_size : float, default 0.1
        Integration step ``h``; the drive is scaled by ``h^alpha``.
    firing_rate : callable, default ``jax.nn.sigmoid``
        Population firing-rate non-linearity ``S`` (applied to both ``E`` and ``I``).
    """

    W_EE: jnp.ndarray
    W_EI: jnp.ndarray
    W_IE: jnp.ndarray
    W_II: jnp.ndarray
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    alpha: float
    tau_E: float
    tau_I: float
    spectral_scale: float
    step_size: float
    firing_rate: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        res_size: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        tau_E: float = 1.0,
        tau_I: float = 2.0,
        spectral_scale: float = 0.95,
        step_size: float = 0.1,
        firing_rate: Callable = jax.nn.sigmoid,
    ):
        kEE, kEI, kIE, kII, kin = jax.random.split(key, 5)
        scale = spectral_scale / jnp.sqrt(res_size)
        # Non-negative synaptic magnitudes; E/I signs live in the update equations.
        self.W_EE = jnp.abs(jax.random.normal(kEE, (res_size, res_size))) * scale
        self.W_EI = jnp.abs(jax.random.normal(kEI, (res_size, res_size))) * scale
        self.W_IE = jnp.abs(jax.random.normal(kIE, (res_size, res_size))) * scale
        self.W_II = jnp.abs(jax.random.normal(kII, (res_size, res_size))) * scale
        self.W_in = jax.random.normal(kin, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        self.alpha = fractional_operator.alpha
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.spectral_scale = spectral_scale
        self.step_size = step_size
        self.firing_rate = firing_rate

    def __call__(self, u_t: jnp.ndarray, z_history: jnp.ndarray):
        """Advance one step.

        Parameters
        ----------
        u_t : array, shape ``(in_features,)``
        z_history : array, shape ``(history_length, 2 * res_size)``
            Rolling buffer of stacked states ``[E; I]``; row 0 is ``z_{k-1}``.

        Returns
        -------
        (z_next, new_history)
        """
        op = self.fractional_operator
        n = self.W_EE.shape[0]
        current = z_history[0]
        E_prev, I_prev = current[:n], current[n:]

        syn_E = self.W_EE @ E_prev - self.W_EI @ I_prev + self.W_in @ u_t
        syn_I = self.W_IE @ E_prev - self.W_II @ I_prev
        g_E = (-E_prev + self.firing_rate(syn_E)) / self.tau_E**self.alpha
        g_I = (-I_prev + self.firing_rate(syn_I)) / self.tau_I**self.alpha
        g = jnp.concatenate([g_E, g_I])

        fractional_memory = op(z_history)
        forcing = op.forcing_factor * self.step_size**self.alpha
        z_next = op.leading * current - fractional_memory + forcing * g
        return z_next, _roll_in(z_history, z_next)


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
