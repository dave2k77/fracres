"""Phantom-brain models: reservoir + readout assembled into a generative model.

A *phantom brain* maps a stochastic driving trajectory ``U_drive`` (shape
``(time_steps, in_features)``) to a synthetic observable trajectory ``Y_hat``
(shape ``(time_steps, out_features)``) by scanning the reservoir over time and
projecting each state through the readout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from fracres.kernels import AbstractFractionalKernel
from fracres.readout import TopologicalReadout
from fracres.reservoirs import FractionalReservoir, qSOCFractionalReservoir


class PhantomBrain(eqx.Module):
    """Fractional reservoir + linear readout (no homeostasis)."""

    reservoir: FractionalReservoir
    readout: TopologicalReadout

    def __init__(
        self,
        in_features: int,
        res_size: int,
        out_features: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        spectral_scale: float = 0.95,
        step_size: float = 0.1,
        decay: float = 1.0,
    ):
        k1, k2 = jax.random.split(key)
        self.reservoir = FractionalReservoir(
            in_features, res_size, fractional_operator, k1, spectral_scale, step_size, decay
        )
        self.readout = TopologicalReadout(res_size, out_features, k2)

    def __call__(self, U_drive: jnp.ndarray) -> jnp.ndarray:
        """Return only the observable trajectory ``Y_hat``."""
        _, Y_hat = self.simulate(U_drive)
        return Y_hat

    def simulate(self, U_drive: jnp.ndarray):
        """Return ``(X_states, Y_hat)`` -- reservoir states and observables."""
        res_size = self.reservoir.W_res.shape[0]

        def step_fn(history_buffer, u_t):
            x_next, updated = self.reservoir(u_t, history_buffer)
            return updated, (x_next, self.readout(x_next))

        init_history = jnp.zeros((self.reservoir.history_length, res_size))
        _, (X_states, Y_hat) = jax.lax.scan(step_fn, init_history, U_drive)
        return X_states, Y_hat


class qSOCPhantomBrain(eqx.Module):
    """Phantom brain with a qSOC homeostatic reservoir.

    Carries the adaptive threshold ``b_t`` through the scan in addition to the
    history buffer, and exposes the threshold trajectory for analysis.
    """

    reservoir: qSOCFractionalReservoir
    readout: TopologicalReadout

    def __init__(
        self,
        in_features: int,
        res_size: int,
        out_features: int,
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
        self.reservoir = qSOCFractionalReservoir(
            in_features, res_size, fractional_operator, k1,
            spectral_scale, step_size, decay, E_crit, tau_b, gamma, tau_soc,
        )
        self.readout = TopologicalReadout(res_size, out_features, k2)

    def __call__(self, U_drive: jnp.ndarray, dt: float = 0.01) -> jnp.ndarray:
        _, Y_hat, _, _ = self.simulate(U_drive, dt)
        return Y_hat

    def simulate(self, U_drive: jnp.ndarray, dt: float = 0.01):
        """Return ``(X_states, Y_hat, B_thresholds, E_energy)``.

        ``E_energy`` is the windowed macroscopic-energy trajectory tracked by the
        qSOC controller (scalar per time step).
        """
        res_size = self.reservoir.W_res.shape[0]

        def step_fn(carry, u_t):
            x_history, b_t, e_t = carry
            x_next, updated_history, b_next, e_next = self.reservoir(
                u_t, x_history, b_t, e_t, dt
            )
            y_hat_t = self.readout(x_next)
            return (updated_history, b_next, e_next), (x_next, y_hat_t, b_t, e_t)

        init_history = jnp.zeros((self.reservoir.history_length, res_size))
        init_bias = jnp.zeros((res_size,))
        init_energy = jnp.array(0.0)
        _, (X_states, Y_hat, B_thresholds, E_energy) = jax.lax.scan(
            step_fn, (init_history, init_bias, init_energy), U_drive
        )
        return X_states, Y_hat, B_thresholds, E_energy
