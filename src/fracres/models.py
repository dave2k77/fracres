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

from typing import Callable

from fracres.kernels import AbstractFractionalKernel
from fracres.readout import TopologicalReadout
from fracres.reservoirs import (
    FractionalReservoir,
    NeuralFieldReservoir,
    WilsonCowanReservoir,
    qSOCFractionalReservoir,
)


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


class WilsonCowanPhantomBrain(eqx.Module):
    """Phantom brain with a fractional Wilson-Cowan E/I neural-mass reservoir.

    The reservoir state is the stacked ``z = [E; I]`` (width ``2N``); the readout
    sees the *full* E/I state, so both populations contribute to the observable.
    ``simulate`` returns ``(X_states, Y_hat)`` exactly like :class:`PhantomBrain`
    (so it works unchanged with the training / closed-form-ridge utilities); the
    excitatory and inhibitory parts are ``X_states[:, :N]`` and ``X_states[:, N:]``.
    Use :meth:`split` to separate them.
    """

    reservoir: WilsonCowanReservoir
    readout: TopologicalReadout

    def __init__(
        self,
        in_features: int,
        res_size: int,
        out_features: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        tau_E: float = 1.0,
        tau_I: float = 2.0,
        spectral_scale: float = 0.95,
        step_size: float = 0.1,
        firing_rate: Callable = jax.nn.sigmoid,
    ):
        k1, k2 = jax.random.split(key)
        self.reservoir = WilsonCowanReservoir(
            in_features, res_size, fractional_operator, k1,
            tau_E, tau_I, spectral_scale, step_size, firing_rate,
        )
        # The readout reads the full stacked E/I state (width 2N).
        self.readout = TopologicalReadout(2 * res_size, out_features, k2)

    def __call__(self, U_drive: jnp.ndarray) -> jnp.ndarray:
        _, Y_hat = self.simulate(U_drive)
        return Y_hat

    def simulate(self, U_drive: jnp.ndarray):
        """Return ``(X_states, Y_hat)`` -- stacked E/I states ``[E; I]`` and observables."""
        n = self.reservoir.W_EE.shape[0]

        def step_fn(z_history, u_t):
            z_next, updated = self.reservoir(u_t, z_history)
            return updated, (z_next, self.readout(z_next))

        init_history = jnp.zeros((self.reservoir.history_length, 2 * n))
        _, (X_states, Y_hat) = jax.lax.scan(step_fn, init_history, U_drive)
        return X_states, Y_hat

    def split(self, X_states: jnp.ndarray):
        """Split a stacked state trajectory into ``(E_states, I_states)``."""
        n = self.reservoir.W_EE.shape[0]
        return X_states[..., :n], X_states[..., n:]


class NeuralFieldPhantomBrain(eqx.Module):
    """Phantom brain with a spatial Amari neural-field reservoir (KB §2.2).

    The reservoir state is the field ``u`` over the cortical sheet (the ring of
    ``N`` sites); the readout maps the whole field to the observable.
    ``simulate`` returns ``(X_states, Y_hat)`` like :class:`PhantomBrain`, so it
    works unchanged with the training / closed-form-ridge utilities;
    ``X_states[t]`` is the spatial field snapshot at step ``t``.
    """

    reservoir: NeuralFieldReservoir
    readout: TopologicalReadout

    def __init__(
        self,
        in_features: int,
        n_nodes: int,
        out_features: int,
        fractional_operator: AbstractFractionalKernel,
        key: jax.Array,
        sigma_e: float = 2.0,
        sigma_i: float = 4.0,
        A_e: float = 1.0,
        A_i: float = 0.5,
        tau: float = 1.0,
        step_size: float = 0.1,
        firing_rate: Callable = jax.nn.sigmoid,
    ):
        k1, k2 = jax.random.split(key)
        self.reservoir = NeuralFieldReservoir(
            in_features, n_nodes, fractional_operator, k1,
            sigma_e, sigma_i, A_e, A_i, tau, step_size, firing_rate,
        )
        self.readout = TopologicalReadout(n_nodes, out_features, k2)

    def __call__(self, U_drive: jnp.ndarray) -> jnp.ndarray:
        _, Y_hat = self.simulate(U_drive)
        return Y_hat

    def simulate(self, U_drive: jnp.ndarray):
        """Return ``(X_states, Y_hat)`` -- field snapshots over the sheet and observables."""
        n = self.reservoir.W_res.shape[0]

        def step_fn(u_history, u_t):
            u_next, updated = self.reservoir(u_t, u_history)
            return updated, (u_next, self.readout(u_next))

        init_history = jnp.zeros((self.reservoir.history_length, n))
        _, (X_states, Y_hat) = jax.lax.scan(step_fn, init_history, U_drive)
        return X_states, Y_hat
