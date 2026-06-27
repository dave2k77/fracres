"""Readout layer.

The readout is the *only* trained part of the model: a linear map from the
high-dimensional reservoir state to the observable (e.g. EEG/MEG) space. It can
be fit in closed form by ridge regression, or learned by gradient descent
together with a topological (Besov) regulariser -- see :mod:`fracres.training`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


class TopologicalReadout(eqx.Module):
    """Linear readout ``y = W_out x``.

    Parameters
    ----------
    res_size : int
        Reservoir dimension (input to the readout).
    out_features : int
        Observable output dimension.
    key : jax.Array
        PRNG key for initialisation.
    """

    W_out: jnp.ndarray

    def __init__(self, res_size: int, out_features: int, key: jax.Array):
        self.W_out = jax.random.normal(key, (out_features, res_size))

    def __call__(self, x_t: jnp.ndarray) -> jnp.ndarray:
        return self.W_out @ x_t
