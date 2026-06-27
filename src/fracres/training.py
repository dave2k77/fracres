"""Training: Besov-regularised readout optimisation.

Only the readout is trained. ``eqx.filter_value_and_grad`` differentiates through
the model but, combined with the optimiser acting on the readout leaves, leaves
the frozen reservoir untouched. The loss combines predictive MSE with the
topological Besov penalty of :mod:`fracres.regularizers`.
"""
from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
import optax

from fracres.regularizers import littlewood_paley_penalty


def compute_loss(model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg, margin=0.05):
    """MSE + ``lambda_reg`` * Besov-norm penalty.

    The Besov indices are derived from the physical parameters, staying strictly
    inside the regularity bounds ``p < alpha`` and ``s < H``::

        p = alpha - margin,   s = H - margin,   q = p
    """
    p = alpha - margin  # integrability bound p < alpha
    s = H_hurst - margin  # smoothness bound s < H
    q = p  # standard Besov-Slobodeckij geometry

    Y_hat = model(U_drive)
    mse_loss = jnp.mean((Y_target - Y_hat) ** 2)
    besov_loss = littlewood_paley_penalty(Y_hat, masks, s, p, q)
    return mse_loss + lambda_reg * besov_loss


@eqx.filter_jit
def train_step(
    model, optimizer, opt_state, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg
):
    """One optimisation step. Returns ``(model, opt_state, loss)``."""
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
