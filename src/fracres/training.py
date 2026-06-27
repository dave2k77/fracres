"""Training: Besov-regularised readout optimisation.

Only the readout is trained. This is enforced *explicitly* by partitioning the
model so that just ``readout.W_out`` is differentiable: the frozen reservoir
weights ``W_res``/``W_in`` and the fixed fractional-kernel ``weights`` never
receive gradients (knowledge base v2 §6.2). The loss combines predictive MSE with
the topological Besov penalty of :mod:`fracres.regularizers`.

Besov index convention (knowledge base v2 §4.2)
-----------------------------------------------
The integrability index ``p`` is bounded by the *heavy-tail stability index*
``alpha_stable`` of the driving law -- NOT the fractional-derivative order. For a
Gaussian fGn drive ``alpha_stable = 2``. The smoothness index obeys the combined
bound ``s < min{H, 1/p}``::

    p = alpha_stable - margin          # p < alpha_S   (finite p-th moments)
    s = min(H, 1/p) - margin           # s < min{H, 1/p}
    q = p

Usage
-----
Initialise the optimiser over the *trainable partition only*::

    diff, _ = eqx.partition(model, readout_filter_spec(model))
    opt_state = optimizer.init(diff)
    model, opt_state, loss = train_step(model, optimizer, opt_state, ...)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from fracres.regularizers import littlewood_paley_penalty


def readout_filter_spec(model):
    """Boolean pytree selecting only ``model.readout.W_out`` as trainable.

    Everything else (reservoir weights, kernel weights, hyper-parameters) is
    marked ``False`` and is therefore frozen under :func:`equinox.partition`.
    """
    spec = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(lambda m: m.readout.W_out, spec, replace=True)


def besov_indices(H_hurst, alpha_stable=2.0, margin=0.05):
    """Derive ``(s, p, q)`` from the physical parameters, inside the strict bounds.

    ``p < alpha_stable`` (heavy-tail integrability) and ``s < min{H, 1/p}``
    (fBm smoothness intersected with the Besov embedding scale).
    """
    p = alpha_stable - margin  # p < alpha_S
    s = min(H_hurst, 1.0 / p) - margin  # s < min{H, 1/p}
    q = p
    return s, p, q


def compute_loss(
    model, U_drive, Y_target, masks, H_hurst, lambda_reg, alpha_stable=2.0, margin=0.05
):
    """MSE + ``lambda_reg`` * Besov-norm penalty.

    The Besov indices come from :func:`besov_indices`; see the module docstring
    for the bound convention. ``alpha_stable`` is the driving law's stability
    index (``2.0`` for Gaussian fGn), not the fractional-derivative order.
    """
    s, p, q = besov_indices(H_hurst, alpha_stable, margin)

    Y_hat = model(U_drive)
    mse_loss = jnp.mean((Y_target - Y_hat) ** 2)
    besov_loss = littlewood_paley_penalty(Y_hat, masks, s, p, q)
    return mse_loss + lambda_reg * besov_loss


@eqx.filter_jit
def train_step(
    model, optimizer, opt_state, U_drive, Y_target, masks, H_hurst, lambda_reg,
    alpha_stable=2.0,
):
    """One optimisation step updating *only* the readout. Returns ``(model, opt_state, loss)``.

    The model is partitioned into a differentiable part (``readout.W_out``) and a
    frozen static part; gradients and optimiser updates touch the former only, so
    the reservoir and fractional kernels remain exactly fixed.
    """
    diff, static = eqx.partition(model, readout_filter_spec(model))

    def loss_fn(diff):
        merged = eqx.combine(diff, static)
        return compute_loss(merged, U_drive, Y_target, masks, H_hurst, lambda_reg, alpha_stable)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(diff)
    updates, opt_state = optimizer.update(grads, opt_state, diff)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
