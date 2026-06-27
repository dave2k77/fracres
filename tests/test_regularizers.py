import jax.numpy as jnp

from fracres import littlewood_paley_penalty, make_dyadic_masks


def test_dyadic_masks_partition_frequencies():
    T = 64
    masks = make_dyadic_masks(T)
    # No frequency bin is claimed by more than one dyadic band.
    assert jnp.all(masks.sum(axis=0) <= 1.0)
    assert masks.shape[1] == T


def test_penalty_is_finite_nonnegative():
    T, F = 128, 3
    Y = jnp.sin(jnp.linspace(0, 20, T))[:, None] * jnp.ones((1, F))
    masks = make_dyadic_masks(T)
    val = littlewood_paley_penalty(Y, masks, s=0.6, p=0.7, q=0.7)
    assert jnp.isfinite(val)
    assert val >= 0.0
