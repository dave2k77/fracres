"""Stability and spectral control for the fractional reservoir (KB v2 §3.4).

For the linearised autonomous system ``D^alpha x = A x`` with
``A = W_res - Lambda`` -- the Jacobian of ``D^alpha x = -Lambda x + tanh(W_res x)``
at the origin (``tanh'(0) = 1``) -- the fractional stability criterion is
Matignon's theorem (1996):

    asymptotically stable  <=>  |arg(lambda_i(A))| > alpha * pi / 2   for all i.

The *unstable* set is the cone of half-angle ``alpha*pi/2`` about the **positive**
real axis. As ``alpha -> 1`` it fills the right half-plane (classical
``Re(lambda) < 0``); for ``alpha < 1`` it shrinks, so the stable region is
*larger* -- a fractional reservoir can sit near criticality with a "hotter"
connectome than a classical ESN. Verified thresholds: ``alpha=0.5 -> 45 deg``,
``0.7 -> 63 deg``, ``0.9 -> 81 deg``, ``1.0 -> 90 deg``.

This module provides the diagnostic :func:`matignon_diagnostics` and two ways to
set the operating point by scaling ``W_res``: classical
:func:`set_spectral_radius` and the Matignon-aware :func:`set_matignon_margin`
(the fractional "edge of chaos" -- a target distance from the wedge boundary).

Key identity used by the control: because ``I`` shares every eigenvector of
``W_res``, the eigenvalues of ``s W_res - decay I`` are exactly ``s mu_i - decay``
where ``mu_i = eigvals(W_res)``. So ``min_i |arg(.)|`` as a function of the scale
``s`` is evaluated without re-eigendecomposing, and it is monotone in ``s``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import equinox as eqx


def _reservoir_of(obj):
    """Accept a reservoir or a model that has a ``.reservoir`` attribute."""
    return obj.reservoir if hasattr(obj, "reservoir") else obj


def _with_W_res(obj, new_W_res):
    """Return ``obj`` (reservoir or model) with ``W_res`` replaced."""
    if hasattr(obj, "reservoir"):
        new_res = eqx.tree_at(lambda r: r.W_res, obj.reservoir, new_W_res)
        return eqx.tree_at(lambda m: m.reservoir, obj, new_res)
    return eqx.tree_at(lambda r: r.W_res, obj, new_W_res)


def system_matrix(obj) -> jnp.ndarray:
    """Linearised system matrix ``A = W_res - Lambda`` (``Lambda = decay * I``)."""
    res = _reservoir_of(obj)
    n = res.W_res.shape[0]
    return res.W_res - res.decay * jnp.eye(n, dtype=res.W_res.dtype)


class MatignonDiagnostics(NamedTuple):
    """Stability summary for a (fractional) reservoir. Angles in radians."""

    alpha: float
    threshold: float  # alpha * pi / 2
    min_arg: float  # min_i |arg(lambda_i(A))|
    margin: float  # min_arg - threshold  (> 0 => stable)
    stable: bool
    spectral_radius: float  # rho(W_res)   (ESP diagnostic)
    max_singular_value: float  # sigma_max(W_res)  (sufficient-ESP diagnostic)

    @property
    def threshold_deg(self) -> float:
        return self.threshold * 180.0 / 3.141592653589793

    @property
    def critical_alpha(self) -> float:
        """Largest ``alpha`` for which this ``A`` is Matignon-stable (``2 min_arg/pi``)."""
        return 2.0 * self.min_arg / 3.141592653589793


def matignon_diagnostics(obj) -> MatignonDiagnostics:
    """Compute Matignon stability + ESP diagnostics for a reservoir/model."""
    res = _reservoir_of(obj)
    A = system_matrix(res)
    min_arg = float(jnp.min(jnp.abs(jnp.angle(jnp.linalg.eigvals(A)))))
    threshold = float(res.alpha) * 3.141592653589793 / 2.0
    W = res.W_res
    rho = float(jnp.max(jnp.abs(jnp.linalg.eigvals(W))))
    smax = float(jnp.max(jnp.linalg.svd(W, compute_uv=False)))
    return MatignonDiagnostics(
        alpha=float(res.alpha),
        threshold=threshold,
        min_arg=min_arg,
        margin=min_arg - threshold,
        stable=min_arg > threshold,
        spectral_radius=rho,
        max_singular_value=smax,
    )


def set_spectral_radius(obj, target_rho: float):
    """Rescale ``W_res`` to a target spectral radius ``rho(W_res) = target_rho``."""
    res = _reservoir_of(obj)
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(res.W_res)))
    return _with_W_res(obj, res.W_res * (target_rho / rho))


def matignon_edge_scale(obj, *, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Scale ``s_edge`` putting the system exactly on the Matignon stability boundary.

    The largest ``s`` for which ``s W_res - decay I`` keeps every eigenvalue out of
    the unstable cone (``min_i |arg(s mu_i - decay)| >= alpha*pi/2``,
    ``mu_i = eigvals(W_res)``).

    Found by bisection on the *stability* predicate, which is monotone: each
    ``|arg(s mu_i - decay)|`` decreases as ``s`` grows (the eigenvalue ray fans out
    from ``-decay`` toward ``arg(mu_i)``), so once a mode enters the cone it stays
    in -- there is a single stable→unstable transition. Real eigenvalues (a jump
    from ``arg = pi`` to ``arg = 0`` at ``s = decay/a``) are handled correctly.

    Returns ``inf`` if no eigenvalue can be driven into the cone by scaling (the
    connectome is unconditionally stable). Requires ``decay > 0``.
    """
    res = _reservoir_of(obj)
    decay = float(res.decay)
    if decay <= 0.0:
        raise ValueError("Matignon control requires decay > 0 (a positive leak).")
    threshold = float(res.alpha) * 3.141592653589793 / 2.0
    mu = jnp.linalg.eigvals(res.W_res)

    def stable(s: float) -> bool:
        return float(jnp.min(jnp.abs(jnp.angle(s * mu - decay)))) >= threshold

    # s -> 0 is stable (all eigenvalues -> -decay, arg = pi). Grow until unstable.
    hi = 1.0
    while stable(hi) and hi < 1e8:
        hi *= 2.0
    if stable(hi):
        return float("inf")  # never enters the cone
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1.0):
            break
    return lo  # largest stable scale


def set_edge_of_chaos(obj, safety_factor: float = 0.95):
    """Scale ``W_res`` to ``safety_factor * s_edge`` (the fractional edge of chaos).

    Rescales the connectome so the least-stable mode sits at a controlled distance
    from the Matignon wedge: ``safety_factor < 1`` is stable (inside the wedge),
    ``= 1`` is exactly on the boundary (marginal), ``> 1`` is deliberately
    unstable. Unlike pushing ``rho(W_res) -> 1``, this targets eigenvalue
    *arguments* against the fractional threshold ``alpha*pi/2`` (KB v2 §3.4), so it
    exploits the larger stable region available for ``alpha < 1``.

    Raises
    ------
    ValueError
        If ``decay <= 0`` or the connectome is unconditionally stable under scaling
        (``s_edge`` is infinite -- no eigenvalue argument is inside the cone).
    """
    res = _reservoir_of(obj)
    s_edge = matignon_edge_scale(res)
    if not jnp.isfinite(s_edge):
        raise ValueError(
            "no Matignon edge reachable by scaling: every eigenvalue argument is "
            "already outside the unstable cone. Increase the connectome scale first."
        )
    return _with_W_res(obj, res.W_res * (safety_factor * s_edge))
