"""Performance-tier scaling guards for the reservoir scan.

Deselected by default (``-m "not performance"`` in pyproject); run explicitly with
``python -m pytest -m performance``. Timing is machine-dependent and must not gate
the unit suite.

``PhantomBrain.simulate`` costs ``O(N^2)`` per step for the connectome product and
``O(L * N)`` for the fractional memory. Rather than pin a brittle wall-clock
budget, these fit the empirical log-log scaling exponent across sizes and assert
it stays well below the next polynomial order -- catching accidental algorithmic
regressions (a dense ``O(N^3)`` path, or memory going ``O(L^2)``).
"""
from __future__ import annotations

import math
import time

import pytest

jax = pytest.importorskip("jax")

from fracres import GLKernel, PhantomBrain, generate_fbm_increments

pytestmark = pytest.mark.performance

T = 300
REPS = 5
KEY = jax.random.PRNGKey(0)


def _median_time(model, U) -> float:
    run = jax.jit(lambda u: model.simulate(u))
    jax.block_until_ready(run(U))  # compile + warm up
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        jax.block_until_ready(run(U))
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def _scaling_exponent(sizes, make_model) -> float:
    drive = generate_fbm_increments(T, H=0.7, key=KEY)[:, None]
    logs_x, logs_t = [], []
    for x in sizes:
        logs_x.append(math.log(x))
        logs_t.append(math.log(_median_time(make_model(x), drive)))
    mx = sum(logs_x) / len(logs_x)
    mt = sum(logs_t) / len(logs_t)
    cov = sum((a - mx) * (b - mt) for a, b in zip(logs_x, logs_t, strict=True))
    var = sum((a - mx) ** 2 for a in logs_x)
    return cov / var


def test_scan_is_subcubic_in_res_size():
    exponent = _scaling_exponent(
        [64, 128, 256, 512],
        lambda n: PhantomBrain(1, n, 1, GLKernel(0.8, 50), key=KEY),
    )
    assert exponent < 2.5, f"scan scaled as ~O(N^{exponent:.2f}) in res_size"


def test_scan_is_subquadratic_in_history_length():
    exponent = _scaling_exponent(
        [25, 50, 100, 200],
        lambda length: PhantomBrain(1, 96, 1, GLKernel(0.8, length), key=KEY),
    )
    assert exponent < 1.5, f"scan scaled as ~O(L^{exponent:.2f}) in history_length"
