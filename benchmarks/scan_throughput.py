"""Scan-throughput benchmark for the fractional reservoir.

``PhantomBrain.simulate`` runs a ``jax.lax.scan`` over ``T`` steps; each step costs
``O(N^2)`` for the ``W_res @ x`` connectome product plus ``O(L * N)`` for the
fractional-memory einsum and history roll (``N = res_size``, ``L = history_length``).
This script measures wall-clock throughput as ``N`` and ``L`` vary and fits the
empirical scaling exponent of each sweep.

JAX runs asynchronously, so every timed call is JIT-compiled, warmed up, and
forced to completion with ``block_until_ready`` before the median of several reps
is taken. Numbers are machine-dependent -- use them for relative scaling, not as
absolute targets.

Run:  python benchmarks/scan_throughput.py
"""
import math
import time

import jax

from fracres import GLKernel, PhantomBrain, generate_fbm_increments

T = 500
REPS = 5
KEY = jax.random.PRNGKey(0)


def median_time(model, U) -> float:
    """Median wall-clock time (seconds) of ``model.simulate(U)`` over ``REPS`` runs."""
    run = jax.jit(lambda u: model.simulate(u))
    jax.block_until_ready(run(U))  # compile + warm up
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        jax.block_until_ready(run(U))
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def fit_exponent(xs, ts) -> float:
    """Least-squares slope of ``log(time)`` vs ``log(x)`` -- the scaling exponent."""
    lx = [math.log(x) for x in xs]
    lt = [math.log(t) for t in ts]
    mx, mt = sum(lx) / len(lx), sum(lt) / len(lt)
    cov = sum((a - mx) * (b - mt) for a, b in zip(lx, lt, strict=True))
    var = sum((a - mx) ** 2 for a in lx)
    return cov / var


def _sweep(label, configs, make_model):
    drive = generate_fbm_increments(T, H=0.7, key=KEY)[:, None]
    print(f"\n{label}")
    print(f"  {'param':>6} | {'time/run':>10} {'us/step':>9} {'steps/s':>12}")
    xs, ts = [], []
    for x in configs:
        t = median_time(make_model(x), drive)
        xs.append(x)
        ts.append(t)
        print(f"  {x:>6} | {t * 1e3:>8.2f}ms {t / T * 1e6:>8.2f} {T / t:>12,.0f}")
    print(f"  empirical scaling exponent: {fit_exponent(xs, ts):.2f}")


def main():
    print(f"Scan throughput  (T={T} steps, median of {REPS} reps, "
          f"JIT-compiled, CPU)")

    _sweep(
        "res_size N sweep (L=64):  per-step cost ~ O(N^2)",
        [64, 128, 256, 512, 1024],
        lambda n: PhantomBrain(1, n, 1, GLKernel(0.8, 64), key=KEY),
    )
    _sweep(
        "history_length L sweep (N=128):  per-step cost ~ O(L*N)",
        [16, 32, 64, 128, 256],
        lambda length: PhantomBrain(1, 128, 1, GLKernel(0.8, length), key=KEY),
    )
    print("\n(N grows ~quadratically once it dominates the scan overhead; L is"
          " linear and stays cheap until L*N approaches N^2.)")


if __name__ == "__main__":
    main()
