"""Tests for fracres.inverse (mechanism inference / inverse problem).

The recovery test is deterministic by construction: the synthetic "data" is a
phantom output generated with the *same* seed scheme the grid search uses
(common random numbers), so the true grid point must score objective ~ 0 and
rank first. No mne or downloaded data needed; the ds002778 integration test is
opt-in, as usual.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from fracres.data import EEGRecording, dataset_root, load_subject
from fracres.drivers import generate_fbm, generate_fbm_increments
from fracres.inverse import grid_search, metric_distance
from fracres.kernels import GLKernel
from fracres.metrics import SignalMetrics
from fracres.models import PhantomBrain

TRUE = {"alpha": 0.7, "hurst": 0.7, "decay": 2.0}
RES_SIZE, HISTORY = 50, 50  # small, for test speed


def _phantom_output(alpha, hurst, decay, t_steps, drive_kind="fgn",
                    seed_idx=0, base_seed=0):
    """Reproduce exactly what grid_search's evaluator simulates at one point."""
    key = jax.random.PRNGKey(base_seed + 104729 * seed_idx)
    k_model, k_drive = jax.random.split(key)
    kernel = GLKernel(alpha=alpha, history_length=HISTORY)
    model = PhantomBrain(
        1, RES_SIZE, 1, kernel,
        key=k_model, spectral_scale=0.95, step_size=0.4, decay=decay,
    )
    drive = (
        generate_fbm(t_steps, H=hurst, key=k_drive)
        if drive_kind == "fbm"
        else generate_fbm_increments(t_steps, H=hurst, key=k_drive)
    )
    drive = drive[:, None]
    _, y_hat = model.simulate(drive)
    return np.asarray(y_hat[:, 0])


def _recording_from(trace, n=4000):
    filler = np.random.default_rng(0).standard_normal(n)
    return EEGRecording(
        data=np.vstack([trace[:n], filler]),
        sfreq=128.0,
        ch_names=["Cz", "Oz"],
        positions=np.zeros((2, 3)),
        subject="sub-syn0",
        group="hc",
    )


def test_metric_distance_zero_and_weighted():
    m = SignalMetrics(hurst_dfa=1.0, spectral_beta=1.5, hurst_spectral=1.25)
    assert metric_distance(m, m) == 0.0
    other = SignalMetrics(hurst_dfa=1.1, spectral_beta=1.5, hurst_spectral=1.25)
    assert metric_distance(m, other) == pytest.approx(0.1)
    # Weights multiply the squared deviations: sqrt(4 * 0.1^2) = 0.2.
    assert metric_distance(m, other, weights=(4.0, 1.0)) == pytest.approx(0.2)


def test_grid_search_recovers_true_mechanism():
    n = 4000
    trace = _phantom_output(**TRUE, t_steps=n)
    rec = _recording_from(trace, n)
    result = grid_search(
        rec,
        channel="Cz",
        alphas=[0.5, 0.7, 0.9],
        hursts=[0.5, 0.7, 0.9],
        decays=[TRUE["decay"]],
        reference=None,
        res_size=RES_SIZE,
        history_length=HISTORY,
        n_seeds=1,
        base_seed=0,
    )
    best = result.best
    # Common random numbers: the true point replays the target draw exactly.
    assert (best.alpha, best.hurst, best.decay) == (
        TRUE["alpha"], TRUE["hurst"], TRUE["decay"],
    )
    assert best.objective == pytest.approx(0.0, abs=1e-6)
    # Identifiability: the closest competitor keeps the true drive H and moves
    # only alpha (the softer direction at finite T); every wrong-H point is
    # far worse, so H is identified sharply.
    runner_up = result.points[1]
    assert runner_up.hurst == TRUE["hurst"] and runner_up.alpha != TRUE["alpha"]
    wrong_h = [p for p in result.points if p.hurst != TRUE["hurst"]]
    assert min(p.objective for p in wrong_h) > 0.3
    assert "0.70" in result.leaderboard()


def test_grid_search_recovers_fbm_driven_mechanism():
    """Same exact-replay recovery, but with the integrated (fBm) drive."""
    n = 4000
    trace = _phantom_output(**TRUE, t_steps=n, drive_kind="fbm")
    rec = _recording_from(trace, n)
    result = grid_search(
        rec,
        channel="Cz",
        alphas=[0.5, 0.7, 0.9],
        hursts=[0.5, 0.7, 0.9],
        decays=[TRUE["decay"]],
        drive_kind="fbm",
        reference=None,
        res_size=RES_SIZE,
        history_length=HISTORY,
        n_seeds=1,
        base_seed=0,
    )
    assert (result.best.alpha, result.best.hurst, result.best.decay) == (
        TRUE["alpha"], TRUE["hurst"], TRUE["decay"],
    )
    assert result.best.objective == pytest.approx(0.0, abs=1e-6)


def test_grid_search_rejects_unknown_drive_kind():
    rec = _recording_from(np.random.default_rng(1).standard_normal(4000))
    with pytest.raises(ValueError, match="drive_kind"):
        grid_search(rec, "Cz", [0.7], [0.7], drive_kind="pink", reference=None,
                    res_size=RES_SIZE, history_length=HISTORY, n_seeds=1)


# --- Integration: real ds002778 mirror (opt-in) -------------------------------

REAL_ROOT = dataset_root()


@pytest.mark.skipif(
    not (REAL_ROOT / "participants.tsv").exists(),
    reason="local ds002778 mirror not present (data/ds002778)",
)
def test_grid_search_real_recording_smoke():
    rec = load_subject(REAL_ROOT, "sub-pd3", session="off", resample=64.0)
    result = grid_search(
        rec,
        channel="Cz",
        alphas=[0.6, 0.9],
        hursts=[0.5, 0.7],
        decays=[2.0],
        res_size=100,
        n_seeds=1,
    )
    assert len(result.points) == 4
    best = result.best
    assert 0.0 < best.alpha <= 1.0 and 0.0 < best.hurst < 1.0
    assert np.isfinite(best.objective)
    assert np.isfinite(result.data_metrics.hurst_dfa)
