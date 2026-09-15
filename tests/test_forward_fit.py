"""Tests for fracres.forward_fit (real-data forward-fit workflow).

Unit tests run on a synthetic :class:`EEGRecording` -- channel B is a lagged,
smoothed copy of the drive channel A (expressible by the reservoir's memory,
so reconstructible), channel C is independent noise (not reconstructible).
No mne or downloaded data needed. The ds002778 integration test at the bottom
is opt-in, like the data-module one.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from fracres.data import EEGRecording, dataset_root, load_subject
from fracres.forward_fit import channel_matrix, forward_fit
from fracres.kernels import GLKernel
from fracres.models import PhantomBrain


def _synthetic_recording(n=6000, seed=0):
    """A drives; B = lagged/smoothed A + mild noise; C = independent noise."""
    from fracres.drivers import generate_fbm_increments

    rng = np.random.default_rng(seed)
    a = np.asarray(
        generate_fbm_increments(n, H=0.7, key=jax.random.PRNGKey(seed))
    )
    kernel = np.ones(25) / 25  # 25-sample boxcar: memory the reservoir holds
    b = np.convolve(a, kernel, mode="same") + 0.1 * rng.standard_normal(n)
    c = rng.standard_normal(n)
    data = np.vstack([a, b, c])
    return EEGRecording(
        data=data,
        sfreq=256.0,
        ch_names=["A", "B", "C"],
        positions=np.zeros((3, 3)),
        subject="sub-syn0",
        group="hc",
    )


def _model(seed=0):
    kernel = GLKernel(alpha=0.8, history_length=100)
    return PhantomBrain(
        1, 200, 2, kernel, key=jax.random.PRNGKey(seed), step_size=0.4, decay=2.0
    )


def test_channel_matrix_average_reference_and_zscore():
    rec = _synthetic_recording(n=500)
    x, names = channel_matrix(rec, reference="average", zscore=True)
    assert names == ["A", "B", "C"]
    assert x.shape == (500, 3)
    # z-scored per channel.
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-10)
    # Common average reference (before z-scoring rescales channels):
    # channels sum to zero at every time step.
    x_ref, _ = channel_matrix(rec, reference="average", zscore=False)
    assert np.allclose(x_ref.sum(axis=1), 0.0, atol=1e-10)
    x_raw, _ = channel_matrix(rec, reference=None, zscore=False)
    assert np.allclose(x_raw, rec.data.T)


def test_forward_fit_reconstructs_expressible_channel():
    rec = _synthetic_recording()
    result = forward_fit(
        rec,
        drive_channels=["A"],
        target_channels=["B", "C"],
        model=_model(),
        reference=None,
    )
    assert result.correlation.shape == (2,)
    corr_b, corr_c = result.correlation
    # The lagged/smoothed channel is reconstructible; the independent one is not.
    assert corr_b > 0.4
    assert corr_b > corr_c + 0.3
    assert result.nmse[0] < result.nmse[1]
    # Metrics exist per target and are finite.
    for m in result.data_metrics + result.model_metrics:
        assert np.isfinite(m.hurst_dfa) and np.isfinite(m.spectral_beta)
    # Held-out arrays only cover the tail.
    assert result.y_true.shape[0] == rec.data.shape[1] - result.split
    assert "B" in result.metrics_table() and "C" in result.metrics_table()


def test_forward_fit_validates_split():
    rec = _synthetic_recording(n=1000)
    with pytest.raises(ValueError, match="washout < split"):
        forward_fit(
            rec, ["A"], ["B"], _model(), washout=900, train_frac=0.7, reference=None
        )


# --- Integration: real ds002778 mirror (opt-in) -------------------------------

REAL_ROOT = dataset_root()


@pytest.mark.skipif(
    not (REAL_ROOT / "participants.tsv").exists(),
    reason="local ds002778 mirror not present (data/ds002778)",
)
def test_forward_fit_real_pd_recording():
    rec = load_subject(REAL_ROOT, "sub-pd3", session="off", resample=64.0)
    kernel = GLKernel(alpha=0.8, history_length=100)
    model = PhantomBrain(
        1, 300, 3, kernel, key=jax.random.PRNGKey(0), step_size=0.4, decay=2.0
    )
    result = forward_fit(
        rec, drive_channels=["Fz"], target_channels=["Cz", "Pz", "O1"], model=model
    )
    assert np.all(np.abs(result.correlation) <= 1.0)
    assert np.isfinite(result.nmse).all()
    for m in result.data_metrics + result.model_metrics:
        assert np.isfinite(m.hurst_dfa) and np.isfinite(m.spectral_beta)
