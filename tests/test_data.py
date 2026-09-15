"""Tests for fracres.data (BIDS EEG ingestion).

Unit tests build a tiny synthetic BIDS tree in a tmp dir and synthesise an EDF
recording with mne, so they need the optional ``data`` extra but no downloaded
dataset. The ds002778 integration test at the bottom runs only when the local
mirror exists (``data/ds002778`` next to the repo); elsewhere it skips, so CI
stays data-free -- the same opt-in pattern as the hpfracc cross-reference.
"""
from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne", reason="optional 'data' extra not installed")

from fracres.data import (
    Participant,
    dataset_root,
    list_sessions,
    list_subjects,
    load_participants,
    load_recording,
    load_subject,
    recording_path,
)

CHANNELS = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "EXG1", "EXG2"]
SFREQ = 512.0


def _make_bids_tree(tmp_path):
    """Synthetic 2-subject BIDS tree: one hc, one pd, with participants.tsv."""
    root = tmp_path / "ds000000"
    for sub in ("sub-hc1", "sub-pd2"):
        eeg_dir = root / sub / "eeg"
        eeg_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((len(CHANNELS), 5120)) * 1e-5
        info = mne.create_info(CHANNELS, SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        mne.export.export_raw(eeg_dir / f"{sub}_task-rest_eeg.edf", raw, verbose=False)
    (root / "participants.tsv").write_text(
        "participant_id\tGroup\tage\tsex\n"
        "sub-hc1\thc\t65\tM\n"
        "sub-pd2\tpd\t71\tF\n",
        encoding="utf-8",
    )
    return root


def test_list_subjects_and_group_filter(tmp_path):
    root = _make_bids_tree(tmp_path)
    assert list_subjects(root) == ["sub-hc1", "sub-pd2"]
    assert list_subjects(root, group="pd") == ["sub-pd2"]
    assert list_subjects(root, group="hc") == ["sub-hc1"]


def test_load_participants_parses_group_and_metadata(tmp_path):
    root = _make_bids_tree(tmp_path)
    participants = load_participants(root)
    assert participants == [
        Participant("sub-hc1", "hc", {"age": "65", "sex": "M"}),
        Participant("sub-pd2", "pd", {"age": "71", "sex": "F"}),
    ]
    assert participants[1].is_pd
    assert not participants[0].is_pd


def test_recording_path_resolution_and_missing(tmp_path):
    root = _make_bids_tree(tmp_path)
    path = recording_path(root, "sub-pd2", task="rest")
    assert path.name == "sub-pd2_task-rest_eeg.edf"
    with pytest.raises(FileNotFoundError):
        recording_path(root, "sub-pd2", task="walk")


def test_load_recording_drops_exg_and_attaches_geometry(tmp_path):
    root = _make_bids_tree(tmp_path)
    rec = load_recording(recording_path(root, "sub-hc1"))
    # EXG mastoid/EOG channels excluded; scalp channels kept, order preserved.
    assert rec.ch_names == ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
    assert rec.data.shape == (6, 5120)
    assert rec.sfreq == SFREQ
    assert rec.subject == "sub-hc1"
    assert rec.group == "hc"
    assert rec.duration == pytest.approx(10.0)
    # Montage positions are finite 3-vectors on a head-sized sphere.
    assert rec.positions.shape == (6, 3)
    radii = np.linalg.norm(rec.positions, axis=1)
    assert np.all(radii > 0.05) and np.all(radii < 0.15)


def test_load_recording_resample(tmp_path):
    root = _make_bids_tree(tmp_path)
    rec = load_recording(recording_path(root, "sub-pd2"), resample=256.0)
    assert rec.sfreq == pytest.approx(256.0)
    assert rec.data.shape[1] == 2560


def test_load_subject_end_to_end_and_channel_access(tmp_path):
    root = _make_bids_tree(tmp_path)
    rec = load_subject(root, "sub-hc1")
    assert rec.channel("C3").shape == (5120,)
    data_jax, pos_jax = rec.to_jax()
    assert data_jax.shape == rec.data.shape
    assert pos_jax.shape == rec.positions.shape


def test_session_layout_resolution(tmp_path):
    """BIDS session level: single session auto-resolves, multiple requires one."""
    root = tmp_path / "ds000001"
    rng = np.random.default_rng(1)
    info = mne.create_info(CHANNELS, SFREQ, ch_types="eeg")
    # Single-session control subject.
    eeg_dir = root / "sub-hc1" / "ses-hc" / "eeg"
    eeg_dir.mkdir(parents=True)
    data = rng.standard_normal((len(CHANNELS), 512)) * 1e-5
    raw = mne.io.RawArray(data, info, verbose=False)
    mne.export.export_raw(
        eeg_dir / "sub-hc1_ses-hc_task-rest_eeg.edf", raw, verbose=False
    )
    # Two-session patient (medication off/on).
    for ses in ("off", "on"):
        eeg_dir = root / "sub-pd2" / f"ses-{ses}" / "eeg"
        eeg_dir.mkdir(parents=True)
        mne.export.export_raw(
            eeg_dir / f"sub-pd2_ses-{ses}_task-rest_eeg.edf", raw, verbose=False
        )

    assert list_sessions(root, "sub-hc1") == ["hc"]
    assert list_sessions(root, "sub-pd2") == ["off", "on"]

    rec = load_subject(root, "sub-hc1")  # single session: no session= needed
    assert rec.session == "hc"
    with pytest.raises(ValueError, match="multiple sessions"):
        recording_path(root, "sub-pd2")
    rec = load_subject(root, "sub-pd2", session="off")
    assert rec.session == "off"
    with pytest.raises(FileNotFoundError):
        recording_path(root, "sub-pd2", session="placebo")


# --- Integration: real ds002778 mirror (opt-in, skipped without the data) ----

REAL_ROOT = dataset_root()


@pytest.mark.skipif(
    not (REAL_ROOT / "participants.tsv").exists(),
    reason="local ds002778 mirror not present (data/ds002778)",
)
def test_ds002778_integration():
    participants = load_participants(REAL_ROOT)
    groups = {p.group for p in participants}
    assert groups == {"pd", "hc"}
    assert len(participants) >= 30

    # Healthy control: single ses-hc auto-resolves.
    hc = next(p for p in participants if not p.is_pd)
    rec = load_subject(REAL_ROOT, hc.subject, resample=256.0)
    assert rec.session == "hc"
    # PD subject: medication OFF/ON sessions both load.
    pd = next(p for p in participants if p.is_pd)
    assert list_sessions(REAL_ROOT, pd.subject) == ["off", "on"]
    for ses in ("off", "on"):
        rec = load_subject(REAL_ROOT, pd.subject, session=ses, resample=256.0)
        assert rec.session == ses
        assert rec.n_channels == 32  # BioSemi 32 scalp channels
        assert rec.sfreq == pytest.approx(256.0)
        assert rec.duration > 60.0  # resting-state runs are minutes long
        assert np.isfinite(rec.data).all()
        assert rec.data.dtype == np.float64
        # All 64 scalp channels must get montage positions (no NaNs).
        assert np.isfinite(rec.positions).all()
