"""Real-data ingestion: BIDS-organised EEG recordings.

This module turns fracres from a phantom generator into an instrument: it loads
real scalp EEG (starting with the UC San Diego resting-state Parkinson's
dataset, OpenNeuro `ds002778`_) into plain arrays that the reservoir, drive, and
metrics machinery can consume.

The dataset is BIDS-organised BioSemi 32-channel BDF at 512 Hz (32 scalp
channels + 8 EXG mastoid/ocular, ~200 s unreferenced CMS/DRL recordings;
re-referencing and filtering are the analyst's choice and left to the caller):

.. code-block:: text

    ds002778/
      participants.tsv          # group (pd/hc), age, sex, UPDRS, ...
      sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf
      sub-pd3/ses-off/eeg/sub-pd3_ses-off_task-rest_eeg.bdf   # medication OFF
      sub-pd3/ses-on/eeg/sub-pd3_ses-on_task-rest_eeg.bdf     # medication ON
      ...

`mne`_ is an *optional* dependency (``pip install fracres[data]``); it is
imported lazily so the core package never requires it.

.. _ds002778: https://openneuro.org/datasets/ds002778
.. _mne: https://mne.tools

Data-use note: ds002778 is CC0, but the curators ask to be emailed
(arockhil@uoregon.edu) before journal submission, and the associated papers
(Jackson et al. 2019, eNeuro; Swann et al. 2015, Ann. Neurol.; George et al.
2013, NeuroImage Clin.) cited. They explicitly advise *against* training
PD-vs-HC classifiers on it (n too small for honest statistics) -- our use is
forward modelling and LRD/criticality statistics, which is in scope.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# BioSemi 32 scalp montage shipped with mne; the ds002778 BDFs store no sensor
# positions, so geometry comes from the standard 10/5-equivalent layout.
DEFAULT_MONTAGE = "biosemi32"


def _mne():
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "fracres.data requires the optional 'data' extra: "
            "pip install 'fracres[data]' (pulls in mne)."
        ) from exc
    return mne


@dataclass
class Participant:
    """One row of a BIDS ``participants.tsv``."""

    subject: str
    group: str  # "pd" or "hc"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_pd(self) -> bool:
        return self.group.lower() == "pd"


@dataclass
class EEGRecording:
    """A loaded resting-state EEG recording.

    Attributes
    ----------
    data : array, shape ``(n_channels, n_times)``
        EEG in volts (mne's native unit), scalp channels only.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel labels, ordered as the rows of ``data``.
    positions : array, shape ``(n_channels, 3)``
        Sensor positions (metres) from the standard montage.
    subject : str
        BIDS subject label, e.g. ``"sub-pd3"``.
    group : str
        ``"pd"`` or ``"hc"`` (from the subject label).
    session : str or None
        Session label without the ``ses-`` prefix (e.g. ``"off"``/``"on"``/
        ``"hc"``), ``None`` when the dataset has no session level.
    """

    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    positions: np.ndarray
    subject: str
    group: str
    session: str | None = None

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return self.data.shape[1] / self.sfreq

    def channel(self, name: str) -> np.ndarray:
        """Return the trace of a single channel by label."""
        return self.data[self.ch_names.index(name)]

    def to_jax(self):
        """Return ``(data, positions)`` as JAX arrays for the reservoir stack."""
        import jax.numpy as jnp

        return jnp.asarray(self.data), jnp.asarray(self.positions)


def dataset_root() -> Path:
    """Default location of the local ds002778 mirror (``<repo>/data/ds002778``)."""
    return Path(__file__).resolve().parents[2] / "data" / "ds002778"


def list_subjects(root: str | Path, group: str | None = None) -> list[str]:
    """List BIDS subject labels under *root*, optionally filtered by group.

    Parameters
    ----------
    root : path
        Dataset root containing ``sub-*/`` directories.
    group : {"pd", "hc"}, optional
        Keep only Parkinson's (``"pd"``) or healthy-control (``"hc"``) subjects.
    """
    root = Path(root)
    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    if group is not None:
        group = group.lower()
        subjects = [s for s in subjects if s.removeprefix("sub-").startswith(group)]
    return subjects


def load_participants(root: str | Path) -> list[Participant]:
    """Parse ``participants.tsv`` into :class:`Participant` records.

    The group is taken from the ``Group`` column when present, else inferred
    from the subject label prefix (``pd``/``hc``).
    """
    path = Path(root) / "participants.tsv"
    participants = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            subject = row["participant_id"]
            group = row.get("Group") or "".join(
                c for c in subject.removeprefix("sub-") if c.isalpha()
            )
            metadata = {
                k: v for k, v in row.items() if k not in ("participant_id", "Group")
            }
            participants.append(
                Participant(subject=subject, group=group, metadata=metadata)
            )
    return participants


def list_sessions(root: str | Path, subject: str) -> list[str]:
    """Session labels (without ``ses-``) for one subject; empty if none."""
    sub_dir = Path(root) / subject
    return sorted(
        p.name.removeprefix("ses-") for p in sub_dir.glob("ses-*") if p.is_dir()
    )


def recording_path(
    root: str | Path,
    subject: str,
    task: str = "rest",
    session: str | None = None,
) -> Path:
    """Resolve the BIDS EEG file for one subject/session/task.

    Handles both layouts: ``<root>/<subject>/eeg/`` (no session level) and
    ``<root>/<subject>/ses-<label>/eeg/`` (BIDS sessions). When the subject has
    a single session it is used automatically; with several (e.g. ds002778 PD
    subjects, ``ses-off``/``ses-on`` medication) *session* is required.

    Parameters
    ----------
    session : str, optional
        Session label with or without the ``ses-`` prefix (``"off"`` or
        ``"ses-off"``).
    """
    root = Path(root)
    sub_dir = root / subject
    ses_dirs = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir())
    if ses_dirs:
        if session is None:
            if len(ses_dirs) > 1:
                labels = [p.name.removeprefix("ses-") for p in ses_dirs]
                raise ValueError(
                    f"{subject} has multiple sessions {labels}; pass session=..."
                )
        else:
            wanted = "ses-" + session.removeprefix("ses-")
            ses_dirs = [p for p in ses_dirs if p.name == wanted]
            if not ses_dirs:
                raise FileNotFoundError(f"No session {wanted!r} for {subject}")
        eeg_dir = ses_dirs[0] / "eeg"
        stem = f"{subject}_{ses_dirs[0].name}_task-{task}_eeg"
    else:
        eeg_dir = sub_dir / "eeg"
        stem = f"{subject}_task-{task}_eeg"
    matches = sorted(
        p for ext in ("bdf", "edf", "set", "fif") for p in eeg_dir.glob(f"{stem}.{ext}")
    )
    if not matches:
        raise FileNotFoundError(
            f"No EEG file for {subject} session={session!r} "
            f"task={task!r} under {eeg_dir}"
        )
    return matches[0]


def load_recording(
    path: str | Path,
    montage: str = DEFAULT_MONTAGE,
    resample: float | None = None,
    subject: str | None = None,
) -> EEGRecording:
    """Load one BDF/EDF EEG file into an :class:`EEGRecording`.

    Keeps scalp EEG channels only (drops BioSemi EXG mastoid/EOG and the
    status channel), attaches standard-montage sensor positions, and optionally
    resamples.

    Parameters
    ----------
    path : path
        The ``*_eeg.bdf`` (or ``.edf``/``.set``) file.
    montage : str
        mne standard montage name for sensor positions (BioSemi 64 default).
    resample : float, optional
        Target sampling frequency in Hz (e.g. 256 to halve memory/compute).
        ``None`` keeps the native rate.
    subject : str, optional
        BIDS subject label; inferred from the filename when omitted.
    """
    mne = _mne()
    path = Path(path)
    readers = {
        ".bdf": mne.io.read_raw_bdf,
        ".edf": mne.io.read_raw_edf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    ext = path.suffix.lower()
    if ext not in readers:
        raise ValueError(f"Unsupported EEG format {ext!r} for {path}")
    raw = readers[ext](path, preload=True, verbose=False)

    if subject is None:
        subject = path.name.split("_")[0]
    group = "".join(c for c in subject.removeprefix("sub-") if c.isalpha())
    session = next(
        (
            tok.removeprefix("ses-")
            for tok in path.name.split("_")
            if tok.startswith("ses-")
        ),
        None,
    )

    # Scalp EEG only: EXG* are mastoids/EOG, Status is the trigger line.
    raw.pick([ch for ch in raw.ch_names if not ch.startswith(("EXG", "Status"))])
    raw.set_montage(mne.channels.make_standard_montage(montage), on_missing="ignore")
    if resample is not None:
        raw.resample(resample)

    data = raw.get_data()  # volts, (n_channels, n_times)
    pos = np.array(
        [raw.info["chs"][i]["loc"][:3] for i in range(len(raw.ch_names))]
    )
    return EEGRecording(
        data=data,
        sfreq=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        positions=pos,
        subject=subject,
        group=group,
        session=session,
    )


def load_subject(
    root: str | Path,
    subject: str,
    task: str = "rest",
    session: str | None = None,
    **kwargs,
) -> EEGRecording:
    """Load one subject's task recording straight from the dataset root."""
    path = recording_path(root, subject, task=task, session=session)
    return load_recording(path, subject=subject, **kwargs)
