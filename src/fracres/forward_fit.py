"""Forward-fit workflow: drive the phantom with real EEG, fit, compare.

The core experiment of the real-data track (ROADMAP Next 1). The pipeline:

1. **Preprocess** a loaded :class:`~fracres.data.EEGRecording` -- average
   re-reference (the ds002778 BioSemi recordings are unreferenced) and per-channel
   z-scoring, so the reservoir sees unit-variance drives.
2. **Drive** the frozen fractional reservoir with one (or a few) real channels.
3. **Fit** the linear readout (closed-form ridge) to reconstruct the *remaining*
   channels, on a training segment only.
4. **Evaluate** on the held-out tail: per-channel correlation and normalised MSE
   (predictive skill), and DFA-H / spectral-:math:`\\beta` of model output vs
   recorded channel (does the phantom reproduce the data's statistics -- the
   success criterion, not just pointwise prediction).

Reconstruction (target channel at time ``t`` from the state at ``t``) is the
first rung; forecasting with an explicit horizon is a follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from fracres.data import EEGRecording
from fracres.metrics import SignalMetrics, signal_metrics
from fracres.training import fit_ridge_readout


def channel_matrix(
    recording: EEGRecording,
    reference: str | None = "average",
    zscore: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Preprocessed ``(n_times, n_channels)`` matrix + channel names.

    Parameters
    ----------
    reference : {"average", None}
        ``"average"`` subtracts the instantaneous cross-channel mean (common
        average reference) -- removes the dominant unreferenced-BioSemi offset
        and shared noise. ``None`` keeps the recorded potentials.
    zscore : bool
        Standardise each channel to zero mean / unit variance (over time).
    """
    x = np.asarray(recording.data, dtype=np.float64)  # (C, T)
    if reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    elif reference is not None:
        raise ValueError(f"unknown reference {reference!r}")
    if zscore:
        x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x.T.copy(), list(recording.ch_names)


@dataclass
class ForwardFitResult:
    """Outcome of :func:`forward_fit`.

    ``correlation`` / ``nmse`` are predictive-skill measures on the held-out
    tail; ``data_metrics`` / ``model_metrics`` are the per-target-channel LRD
    statistics whose agreement is the experiment's success criterion.
    """

    drive_channels: list[str]
    target_channels: list[str]
    correlation: np.ndarray  # (n_targets,) held-out Pearson correlation
    nmse: np.ndarray  # (n_targets,) held-out MSE / target variance
    data_metrics: list[SignalMetrics] = field(repr=False)
    model_metrics: list[SignalMetrics] = field(repr=False)
    y_true: np.ndarray = field(repr=False)  # (T_test, n_targets)
    y_pred: np.ndarray = field(repr=False)
    split: int = 0
    washout: int = 0
    beta: float = 0.0

    @property
    def mean_correlation(self) -> float:
        return float(np.mean(self.correlation))

    def metrics_table(self) -> str:
        """Per-target comparison of LRD statistics (data vs phantom output)."""
        lines = [
            f"{'channel':<10} {'DFA-H data':>10} {'DFA-H model':>11} "
            f"{'beta data':>9} {'beta model':>10} {'corr':>6}"
        ]
        for ch, dm, mm, c in zip(
            self.target_channels,
            self.data_metrics,
            self.model_metrics,
            self.correlation,
            strict=True,
        ):
            lines.append(
                f"{ch:<10} {dm.hurst_dfa:>10.3f} {mm.hurst_dfa:>11.3f} "
                f"{dm.spectral_beta:>9.2f} {mm.spectral_beta:>10.2f} {c:>6.3f}"
            )
        return "\n".join(lines)


def forward_fit(
    recording: EEGRecording,
    drive_channels: list[str],
    target_channels: list[str],
    model,
    beta: float = 1e-2,
    washout: int = 500,
    train_frac: float = 0.7,
    reference: str | None = "average",
) -> ForwardFitResult:
    """Drive *model* with real channels and reconstruct the remaining ones.

    Parameters
    ----------
    recording : EEGRecording
        Loaded recording (see :mod:`fracres.data`).
    drive_channels, target_channels : list of str
        Channel labels feeding the reservoir / reconstructed by the readout.
        ``model`` must have matching ``in_features`` / ``out_features``.
    model : PhantomBrain (or compatible)
        Frozen-reservoir model; only the readout is fit.
    beta : float
        Ridge regularisation for the closed-form readout.
    washout : int
        Initial samples excluded from fitting (reservoir transient).
    train_frac : float
        Fraction of the recording used to fit; the tail is held out.
    reference : {"average", None}
        Preprocessing reference, see :func:`channel_matrix`.
    """
    x, names = channel_matrix(recording, reference=reference, zscore=True)
    drive_idx = [names.index(c) for c in drive_channels]
    target_idx = [names.index(c) for c in target_channels]

    t_total = x.shape[0]
    split = int(train_frac * t_total)
    if not (washout < split < t_total):
        raise ValueError(
            f"need washout < split < T, got {washout}, {split}, {t_total}"
        )

    drive = jnp.asarray(x[:, drive_idx])
    target = jnp.asarray(x[:, target_idx])

    x_states, _ = model.simulate(drive)  # frozen reservoir, (T, N)
    w_out = fit_ridge_readout(x_states[:split], target[:split], beta, washout=washout)
    y_pred = np.asarray(x_states @ w_out.T)
    y_true = np.asarray(target)

    yt, yp = y_true[split:], y_pred[split:]
    correlation = np.array(
        [
            np.corrcoef(yp[:, j], yt[:, j])[0, 1]
            for j in range(len(target_channels))
        ]
    )
    nmse = np.array(
        [
            np.mean((yp[:, j] - yt[:, j]) ** 2) / np.var(yt[:, j])
            for j in range(len(target_channels))
        ]
    )
    data_metrics = [signal_metrics(yt[:, j]) for j in range(len(target_channels))]
    model_metrics = [signal_metrics(yp[:, j]) for j in range(len(target_channels))]

    return ForwardFitResult(
        drive_channels=list(drive_channels),
        target_channels=list(target_channels),
        correlation=correlation,
        nmse=nmse,
        data_metrics=data_metrics,
        model_metrics=model_metrics,
        y_true=yt,
        y_pred=yp,
        split=split,
        washout=washout,
        beta=beta,
    )
