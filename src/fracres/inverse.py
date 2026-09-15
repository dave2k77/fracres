"""Mechanism inference: the inverse problem on real EEG.

Given a recorded channel's long-range-dependence statistics, ask *which
generative mechanism reproduces them*: sweep the phantom's fractional order
:math:`\\alpha`, the drive's Hurst exponent ``H``, and the reservoir leak
``decay`` (:math:`\\lambda`), and score each point by the distance between the
model output's metrics (DFA-H, spectral-:math:`\\beta`) and the data's
(ROADMAP Next 1, "what fractional order does this brain region act like?").

The comparison is **generative**: the reservoir is driven by synthetic fGn of
known ``H`` (not by another channel -- that is the forward-fit workflow's job),
so ``H`` here is a *property of the inferred mechanism*, and the OFF/ON
medication contrast becomes a question about how the best-fit
:math:`(\\alpha, H, \\lambda)` moves with dopaminergic state.

Two drive regimes (``drive_kind``, via :mod:`fracres.drivers`):

- ``"fgn"`` -- fractional Gaussian noise (stationary increments). A stable
  reservoir output then lives at :math:`\\beta < 1`, DFA-H :math:`< 1`.
- ``"fbm"`` -- fractional Brownian motion (:func:`fracres.drivers.generate_fbm`,
  the z-scored cumulative sum of the fGn). Scalp EEG is fBm-*like*
  (nonstationary: :math:`\\beta \\approx 1.5-2`, DFA-H :math:`> 1`), and a
  contractive reservoir cannot produce :math:`\\beta > 1` from a stationary
  drive -- an fGn sweep against raw EEG bottoms out at a large residual, which
  is itself the diagnostic that the drive must be integrated. With an fBm
  drive the reservoir passes the low-frequency power law through, so
  anti-persistent ``H`` (:math:`\\approx 0.3-0.4`) reproduces the EEG regime
  (:math:`\\beta = 2H + 1`, DFA-H :math:`= H + 1`).

Design choices:

- **Common random numbers**: every grid point is evaluated with the *same* set
  of model/drive seeds, so grid-to-grid differences reflect the parameters,
  not the draws; metrics are averaged over the seed ensemble.
- **Length-matched simulation**: the phantom is simulated for exactly the
  target channel's sample count, so both sides' estimators see the same
  finite-sample conditions (DFA and log-periodogram biases are length-dependent).
- **Objective** is the (optionally weighted) Euclidean distance on the
  ``(hurst_dfa, spectral_beta)`` metric vector; both components are O(1) for
  EEG, so no rescaling is needed by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np

from fracres.data import EEGRecording
from fracres.drivers import generate_fbm, generate_fbm_increments
from fracres.forward_fit import channel_matrix
from fracres.kernels import GLKernel
from fracres.metrics import SignalMetrics, signal_metrics
from fracres.models import PhantomBrain

MetricVector = tuple[float, float]


def _vector(m: SignalMetrics) -> MetricVector:
    return (m.hurst_dfa, m.spectral_beta)


def metric_distance(
    model: SignalMetrics,
    data: SignalMetrics,
    weights: MetricVector = (1.0, 1.0),
) -> float:
    """Weighted Euclidean distance on ``(hurst_dfa, spectral_beta)``."""
    dm, dd = _vector(model), _vector(data)
    return float(
        np.sqrt(sum(w * (a - b) ** 2 for a, b, w in zip(dm, dd, weights, strict=True)))
    )


@dataclass
class GridPoint:
    """One evaluated mechanism configuration."""

    alpha: float
    hurst: float
    decay: float
    drive_kind: str
    objective: float
    model_metrics: SignalMetrics = field(repr=False)  # seed-averaged


@dataclass
class MechanismFitResult:
    """Outcome of :func:`grid_search`: every evaluated point, best first."""

    channel: str
    subject: str
    session: str | None
    data_metrics: SignalMetrics = field(repr=False)
    points: list[GridPoint] = field(repr=False)
    n_seeds: int = 1

    @property
    def best(self) -> GridPoint:
        return self.points[0]

    def leaderboard(self, k: int = 5) -> str:
        """Top-``k`` grid points with their achieved metrics."""
        dm = self.data_metrics
        lines = [
            f"data:  DFA-H={dm.hurst_dfa:.3f}  beta={dm.spectral_beta:.2f}",
            f"{'rank':<5}{'alpha':>7}{'H':>6}{'decay':>7}{'obj':>8}"
            f"{'DFA-H':>8}{'beta':>7}",
        ]
        for rank, p in enumerate(self.points[:k], 1):
            lines.append(
                f"{rank:<5}{p.alpha:>7.2f}{p.hurst:>6.2f}{p.decay:>7.2f}"
                f"{p.objective:>8.3f}{p.model_metrics.hurst_dfa:>8.3f}"
                f"{p.model_metrics.spectral_beta:>7.2f}"
            )
        return "\n".join(lines)


def _simulate_metrics(
    alpha: float,
    hurst: float,
    decay: float,
    t_steps: int,
    drive_kind: str,
    res_size: int,
    history_length: int,
    spectral_scale: float,
    step_size: float,
    n_seeds: int,
    base_seed: int,
) -> SignalMetrics:
    """Seed-averaged output metrics of the phantom at one grid point.

    The same ``(model, drive)`` seed pairs are used for every grid point
    (common random numbers), so comparisons across the grid are not confounded
    by draw-to-draw variance.
    """
    dfa_vals, beta_vals = [], []
    for s in range(n_seeds):
        key = jax.random.PRNGKey(base_seed + 104729 * s)
        k_model, k_drive = jax.random.split(key)
        kernel = GLKernel(alpha=alpha, history_length=history_length)
        model = PhantomBrain(
            1, res_size, 1, kernel,
            key=k_model, spectral_scale=spectral_scale, step_size=step_size,
            decay=decay,
        )
        drive = (
            generate_fbm(t_steps, H=hurst, key=k_drive)
            if drive_kind == "fbm"
            else generate_fbm_increments(t_steps, H=hurst, key=k_drive)
        )
        drive = drive[:, None]
        _, y_hat = model.simulate(drive)
        m = signal_metrics(np.asarray(y_hat[:, 0]))
        dfa_vals.append(m.hurst_dfa)
        beta_vals.append(m.spectral_beta)
    return SignalMetrics(
        hurst_dfa=float(np.mean(dfa_vals)),
        spectral_beta=float(np.mean(beta_vals)),
        hurst_spectral=float(0.5 * (np.mean(beta_vals) + 1.0)),
    )


def grid_search(
    recording: EEGRecording,
    channel: str,
    alphas: list[float],
    hursts: list[float],
    decays: list[float] = (2.0,),
    drive_kind: str = "fgn",
    reference: str | None = "average",
    res_size: int = 200,
    history_length: int = 100,
    spectral_scale: float = 0.95,
    step_size: float = 0.4,
    n_seeds: int = 3,
    weights: MetricVector = (1.0, 1.0),
    base_seed: int = 0,
) -> MechanismFitResult:
    """Score every ``(alpha, H, decay)`` combination against one EEG channel.

    Parameters
    ----------
    recording : EEGRecording
        Loaded recording; the channel is preprocessed with the same average
        reference + z-scoring as the forward-fit workflow.
    channel : str
        Target channel label ("what fractional order does *this* region act
        like?").
    reference : {"average", None}
        Preprocessing reference, see :func:`~fracres.forward_fit.channel_matrix`.
    alphas, hursts, decays : sequence of float
        Grid axes: kernel order, drive Hurst exponent, reservoir leak.
    drive_kind : {"fgn", "fbm"}
        Drive regime (see module docstring). ``"fbm"`` is the well-posed
        choice for raw scalp EEG (:math:`\\beta > 1`); ``"fgn"`` suits
        stationary/increment-domain targets.
    res_size, history_length, spectral_scale, step_size
        Fixed model hyperparameters (the tuned defaults).
    n_seeds : int
        Ensemble size per grid point (common random numbers across the grid).
    weights : (float, float)
        Objective weights on (DFA-H, spectral-beta) mismatches.

    Returns
    -------
    MechanismFitResult
        Points sorted by objective (best first).
    """
    x, names = channel_matrix(recording, reference=reference, zscore=True)
    trace = x[:, names.index(channel)]
    data_metrics = signal_metrics(trace)
    t_steps = x.shape[0]
    if drive_kind not in ("fgn", "fbm"):
        raise ValueError(f"unknown drive_kind {drive_kind!r}")

    points = []
    for alpha in alphas:
        for hurst in hursts:
            for decay in decays:
                model_metrics = _simulate_metrics(
                    alpha, hurst, decay, t_steps, drive_kind, res_size, history_length,
                    spectral_scale, step_size, n_seeds, base_seed,
                )
                points.append(
                    GridPoint(
                        alpha=alpha,
                        hurst=hurst,
                        decay=decay,
                        drive_kind=drive_kind,
                        objective=metric_distance(model_metrics, data_metrics, weights),
                        model_metrics=model_metrics,
                    )
                )
    points.sort(key=lambda p: p.objective)
    return MechanismFitResult(
        channel=channel,
        subject=recording.subject,
        session=recording.session,
        data_metrics=data_metrics,
        points=points,
        n_seeds=n_seeds,
    )
