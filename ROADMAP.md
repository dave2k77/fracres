# Roadmap

Tracks the gap between the current scaffold and a publishable model.

**Status (September 2026):** the v0.1 baseline is complete — validated fractional
kernels, four reservoir variants, Matignon / qSOC control, Besov-regularised and
closed-form readout training, LRD + criticality metrics, config system, CI, and
a learning manual. The sections below are the *next* roadmap: from phantom
generator to instrument.

## Completed — v0.1 baseline

The detailed record of the baseline lives in git history; key verified results:

- **Kernels** (GL / L1-Caputo): stable gain-normalised one-step recurrence
  (`leading`/`weights`/`forcing_factor`); convergence orders match theory
  exactly vs analytic $D^\alpha t^\beta$ (GL $O(h)$ ≈ 1.00; L1 $O(h^{2-\alpha})$);
  Mittag-Leffler eigenfunction validation (Caputo vs RL); cross-validated to
  ~1e-6 against `hpfracc` (opt-in, no hard dependency — decision: cross-validate,
  do not couple).
- **Reservoirs**: `FractionalReservoir`, `qSOCFractionalReservoir` (windowed
  energy + unconditionally-stable semi-implicit threshold), `WilsonCowanReservoir`
  (E/I masses, separate $\tau_E^\alpha, \tau_I^\alpha$), `NeuralFieldReservoir`
  (fractional Amari field, Mexican-hat ring connectivity, pattern selection).
- **Stability**: Matignon diagnostics + `set_edge_of_chaos`; demonstrated
  fractional advantage (one $A$ stable for $\alpha<0.84$ where the classical
  $\alpha=1$ ESN is not).
- **Training**: readout-only by construction (frozen-reservoir invariant
  tested); closed-form ridge (memory task corr 0.82→0.51 over delays 1→10) and
  optax + Besov penalty (held-out corr ≈ 0.61 at delay 2).
- **Besov regulariser**: minimising MSE + $\lambda B^s_{p,q}$ collapses
  high-frequency bands and achieved smoothness tracks target $s$; indices
  $p<\alpha_S$, $s<\min\{H,1/p\}$.
- **Drive**: exact Davies–Harte fGn (unit variance in expectation, exact
  autocovariance); tested against analytic $r(k)$ for $H\in\{0.3,0.5,0.7,0.9\}$.
- **Metrics**: DFA-1 Hurst and log-binned spectral exponent recover $H$ on fGn
  (~0.7 accuracy, degrading as $H\to1$); avalanche detection + CSN MLE
  exponents; reservoir lifts a white drive's $H$ above 0.5.
- **Tuning study** ($h$, $\lambda$): defaults raised to (0.4, 2.0) in
  `configs/memory_task.yaml`, recall 0.64→0.95; leak is load-bearing
  ($\lambda=0 \Rightarrow$ ~0 recall); good regime at $g_{eff}\in[0,0.2]$.
- **Infra**: Hypothesis property tests, scan-throughput benchmarks + perf CI
  tier, dataclass+YAML config system, ruff+pytest CI (3.11/3.12), learning
  manual (`docs/learning_manual.md`).

## Next 1 — Real data (the main thrust)

Turn fracres from a phantom generator into an instrument, using the Parkinson's
EEG from the broader LRD project.

- [ ] **Data ingestion**: real EEG/MEG loading with channel geometry and
      sampling-rate handling.
- [ ] **Forward-fit workflow**: drive the reservoir with a real signal, fit the
      readout to reconstruct/forecast held-out channels, compare achieved
      DFA-$H$ / spectral-$\beta$ / avalanche exponents between model output and
      recording. Success = the phantom reproduces the data's statistics.
- [ ] **Mechanism inference (inverse problem)**: sweep $(\alpha, H, \lambda,$
      variant) against real-data statistics — "what fractional order does this
      brain region act like?" Needs an objective comparing model-output metrics
      to data metrics, plus a grid-search/optimiser driver (config system
      already makes the sweep declarative).

## Next 2 — Model expressiveness

Pulled in as the data work demands, in rough dependency order:

- [ ] **Per-node $\alpha$, $\lambda$, $h$**: the scan already broadcasts — a
      small change with large payoff (heterogeneous memory timescales; ties to
      the neural-timescales project).
- [ ] **Multi-channel / spatially-embedded readout**: leadfield-like projection
      $N$ nodes $\to C$ scalp channels; the neural-field variant is the natural
      host (ring sites $\to$ electrode positions).
- [ ] **Heavy-tailed (non-Gaussian) drive**: Lévy/stable fGn analogue so the
      $\alpha_{stable}<2$ Besov machinery is actually exercised — the regime
      where L2 estimators fail structurally.
- [ ] **Structural-break injection**: piecewise-$H$ / regime-switch drives for
      estimator-robustness benchmarks.

## Next 3 — Criticality science (no new code needed)

- [x] **qSOC → avalanches study** (`examples/qsoc_avalanches.py`). Findings:
      (i) supra-threshold population events organise into power-law-like
      avalanches across the $(\alpha, H)$ grid — size $\tau\approx1.2$ (a
      little under the cortical 1.5; CSN MLE at $n\approx70$ avalanches is
      noisy) and duration $\alpha\approx1.7$–$3.0$, rising with the fractional
      order; (ii) the homeostat's measurable role is **robustness, not
      necessity**: with $E_{crit}$ calibrated to the natural operating energy a
      matched no-qSOC control shows the *same* exponents, but after a 5×
      drive-amplitude kick the qSOC run's energy returns to the set point
      ($0.093 \to 0.125 \to 0.091$) and its pre/post-kick exponents are
      statistically indistinguishable ($\tau$ 1.20 vs 1.21); (iii) two
      calibration pitfalls documented in the script — $E_{crit}$ must be
      reachable (tanh saturation caps the energy; an unreachable set point pegs
      the threshold and disables regulation), and post-perturbation z-scores
      must be window-local. Follow-up: multi-seed ensembles for tighter MLE
      error bars; sweep $E_{crit}$ *away* from the natural energy to find where
      the homeostat becomes load-bearing.
- [ ] **E/I balance and criticality**: does the Wilson–Cowan $\tau_E/\tau_I$
      ratio move avalanche exponents as the cortical literature predicts?

## Next 4 — Rigour & reproducibility

- [ ] **qSOC `tau_soc` guidance**: energy low-pass is explicit Euler; study and
      document how to pick `tau_soc` relative to `dt` and the dynamics
      timescale.
- [ ] **Estimator benchmark suite**: bias/variance characterisation of
      `hurst_dfa` / `spectral_exponent` vs known-$H$ ensembles, heavy tails,
      and breaks — feeds the Lp/Besov estimator argument.
- [ ] **Long-history scaling**: adopt hpfracc-style FFT / short-memory / SOE
      acceleration only if a task needs $L \gg 100$.

## Next 5 — Publication track

- [ ] Methods paper: fractional reservoir + validation results (kernel orders,
      Matignon advantage, Besov $s$-tracking). The learning manual is most of a
      supplementary tutorial.
- [ ] Phantom-vs-real-EEG study, once Next 1 lands.

## Open questions

- Per-node vectors: deferred until a task motivates them (see Next 2).
- `tau_soc` selection: see Next 4.
