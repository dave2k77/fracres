# Roadmap

Tracks the gap between the current scaffold and a publishable model. Grouped by
priority; check items off as they land.

## Now — make the core solid
- [ ] **Validate the GL/L1 kernels** against analytic fractional derivatives
      (e.g. $D^\alpha t^\beta$) — port the validation style from `hpfracc`.
- [ ] **Spectral-radius control**: measure and set $\rho(W_\mathrm{res})$
      explicitly instead of relying on the `0.95/sqrt(N)` heuristic.
- [ ] **Closed-form ridge readout** (Tikhonov) as an alternative to the
      gradient-trained readout — knowledge base §1.4.
- [ ] Wire `training.train_step` into an end-to-end fit example with a target
      signal; confirm only the readout updates.

## Next — scientific capability
- [ ] **Excitatory/inhibitory neural-mass** reservoir (Wilson–Cowan form,
      knowledge base §2.2) with separate $\tau_E^\alpha$, $\tau_I^\alpha$.
- [ ] **`NeuralFieldReservoir`** — spatial connectivity kernel over a cortical
      sheet (currently a stub).
- [ ] **Metrics module**: estimate $H$ / spectral slope of generated signals and
      compare to the drive; criticality diagnostics (avalanche statistics).
- [ ] Validate the Besov regulariser actually moves trajectories toward the
      target $B^s_{p,q}$ regularity.

## Later — rigour & reproducibility
- [ ] Property-based tests (Hypothesis) for kernel invariants.
- [ ] Benchmarks (`benchmarks/`) for scan throughput vs `res_size`, history `L`.
- [ ] Config system (e.g. dataclass + YAML) for experiment specs.
- [ ] CI (ruff + pytest) mirroring `hpfracc`.
- [ ] Decide whether to back kernels with `hpfracc` for a single validated
      source of truth (currently self-contained — see kernels interface).

## Open questions
- Numerical stability of the explicit-Euler qSOC threshold update at small
  `tau_b`; consider an implicit/semi-implicit step.
- `generate_fbm_increments` standardises by empirical std — confirm this
  preserves the intended fGn covariance closely enough, or switch to exact
  Davies–Harte scaling.
