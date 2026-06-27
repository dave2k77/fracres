# Roadmap

Tracks the gap between the current scaffold and a publishable model. Grouped by
priority; check items off as they land.

## Done
- [x] **Stabilised the discrete update.** The naive `x_{k-1} - memory + activation`
      form double-counts `x_{k-1}` (effective gain `1 + alpha`) and diverges. The
      reservoir now uses the kernel-supplied `leading`/`weights`/`forcing_factor`
      (unit linear memory gain) plus an `h^alpha`-scaled `-lambda x_{k-1}` leak
      and zero-centred `tanh`, giving bounded state norms. **This deliberately
      departs from the literal coefficients in `docs/knowledge_base.md` §3.2/§6
      (leading coeff 1, `sigmoid`, no `h^alpha`) — the note should be updated to
      match.**

## Now — make the core solid
- [ ] **Validate the GL/L1 kernels** against analytic fractional derivatives
      (e.g. $D^\alpha t^\beta$) — port the validation style from `hpfracc`.
      Confirm the L1 telescoped weights and `Gamma(2-alpha)` forcing are exact.
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
- Step size `h` (`step_size`) and decay `lambda` (`decay`) are currently fixed
  defaults (0.1, 1.0). Treat them as first-class, possibly per-node, parameters
  and study how they trade off against the kernel `leading` gain.
- Explicit-Euler qSOC threshold update is stable at the tested parameters but
  may need an implicit/semi-implicit step at small `tau_b`.
- `generate_fbm_increments` standardises by empirical std — confirm this
  preserves the intended fGn covariance closely enough, or switch to exact
  Davies–Harte scaling.
