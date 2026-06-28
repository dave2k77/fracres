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
      (leading coeff 1, `sigmoid`, no `h^alpha`) — corrected in
      `docs/knowledge_base_v2.md` §3.2.**
- [x] **Froze the reservoir for training** (v2 §6.2). `train_step` partitions the
      model via `readout_filter_spec`; only `readout.W_out` gets gradients
      (`W_res`/`W_in`/kernel weights verified frozen by a smoke test).
- [x] **Corrected the Besov indices** (v2 §4.2). `besov_indices(H, alpha_stable=2)`
      sets `p < alpha_S` (heavy-tail index, not the derivative order) and
      `s = min(H, 1/p) - margin`.
- [x] **Reconciled the qSOC controller** (v2 §5.1 / §10 item 3). The reservoir now
      low-passes a windowed energy state `E` (`tau_soc`) carried through the scan
      and updates the threshold with an unconditionally-stable semi-implicit step,
      so the written and implemented controllers agree. `simulate()` now also
      returns the `E` trajectory.

- [x] **Validated the GL/L1 kernels** against the analytic $D^\alpha t^\beta$
      (v2 §7.1). Added `kernels.apply()` (operator view), `validation.py`
      (`analytic_power_law_derivative`, `convergence_order`),
      `tests/test_validation.py`, and `examples/validate_kernels.py`. Measured
      orders match theory exactly — GL $O(h)$ (order ≈1.00), L1 $O(h^{2-\alpha})$
      (1.68/1.50/1.30/1.10 for $\alpha$=0.3/0.5/0.7/0.9) — confirming the L1
      telescoped weights and `Gamma(2-alpha)` forcing are correct.

- [x] **Validated on the Mittag-Leffler eigenfunction** ($x(0)=1$, exercising the
      Caputo-vs-RL distinction). Added `validation.mittag_leffler`; tests confirm
      GL reproduces the Riemann-Liouville derivative
      $\lambda E_\alpha + t^{-\alpha}/\Gamma(1-\alpha)$ (~1e-3), and both kernels
      recover the Caputo eigenvalue $\lambda E_\alpha$ via
      $D^\alpha_C f = D^\alpha_{RL}(f-f(0))$ (~1e-4). Verified the ML series
      against the $\alpha=1/2$ closed form $e^{z^2}\mathrm{erfc}(-z)$.

## Now — make the core solid
- [x] **Spectral-radius / Matignon control** (KB v2 §3.4). Added `stability.py`:
      `matignon_diagnostics` (min$|\arg\lambda_i(A)|$ vs $\alpha\pi/2$, plus
      $\rho(W)$ and $\sigma_\max(W)$ ESP diagnostics), `set_spectral_radius`
      (measure/set $\rho$ explicitly instead of the `0.95/sqrt(N)` heuristic),
      and the fractional edge-of-chaos control `matignon_edge_scale` /
      `set_edge_of_chaos` (bisection to the wedge boundary; model variant freezes
      the readout). `tests/test_stability.py` (10) and
      `examples/matignon_control.py` demonstrate the fractional advantage: one
      fixed $A$ is Matignon-stable for $\alpha < 0.84$, so the $\alpha=1$ classical
      ESN is unstable while the fractional node is stable.
- [x] **Closed-form ridge readout** (Tikhonov, KB v2 §6.3 / §1.4). Added
      `training.fit_ridge_readout` (pure `(out, N)` solve) and
      `fit_readout_ridge(model, ...)` (simulate → fit → `tree_at` the readout,
      reservoir frozen). `examples/fit_ridge_readout.py` shows the memory task
      (test corr 0.82→0.51 over delays 1→10). Possible extension: augment the
      readout with `[x; u]` / a bias term (KB §1.2).
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
- qSOC threshold now uses an unconditionally-stable semi-implicit step (resolved);
  the energy low-pass `E` still uses explicit Euler — fine as a 1st-order LPF, but
  pick `tau_soc` deliberately relative to `dt` and the dynamics timescale.
- `generate_fbm_increments` standardises by empirical std — confirm this
  preserves the intended fGn covariance closely enough, or switch to exact
  Davies–Harte scaling.
