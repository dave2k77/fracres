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
- [x] Wire `training.train_step` into an end-to-end fit example with a target
      signal; confirm only the readout updates. `examples/train_readout.py` drives
      the full optax + Besov-regulariser loop on the delayed-copy memory task,
      prints train/test curves (held-out corr ≈ 0.61 at delay 2), and asserts the
      frozen-weight invariant: `W_res`/`W_in`/kernel weights come back identical,
      only `readout.W_out` moves (KB v2 §6.2). Gradient descent converges slower
      than the closed-form ridge, as expected — ridge is preferred when the Besov
      prior isn't needed.

## Next — scientific capability
- [x] **Excitatory/inhibitory neural-mass** reservoir (Wilson–Cowan form,
      knowledge base §2.2) with separate $\tau_E^\alpha$, $\tau_I^\alpha$. Added
      `reservoirs.WilsonCowanReservoir` and `models.WilsonCowanPhantomBrain`: two
      populations stacked as $z=[E;I]$ (width $2N$) advanced by the *same*
      validated GL update (shared `leading`/`weights`/`forcing_factor`, since they
      depend only on $\alpha_D$), with per-population drive
      $g_x=(-x+\mathcal{S}(\text{syn}))/\tau_x^{\alpha_D}$. Four non-negative
      connectomes ($W_{EE},W_{EI},W_{IE},W_{II}$); E/I signs live in the equations;
      input enters $E$ only. Firing rate defaults to a logistic sigmoid (the WC
      standard) but is configurable. The readout sees the full E/I state, so it
      works unchanged with the training / ridge utilities. `tests/test_wilson_cowan.py`
      (10) verify bounded dynamics, the separate-$\tau$ knobs, and the frozen-reservoir
      invariant; `examples/wilson_cowan.py` shows the E/I activity and fading memory.
- [x] **`NeuralFieldReservoir`** — spatial connectivity kernel over a cortical
      sheet (Amari 1977, KB §2.2). Replaced the stub with the fractional Amari
      field $\tau^\alpha\mathcal{D}^\alpha u = -u + w * \mathcal{S}(u) + W_{in}u_{ext}$
      on a 1-D periodic ring: a distance-dependent **Mexican-hat** kernel
      (`mexican_hat_kernel` / `ring_distance`, symmetric & circulant, near-zero DC
      gain) replaces random links, and the firing-rate non-linearity sits *inside*
      the convolution ($w * \mathcal{S}(u)$, the Amari signature). Same validated GL
      update; readout reads the whole field, so `NeuralFieldPhantomBrain` works with
      the training / ridge utilities. `tests/test_neural_field.py` (11) verify the
      geometry, frozen connectivity, bounded dynamics, and that a white-noise drive
      makes the field self-organise at the kernel's preferred wavelength;
      `examples/neural_field.py` shows the kernel, spatial pattern selection, and
      fading memory.
- [x] **Metrics module** (`metrics.py`). Long-range dependence: `hurst_dfa`
      (DFA-1, scale range chosen to dodge the small-scale crossover and noisy top
      scales) and `spectral_exponent` (log-binned low-frequency log-periodogram
      regression — the band restriction and binning are essential, a full-band raw
      fit is severely biased/noisy); `signal_metrics` bundles both with the spectral
      estimate mapped onto the $H$ scale ($\beta = 2H-1$ for fGn). Criticality:
      `detect_avalanches` (supra-threshold excursions), `power_law_exponent` (CSN
      MLE), and `avalanche_exponents` $\to (\tau,\alpha)$. Pure NumPy (deterministic
      under the suite's global x64). `tests/test_metrics.py` (14) verify recovery on
      fGn of known $H$ (accurate to ~0.7; degrades as $H\to1$ at finite length — a
      DFA-1 limit), MLE exponent recovery, and avalanche detection;
      `examples/signal_metrics.py` characterises drive vs reservoir output and shows
      the reservoir lifts a white drive's $H$ above 0.5.
- [x] Validate the Besov regulariser actually moves trajectories toward the
      target $B^s_{p,q}$ regularity. Exposed `regularizers.dyadic_band_energies`
      (the per-band $\lVert\Delta_j Y\rVert_p$ building block, now reused by
      `littlewood_paley_penalty`) as a regularity diagnostic. `tests/test_regularizer_validation.py`
      (8) verify the Littlewood-Paley machinery (a single-band cosine deposits its
      energy in the right band; band powers reconstruct the total via Parseval),
      that the penalty ranks smoother signals lower and grows with the target $s$,
      and the operational claim: minimising $\mathrm{MSE}+\lambda B^s_{p,q}$ on a
      rough trajectory collapses its high-frequency bands and **the achieved
      regularity tracks the target $s$** (smoothness $\approx s$). `examples/besov_regularization.py`
      shows the band-energy collapse, the $s$-tracking, and the indices
      `besov_indices` derives from a drive's $H$ (with $1/p$ binding at large $H$).

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
