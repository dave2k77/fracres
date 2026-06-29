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
- [x] Property-based tests (Hypothesis) for kernel invariants.
      `tests/test_kernel_properties.py` (14) assert the *algebraic* operator
      contract across all valid `(alpha, history_length)` — complementing the
      fixed-parameter numerical checks in `test_validation.py`. Covers:
      `weights` length, finiteness, deterministic construction, the strict-past
      weighted-sum identity, row-0 (current-state) independence, and linearity of
      both `__call__` and `apply`; plus the closed forms — GL `leading = alpha`,
      `forcing = 1`, the `c_j` recursion, all strict-past weights `<= 0`, and the
      linear-memory gain in `(0,1)` rising with `L`; L1 `leading = 2 - 2^{1-a}`,
      `forcing = Gamma(2-a)`, weights `< 0`. (Hypothesis flagged that GL `leading`
      equals `alpha` only to a float ULP — `c_1 = (1-(1+alpha))c_0` — so the test
      uses `isclose`, not `==`.)
- [x] Benchmarks (`benchmarks/`) for scan throughput vs `res_size`, history `L`.
      `benchmarks/scan_throughput.py` reports JIT-compiled, `block_until_ready`,
      median-of-reps throughput as $N$ and $L$ sweep, with the fitted log-log
      scaling exponent (measured ~1.2 in $N$ heading to the $O(N^2)$ connectome
      term; ~0.5 in $L$). Mirrored in a performance-tier guard
      `tests/performance/test_scan_scaling_perf.py` (sub-cubic in $N$, sub-quadratic
      in $L$) behind a `performance` marker that is deselected by default
      (`-m "not performance"`) and run in a non-gating `continue-on-error` CI job.
- [x] Config system (dataclass + YAML) for experiment specs. `fracres.config`
      provides validated dataclasses (`ExperimentConfig` over `KernelConfig` /
      `ModelConfig` / `DriveConfig` / `TrainingConfig`, each validating its fields
      in `__post_init__`), YAML `save_config`/`load_config` + `to_dict`/`from_dict`
      round-trip, and factories `build_kernel`/`build_model`/`build_drive`/
      `build_experiment` that turn a spec (plus the integer `seed`) into the live
      objects — variant-specific reservoir kwargs ride in `ModelConfig.params`. A
      run is reproducible from one file: `configs/memory_task.yaml` +
      `examples/config_experiment.py`. Adds `pyyaml` as a core dependency.
      `tests/test_config.py` (23) cover validation, round-trip, the factories
      across model variants, param forwarding, and seed reproducibility.
- [x] CI (ruff + pytest) mirroring `hpfracc`. `.github/workflows/ci.yml` runs a
      `ruff check .` lint job and a `pytest` job across Python 3.11/3.12. Brought
      the tree to ruff-clean under the existing `[tool.ruff]` config (E/F/I/UP/B):
      `docs` excluded (archived starter fragments), per-file `E402` ignores for the
      x64-config files, and ~50 line-wraps. README CI badge added.
- [x] Decide whether to back kernels with `hpfracc` for a single validated
      source of truth. **Decision: cross-validate, do not couple.** The two
      libraries are different operator *views* — `fracres` kernels are a
      decomposed one-step *recurrence* (`leading`/`weights`/`forcing_factor`
      advanced over a rolling buffer in the reservoir scan), while `hpfracc.ops`
      is a *batch* full-history operator (whole signal → `D^alpha x`). The
      underlying weight math is identical (GL binomial recursion; L1
      `b_k = (k+1)^{1-a} - k^{1-a}`), but hpfracc's public API doesn't expose the
      recurrence form the reservoir needs, and it is pre-alpha (provisional ops,
      churning surface). Depending on it would couple `fracres` to a moving
      target for no runtime gain. Instead, `tests/test_hpfracc_crossref.py`
      asserts that `AbstractFractionalKernel.apply` reproduces
      `hpfracc.ops.grunwald_letnikov` / `caputo` to ~1e-6 (all four `alpha`),
      giving the "single source of truth" *assurance* with **zero** hard
      dependency: the module `pytest.importorskip`s hpfracc, so it runs only
      where hpfracc is installed and is skipped by default (incl. CI).
      Revisit a real integration only if `fracres` needs capabilities hpfracc
      already has — per-state/vector `alpha`, non-uniform grids, or
      FFT/short-memory/SOE history acceleration.

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
