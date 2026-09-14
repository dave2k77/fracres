"""qSOC -> avalanches study: does homeostasis produce critical statistics?

ROADMAP "Next 3". The cortical criticality signature is power-law neuronal
avalanches with size exponent tau ~ 1.5 and duration exponent alpha ~ 2.0
(Beggs & Plenz 2003). This script asks three questions of the qSOC phantom
brain:

1. SIGNATURE -- across a small (alpha, H) grid, do the supra-threshold
   population events of the qSOC reservoir organise into avalanches with
   power-law-like size/duration distributions, and where do the exponents land
   relative to (1.5, 2.0)?
2. MECHANISM -- is the homeostat necessary? A matched no-qSOC control
   (``PhantomBrain``, same seed/connectome draw) is characterised the same way.
3. RECOVERY -- after a strong perturbation (drive amplitude x 5 for a segment),
   does the windowed energy E return toward E_crit, and do the avalanche
   exponents before the kick and after recovery agree?

Avalanche pipeline (standard): washout -> z-score each node over time ->
event = |z| > THRESH_Z -> population activity a(t) = event count per step ->
avalanche = maximal run of a(t) > 0; size = total events, duration = steps
(``fracres.metrics.detect_avalanches`` with threshold 0). Exponents are
Clauset-Shalizi-Newman MLEs (``power_law_exponent``), reported both from the
smallest size and from the median size (robustness to the small-event heap).

Two calibration details matter:

* ``E_crit`` must be *reachable*: the reservoir's mean-square state is bounded
  (tanh saturation), so an ``E_crit`` far above the natural energy pegs the
  threshold at maximum excitation and there is no regulation. Here ``E_crit``
  is calibrated per condition to 1.1x the late-window energy of the matched
  no-qSOC reservoir (a short pilot run), so the homeostat regulates *around*
  the operating point.
* After the amplitude kick, z-scores must be computed *per analysis window*:
  normalising against the kicked transient inflates the variance and silently
  erases every event outside the kick.

Run:  python examples/qsoc_avalanches.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from fracres import (
    GLKernel,
    PhantomBrain,
    avalanche_exponents,
    detect_avalanches,
    generate_fbm_increments,
    power_law_exponent,
    qSOCPhantomBrain,
)

# Simulation (tuned defaults from the h/lambda study, ROADMAP OQ#1)
T = 20000
T_PILOT = 5000
RES_SIZE = 300
HISTORY = 100
STEP_SIZE = 0.4
DECAY = 2.0
DT = 0.01  # qSOC controller step
# qSOC homeostat (E_crit is calibrated per condition; see calibrate_ecrit)
E_CRIT_GAIN = 1.1
TAU_B = 5.0
GAMMA = 1.0
TAU_SOC = 10.0
# Avalanche pipeline
WASHOUT = 1000
THRESH_Z = 2.0


def build_qsoc(alpha, e_crit, key):
    kernel = GLKernel(alpha=alpha, history_length=HISTORY)
    return qSOCPhantomBrain(
        in_features=1,
        res_size=RES_SIZE,
        out_features=1,
        fractional_operator=kernel,
        key=key,
        step_size=STEP_SIZE,
        decay=DECAY,
        E_crit=e_crit,
        tau_b=TAU_B,
        gamma=GAMMA,
        tau_soc=TAU_SOC,
    )


def calibrate_ecrit(alpha, hurst, key):
    """1.1x the late-window mean-square state of the matched no-qSOC reservoir.

    A short pilot run of the plain ``PhantomBrain`` (same seed, same drive)
    gives the natural operating energy; regulating slightly *above* it keeps
    the set point reachable (tanh saturation caps the attainable energy).
    """
    k_model, k_noise = jax.random.split(key)
    kernel = GLKernel(alpha=alpha, history_length=HISTORY)
    plain = PhantomBrain(1, RES_SIZE, 1, kernel, key=k_model,
                         step_size=STEP_SIZE, decay=DECAY)
    drive = generate_fbm_increments(T_PILOT, H=hurst, key=k_noise)[:, None]
    X, _ = plain.simulate(drive)
    e_natural = float(jnp.mean(jnp.square(X[WASHOUT:])))
    return E_CRIT_GAIN * e_natural


def population_activity(X_states):
    """Event-count trace a(t): nodes crossing |z| > THRESH_Z at each step.

    z-scores are computed over *this* slice only -- pass pre/post windows
    separately when the trace contains a perturbation transient.
    """
    X = np.asarray(X_states)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    return (np.abs(z) > THRESH_Z).sum(axis=0).astype(float)


def avalanche_summary(activity, label):
    """Exponent table row for one activity trace."""
    ex = avalanche_exponents(activity, threshold=0.0)
    sizes, durations = detect_avalanches(activity, threshold=0.0)
    if ex.n_avalanches == 0:
        print(f"  {label:<38} n=    0  (no supra-threshold avalanches)")
        return ex
    # Robustness: refit above the median (drops the heap of 1-event avalanches).
    tau_med = power_law_exponent(sizes, xmin=float(np.median(sizes)))
    alpha_med = power_law_exponent(durations.astype(float),
                                   xmin=float(np.median(durations)))
    print(f"  {label:<38} n={ex.n_avalanches:>5}  "
          f"tau={ex.tau:5.2f} (med-xmin {tau_med:5.2f})  "
          f"alpha={ex.alpha:5.2f} (med-xmin {alpha_med:5.2f})")
    return ex


def simulate_qsoc(alpha, hurst, key, e_crit=None, T_run=T):
    k_model, k_noise = jax.random.split(key)
    if e_crit is None:
        e_crit = calibrate_ecrit(alpha, hurst, key)
    model = build_qsoc(alpha, e_crit, k_model)
    drive = generate_fbm_increments(T_run, H=hurst, key=k_noise)[:, None]
    X, _, B, E = model.simulate(drive, dt=DT)
    return model, drive, X, B, E, e_crit


def main():
    print(f"qSOC avalanche study: N={RES_SIZE}, T={T}, h={STEP_SIZE}, "
          f"lambda={DECAY}, washout={WASHOUT}, event |z|>{THRESH_Z}")
    print(f"homeostat: E_crit={E_CRIT_GAIN}x natural energy, tau_b={TAU_B}, "
          f"gamma={GAMMA}, tau_soc={TAU_SOC}  "
          f"(cortical target: tau~1.5, alpha~2.0)\n")

    # --- 1. SIGNATURE: (alpha, H) grid --------------------------------------
    print("[1] Signature sweep (qSOC on):")
    for alpha in (0.6, 0.8):
        for hurst in (0.5, 0.7):
            key = jax.random.PRNGKey(100 + int(alpha * 10) + int(hurst * 10))
            _, _, X, _, E, e_crit = simulate_qsoc(alpha, hurst, key)
            e_late = float(E[-500:].mean())
            label = f"alpha={alpha}, H={hurst} (E={e_late:.3f}/{e_crit:.3f})"
            avalanche_summary(population_activity(X[WASHOUT:]), label)

    # --- 2. MECHANISM: matched no-homeostat control -------------------------
    print("\n[2] Mechanism control (alpha=0.8, H=0.7):")
    key = jax.random.PRNGKey(118)
    k_model, k_noise = jax.random.split(key)
    _, _, X_q, _, _, e_crit = simulate_qsoc(0.8, 0.7, key)
    avalanche_summary(population_activity(X_q[WASHOUT:]), "qSOC (homeostat on)")
    kernel = GLKernel(alpha=0.8, history_length=HISTORY)
    plain = PhantomBrain(1, RES_SIZE, 1, kernel, key=k_model,
                         step_size=STEP_SIZE, decay=DECAY)
    drive = generate_fbm_increments(T, H=0.7, key=k_noise)[:, None]
    X_p, _ = plain.simulate(drive)
    avalanche_summary(population_activity(X_p[WASHOUT:]), "no qSOC (control)")

    # --- 3. RECOVERY: drive-amplitude kick ----------------------------------
    print("\n[3] Perturbation recovery (alpha=0.8, H=0.7):")
    key = jax.random.PRNGKey(218)
    model, drive, X, B, E, e_crit = simulate_qsoc(0.8, 0.7, key)
    kick_lo, kick_hi = T // 2, T // 2 + 1000
    drive_kicked = drive.at[kick_lo:kick_hi].multiply(5.0)
    Xk, _, Bk, Ek = model.simulate(drive_kicked, dt=DT)

    # Window-local normalisation: the kicked transient must not set the scale.
    pre = population_activity(Xk[WASHOUT:kick_lo])
    post = population_activity(Xk[kick_hi + 2000:])
    avalanche_summary(pre, "pre-kick")
    avalanche_summary(post, "post-recovery")
    e_pre = float(E[kick_lo - 500:kick_lo].mean())
    e_kick = float(Ek[kick_lo:kick_hi].mean())
    e_post = float(Ek[-500:].mean())
    print(f"  windowed energy E: pre={e_pre:.3f} -> kicked={e_kick:.3f} "
          f"-> late={e_post:.3f}  (E_crit={e_crit:.3f})")

    print("\nDone. Interpretation: exponents near (1.5, 2.0) that survive the "
          "kick and are absent/degraded in the no-qSOC control would support "
          "the homeostat-as-criticality-mechanism hypothesis.")


if __name__ == "__main__":
    main()
