# Learning Manual — Fractional Reservoir Computing & Phantom Brains

A guided tour of the ideas in this repository (`fracres`), built to be *read in
order* as a self-study course. Each part introduces a concept, gives just enough
theory to make it precise, then shows exactly where and how it lives in the code.

This manual is the **pedagogical** companion to two reference documents:

- [`docs/knowledge_base_v2.md`](knowledge_base_v2.md) — the terse, corrected
  theory reference (the "what is true" document; sections cited below as *KB §x*).
- [`ROADMAP.md`](../ROADMAP.md) — what is built, what is next, and the resolved
  open questions (including the worked `h`/`λ` study in Part 11).

> **How to use this.** Read Parts 1–3 first; they fix the vocabulary everything
> else reuses. After that the parts are fairly independent — jump to whatever you
> need. Every part ends with **"Run it"** (an example script) and **"Check your
> understanding"** prompts. Code pointers use `module.function` names you can
> grep for; runnable demonstrations live in [`examples/`](../examples).

---

## Table of contents

1. [The big picture: what is a "phantom brain"?](#part-1)
2. [Reservoir computing foundations](#part-2)
3. [Fractional calculus for nodes](#part-3)
4. [The discrete update: from operator to recurrence](#part-4)
5. [From one node to a brain: the four reservoir variants](#part-5)
6. [Stability: Matignon's theorem and the fractional edge of chaos](#part-6)
7. [Self-organised criticality: the qSOC controller](#part-7)
8. [The driving noise: fractional Gaussian noise](#part-8)
9. [Function-space regularisation: Besov & Littlewood–Paley](#part-9)
10. [Training the readout](#part-10)
11. [Measuring what you made: LRD & avalanche metrics](#part-11)
12. [Validating the kernels against ground truth](#part-12)
13. [Putting it together: the config system & a worked study](#part-13)
- [Appendix A: symbol glossary](#appendix-a)
- [Appendix B: theory → code map](#appendix-b)
- [Appendix C: exercises](#appendix-c)

---

<a name="part-1"></a>
## Part 1 — The big picture: what is a "phantom brain"?

The goal of this project is to **generate synthetic macroscopic brain signals**
(EEG/MEG-like time series) that have the *statistical fingerprints of real
cortex*: long-range temporal dependence (slowly decaying autocorrelations, $1/f$-
like spectra) and signatures of operation near a **critical point** (scale-free
"neuronal avalanches"). A model that produces such signals without being a literal
biophysical simulation is what we call, loosely, a **phantom brain**.

The strategy has three moving parts, and the whole repository is organised around
them:

1. **A reservoir** — a large pool of fixed, randomly connected non-linear nodes
   (Part 2). We never train its internal weights; we only read it out linearly.
2. **Fractional memory in the nodes** — instead of the usual exponentially-fading
   memory, each node integrates its past with a *power-law* kernel (Part 3). This
   is the key novelty: long-range dependence is placed *in the nodes themselves*,
   not engineered into the network topology.
3. **Criticality + the right input** — the reservoir is driven by heavy-tailed
   **fractional Gaussian noise** (Part 8), held near the **edge of chaos** by a
   stability criterion (Part 6) and an optional homeostatic controller (Part 7),
   and its output is nudged onto the correct regularity manifold by a **Besov-space
   regulariser** (Part 9).

We then **measure** the generated signal (Part 11) and **validate** the numerical
machinery (Part 12) to confirm we got what we intended.

> **Why "fractional"?** A classical reservoir produces long-range dependence
> through network structure (carefully tuned connectivity). Here we get it for
> free from the *operator*: one fractional node already has power-law memory,
> whereas an ordinary leaky node has only exponential memory. See the README's
> "Why fractional?" section for the one-paragraph pitch.

---

<a name="part-2"></a>
## Part 2 — Reservoir computing foundations

### 2.1 The architecture (KB §1.1)

Reservoir Computing (Echo State Networks, Jaeger 2001; Liquid State Machines,
Maass 2002) is a recurrent-network idea with a radical simplification: **don't
train the recurrent weights at all.** Three pieces:

- **Input map** $\mathbf{W}_{in}$ — a fixed random projection of the drive into
  the reservoir.
- **Reservoir** $\mathbf{W}_{res}$ — a large random recurrent pool, *frozen* after
  initialisation. It acts as a high-dimensional non-linear memory of the input
  history.
- **Readout** $\mathbf{W}_{out}$ — a linear map from reservoir state to output.
  **This is the only trained part.**

In code these are, respectively, `FractionalReservoir.W_in`,
`FractionalReservoir.W_res`, and `TopologicalReadout.W_out`. A whole model
(`PhantomBrain`) is just `reservoir + readout`.

### 2.2 The Echo State Property (ESP) (KB §1.3)

For the reservoir to be a *function of the input history* (and not of forgotten
initial conditions) it must be **contracting**: nearby states must converge.
Standard ESN facts — note the careful wording, which v1 of the KB got wrong:

- **Necessary** (linearising about the origin, zero input): the spectral radius
  $\rho(\mathbf{W}_{res}) < 1$.
- **Sufficient** (a contraction, since $\tanh$ is 1-Lipschitz): the largest
  singular value $\sigma_{\max}(\mathbf{W}_{res}) < 1$. Conservative — many working
  reservoirs violate it.

In practice classical ESNs scale $\mathbf{W}_{res}$ to $\rho \approx 0.9$–$1.0$.
That is exactly what the `spectral_scale` argument does:
`W_res = normal(...) * spectral_scale / sqrt(res_size)`, which makes
$\rho(\mathbf{W}_{res}) \approx$ `spectral_scale`.

> **Crucial caveat for *this* project:** for the *fractional* reservoir the
> relevant stability statement is **not** the tanh-ESN bound at all — it is the
> **Matignon wedge** condition (Part 6). The spectral-radius heuristic is only a
> first approximation here.

### 2.3 The edge of chaos (KB §3.5)

Reservoirs compute best when poised between order (everything decays, no memory)
and chaos (sensitive dependence destroys the input signal). This "edge of chaos"
is where memory capacity peaks. We spend Part 6 making that precise for fractional
systems and giving a tool (`set_edge_of_chaos`) that puts a reservoir there.

### 2.4 The readout (KB §1.4)

Because only $\mathbf{W}_{out}$ is trained, training is a *linear* problem: collect
the reservoir states $\mathbf{X}$ over time, then solve a regression onto the
target. Two ways (Part 10): closed-form ridge (fast, default) or gradient descent
with the Besov penalty (when you want the topological prior).

**Run it:** `python examples/fit_ridge_readout.py`.
**Check your understanding:** Why does freezing $\mathbf{W}_{res}$ turn a hard
non-convex RNN-training problem into a linear-algebra one? What breaks if
$\rho(\mathbf{W}_{res}) \gg 1$?

---

<a name="part-3"></a>
## Part 3 — Fractional calculus for nodes

### 3.1 Two different α's — read this before anything else (KB §0)

The symbol α is dangerously overloaded. Keep these straight:

| Symbol | Meaning | Range | Enters via |
|---|---|---|---|
| $\alpha_D$ | order of the fractional **derivative** | $(0,1)$ | node dynamics, kernels |
| $\alpha_S$ | **stability index** of a heavy-tailed driving law | $(0,2]$ | Besov integrability bound |
| $H$ | **Hurst exponent** of the fractional Gaussian drive | $(0,1)$ | long-range dependence |

When the code says `alpha` (in `GLKernel`, `L1CaputoKernel`, `kernel.alpha`) it
**always** means $\alpha_D$. The Besov machinery's `alpha_stable` is $\alpha_S$
(and is `2.0` for our Gaussian drive). They are never interchangeable.

### 3.2 What a fractional derivative *is*, intuitively

An ordinary derivative is *local*: $\dot x(t)$ depends only on $x$ in an
infinitesimal neighbourhood of $t$. A **fractional** derivative $\mathcal D^{\alpha_D}$
with $0<\alpha_D<1$ is **non-local in time** — it is a weighted integral over the
*entire past*, with weights that decay as a **power law** $\sim (t-\tau)^{-\alpha_D}$
rather than exponentially. That slow, scale-free decay is precisely the long-range
dependence we want in the nodes.

As $\alpha_D \to 1$ the operator becomes the ordinary first derivative (memory
disappears); as $\alpha_D \to 0$ it approaches the identity (all past weighted
equally). So $\alpha_D$ is a **memory-depth dial**: smaller $\alpha_D$ = longer
memory.

### 3.3 The continuous node equation (KB §3.1)

A fractional reservoir node obeys

$$\mathcal{D}^{\alpha_D}_t \mathbf{x}(t) = -\mathbf{\Lambda}\mathbf{x}(t) + f\big(\mathbf{W}_{res}\mathbf{x}(t) + \mathbf{W}_{in}\mathbf{u}(t)\big)$$

with $\mathbf{\Lambda}=\mathrm{diag}(\lambda_i)$ the node **decay** (leak) and
$f=\tanh$. **The $-\mathbf{\Lambda}\mathbf{x}$ leak is essential** — drop it and
the discretised system diverges (this was a real bug in the v1 pseudocode). We
confirmed empirically that the leak is load-bearing: with $\lambda=0$ the model's
memory-task recall collapses to ~0 (Part 13).

### 3.4 Two discretisation schemes

We need to turn $\mathcal{D}^{\alpha_D}$ into something computable on a uniform
grid $t_k = k h$. Two standard finite-difference schemes are implemented, both as
subclasses of `AbstractFractionalKernel`:

**Grünwald–Letnikov (GL)** — `GLKernel` (KB §3.2). Generalised-binomial
coefficients
$$c_0 = 1,\qquad c_j = \Big(1 - \tfrac{1+\alpha_D}{j}\Big)\,c_{j-1}.$$
Accuracy $O(h)$.

**L1–Caputo** — `L1CaputoKernel` (KB §3.2b), the package default for brain models.
Built from weights $b_m = (m+1)^{1-\alpha_D} - m^{1-\alpha_D}$. Accuracy
$O(h^{2-\alpha_D})$ — better than GL, and it handles non-zero initial values
correctly (the Caputo vs Riemann–Liouville distinction).

```python
from fracres import GLKernel, L1CaputoKernel
gl = GLKernel(alpha=0.8, history_length=100)          # alpha is alpha_D
l1 = L1CaputoKernel(alpha=0.75, history_length=200)   # NB: no dt argument
```

Each kernel precomputes three quantities the reservoir consumes — `leading`,
`weights`, `forcing_factor` — which Part 4 explains.

**Run it:** `python examples/validate_kernels.py` (checks both schemes against the
exact $\mathcal D^{\alpha_D} t^\beta$).
**Check your understanding:** Why does *smaller* $\alpha_D$ mean *longer* memory?
Which scheme would you pick for a signal that starts far from zero, and why?

---

<a name="part-4"></a>
## Part 4 — The discrete update: from operator to recurrence

This part is the conceptual heart of the implementation. Read it slowly.

### 4.1 The gain-normalised one-step update

Plugging a GL/L1 difference into the node equation and solving for the newest
sample gives the recurrence the reservoir actually runs (KB §3.2, §6.1):

$$x_k = \underbrace{\text{leading}\cdot x_{k-1}}_{\text{retain latest state}} \;-\; \underbrace{\sum_{j=2}^{L} w_j\,x_{k-j}}_{\text{strict-past memory}} \;+\; \underbrace{\text{forcing\_factor}\cdot h^{\alpha_D}\,g_k}_{\text{scaled drive}}$$

where the drive supplied by the reservoir is
$$g_k = -\lambda\,x_{k-1} + f(\mathbf W_{res}x_{k-1} + \mathbf W_{in}u_k).$$

So each kernel exposes exactly three numbers (`AbstractFractionalKernel` fields):

- **`leading`** — coefficient on the most recent state $x_{k-1}$;
- **`weights`** — the strict-past coefficients $w_2,\dots,w_L$ (length $L-1$);
- **`forcing_factor`** — a dimensionless scale on the (already $h^{\alpha_D}$-scaled)
  drive.

For GL: `leading = α_D`, `weights = c_2..c_L`, `forcing_factor = 1`.
For L1: `leading = 1 - b_1`, `weights = b_j - b_{j-1}`, `forcing_factor = Γ(2-α_D)`.

### 4.2 The leading-coefficient subtlety (a documented departure)

This is the single most important implementation detail, and `kernels.py`'s module
docstring is worth reading in full. The *naive* update
`x_{k-1} - memory + activation` (leading coefficient 1, no $h^{\alpha_D}$ scaling)
**double-counts** $x_{k-1}$: the effective linear gain on the leading state becomes
$1 + \alpha_D > 1$, and the reservoir diverges.

The fix: set `leading = -c_1 = α_D` (GL). Now the *linear* memory gain is exactly
1, and the decay term $-\lambda x_{k-1}$ pulls it **strictly below 1** — recovering
the Echo State Property. This is a deliberate, documented departure from the literal
coefficients in KB §3.2, and the reason the package converges where the v1
pseudocode did not.

### 4.3 The effective leading gain (a useful lens)

Collecting the $x_{k-1}$ terms, the **effective** linear gain on the most recent
state is

$$g_{\text{eff}} = \text{leading} - \text{forcing\_factor}\cdot h^{\alpha_D}\cdot\lambda.$$

This single quantity is a handy stability/operating-point summary, and Part 13's
study uses it directly. Note the trade-off it exposes: `h` (`step_size`) scales the
*entire* drive block (leak **and** input), while `λ` (`decay`) scales *only the
leak* — so the two are **not** redundant.

### 4.4 Two views of the same kernel

The kernel can be used in two distinct ways, and keeping them separate prevents
confusion:

| View | Method | What it does | Used by |
|---|---|---|---|
| **Recurrence** | `kernel.__call__(x_history)` | returns just the strict-past memory $\sum w_j x_{k-j}$, advanced one step over a rolling buffer | the reservoir scan |
| **Operator** | `kernel.apply(signal, h)` | reconstructs the *full* FIR filter $a=[1,-\text{leading},w_2,\dots]$ and estimates $\mathcal D^{\alpha_D}$ of a whole signal | the validation suite |

The recurrence view is what makes this efficient inside `jax.lax.scan`; the
operator view is what we can check against the analytic $\mathcal D^{\alpha_D} t^\beta$
(Part 12). The KB calls these two "operator views"; the same weight math underlies
both.

**Check your understanding:** Starting from $x_k = \text{leading}\,x_{k-1} - \sum w_j x_{k-j} + \dots$,
show that with `leading = α_D` and $\lambda>0$ the linear gain on $x_{k-1}$ is
below 1. What happens to `g_eff` as you raise `h`?

---

<a name="part-5"></a>
## Part 5 — From one node to a brain: the four reservoir variants

All four reservoirs share the *same* validated GL/L1 update of Part 4 (same
`leading`/`weights`/`forcing_factor`, which depend only on $\alpha_D$). They differ
only in **what plays the role of the drive $g_k$** and **what the connectivity
means**. This is the payoff of the operator/recurrence separation: one numerical
core, four neuroscience models.

### 5.1 `FractionalReservoir` — the base single-population model

The reduction described in Part 4: random connectome $\mathbf W_{res}$, $\tanh$
activation (zero-centred, avoids DC drift — *not* the sigmoid of the v1
pseudocode), scalar decay $\lambda$. This is the workhorse; `PhantomBrain` wraps it.

### 5.2 `WilsonCowanReservoir` — excitatory/inhibitory neural mass (KB §2.2)

$N$ coupled Wilson–Cowan (1972) masses, each split into excitatory $E$ and
inhibitory $I$ subpopulations sharing $\alpha_D$ but with **separate** time
constants $\tau_E, \tau_I$:

$$\tau_E^{\alpha_D}\mathcal D^{\alpha_D} E = -E + \mathcal S_E(W_{EE}E - W_{EI}I + W_{in}u), \qquad \tau_I^{\alpha_D}\mathcal D^{\alpha_D} I = -I + \mathcal S_I(W_{IE}E - W_{II}I).$$

Implementation insights worth internalising:

- Dividing each line by $\tau_x^{\alpha_D}$ turns it into `D^α x = g` with
  $g_x = (-x + \mathcal S(\text{syn}))/\tau_x^{\alpha_D}$ — so **$1/\tau_x^{\alpha_D}$
  plays the role of the decay $\lambda$**, scaling both leak and firing-rate drive.
- The two populations are **stacked** into one state $z=[E;I]$ of width $2N$ and
  advanced with the *single* shared update.
- Connectomes hold **non-negative** synaptic magnitudes (`jnp.abs(...)`); the E/I
  **signs are explicit** in the equations ($-W_{EI}$, $-W_{II}$).
- Firing rate is a **logistic sigmoid** (rates are non-negative — the WC standard),
  not the $\tanh$ of the single-pop reduction; the $-x$ leak keeps the bounded
  positive drive from drifting.

### 5.3 `NeuralFieldReservoir` — spatial Amari field (KB §2.2)

A continuum cortical *sheet*, discretised as $N$ sites on a 1-D periodic **ring**.
The recurrent connectivity is no longer random links but a distance-dependent
**spatial kernel** $W_{ij}=w(d_{ij})$. The Amari (1977) field equation puts the
non-linearity **inside** the spatial convolution — $w * \mathcal S(u)$ — the
defining signature of a field model (contrast $f(\mathbf W_{res}x + \dots)$ where
$f$ wraps the whole sum):

$$\tau^{\alpha_D}\mathcal D^{\alpha_D} u(x,t) = -u(x,t) + [w * \mathcal S(u)](x,t) + W_{in}u_{ext}.$$

The default kernel is a **Mexican hat** (`mexican_hat_kernel`): short-range
excitation minus longer-range inhibition (difference of Gaussians on the
`ring_distance`). Choosing $A_e\sigma_e \approx A_i\sigma_i$ makes the DC
(zero-wavenumber) gain vanish, so the field selects a **spatial pattern** at a
preferred wavelength rather than a uniform mode — the field analogue of the edge of
chaos.

### 5.4 `qSOCFractionalReservoir` — homeostatic criticality

The base reservoir plus a self-regulating threshold that pulls macroscopic energy
toward a critical set-point. This gets its own part (Part 7).

Each reservoir has a matching `*PhantomBrain` model (`PhantomBrain`,
`WilsonCowanPhantomBrain`, `NeuralFieldPhantomBrain`, `qSOCPhantomBrain`) that adds
the readout and a `simulate()` returning `(X_states, Y_hat, ...)`. Because they all
return `(X_states, Y_hat)` first, they drop into the same training utilities
unchanged.

**Run it:** `python examples/wilson_cowan.py`, `python examples/neural_field.py`.
**Check your understanding:** In the Wilson–Cowan model, why are synaptic
magnitudes stored as `abs(...)` while the signs live in the update? What is the
difference between $f(\mathbf W x)$ and $\mathbf W f(x)$, and why does it matter for
a *field* model?

---

<a name="part-6"></a>
## Part 6 — Stability: Matignon's theorem and the fractional edge of chaos

### 6.1 Why the ESN bound is the wrong tool here (KB §3.4)

For the linearised autonomous system $\mathcal D^{\alpha_D}\mathbf x = A\mathbf x$
with $A = \mathbf W_{res} - \mathbf\Lambda$ (the Jacobian of the node equation at
the origin, using $\tanh'(0)=1$), the right stability criterion is **Matignon's
theorem** (1996):

$$\text{asymptotically stable} \iff |\arg(\lambda_i(A))| > \frac{\alpha_D\,\pi}{2}\quad\text{for all } i.$$

The **unstable set** is a cone (wedge) of half-angle $\alpha_D\pi/2$ about the
**positive real axis**. Picture it:

- As $\alpha_D \to 1$ the wedge fills the entire right half-plane — you recover the
  classical condition $\mathrm{Re}(\lambda) < 0$.
- For $\alpha_D < 1$ the wedge *shrinks*, so the **stable region is larger**. A
  fractional reservoir can sit near criticality with a "hotter" connectome than a
  classical ESN could tolerate.

Verified thresholds: $\alpha_D=0.5\to45°$, $0.7\to63°$, $0.9\to81°$, $1.0\to90°$.

### 6.2 The tools (`stability.py`)

- `matignon_diagnostics(model)` → a `MatignonDiagnostics` namedtuple: the wedge
  threshold, the worst-case eigenvalue argument `min_arg`, the `margin`
  (`> 0 ⇒ stable`), plus the ESP diagnostics $\rho(\mathbf W_{res})$ and
  $\sigma_{\max}(\mathbf W_{res})$. Its `critical_alpha` property tells you the
  largest $\alpha_D$ this connectome could tolerate.
- `set_spectral_radius(model, ρ)` — the *classical* control (scale $\mathbf W_{res}$
  to a target spectral radius).
- `set_edge_of_chaos(model, safety_factor=0.95)` — the *Matignon-aware* control.
  It finds, by bisection, the scale `s_edge` that puts the least-stable mode exactly
  on the wedge boundary, then backs off by `safety_factor` (`<1` stable, `=1`
  marginal, `>1` deliberately unstable).

The bisection exploits a neat identity: because $\mathbf I$ shares every eigenvector
of $\mathbf W_{res}$, the eigenvalues of $s\mathbf W_{res} - \lambda\mathbf I$ are
exactly $s\mu_i - \lambda$ (with $\mu_i = \mathrm{eig}(\mathbf W_{res})$) — so the
stability predicate is *monotone in $s$* and needs no re-eigendecomposition.

**Run it:** `python examples/matignon_control.py`.
**Check your understanding:** For a fixed connectome, does *lowering* $\alpha_D$
make the system more or less stable? Why can a fractional reservoir use a larger
connectome scale than a classical ESN?

---

<a name="part-7"></a>
## Part 7 — Self-organised criticality: the qSOC controller

### 7.1 The idea (KB §5)

Real cortex appears to *self-tune* toward criticality (homeostasis). The
`qSOCFractionalReservoir` adds a slow feedback loop: an adaptive threshold $b_t$
(added inside the activation) that nudges the network's macroscopic **energy**
toward a target set-point $E_{crit}$. Two extra scalar states carried through the
scan:

$$\tau_{soc}\,\dot E = \lVert x\rVert^2 - E \quad\text{(leaky-integrator energy window)},$$
$$\tau_b\,\dot b = -b + \gamma\,(E_{crit} - E) \quad\text{(homeostatic threshold)}.$$

If energy runs high, $b$ is pushed down, lowering excitability; if low, $b$ rises.
The threshold enters as `tanh(W_res x + W_in u + b_t)`.

### 7.2 Two numerical-stability details worth learning from

- The **energy window** $E$ uses a plain explicit-Euler low-pass — fine, because a
  first-order LPF is unconditionally stable. Pick $\tau_{soc}$ deliberately relative
  to `dt`.
- The **threshold** $b$ uses a **semi-implicit** Euler step,
  $b \leftarrow \big(b + \tfrac{dt}{\tau_b}\gamma(E_{crit}-E)\big)\big/(1+\tfrac{dt}{\tau_b})$,
  which is **unconditionally stable** and avoids the explicit step's $dt < 2\tau_b$
  restriction. (This was a ROADMAP open question — now resolved. See ROADMAP and
  KB §5.1.)
- The implementation also fixed a v1 inconsistency: the controller now *uses* the
  windowed energy $E$ it defines (v1 defined a window but used the instantaneous
  $\lVert x\rVert^2$).

`qSOCPhantomBrain.simulate()` returns `(X_states, Y_hat, B_thresholds, E_energy)`
so you can watch the controller work.

**Run it:** `python examples/run_qsoc_simulation.py`.
**Check your understanding:** Why is a semi-implicit step "unconditionally stable"
where explicit Euler needs $dt<2\tau_b$? What would happen to the avalanche
statistics (Part 11) if you set $E_{crit}$ too high?

---

<a name="part-8"></a>
## Part 8 — The driving noise: fractional Gaussian noise

### 8.1 Why heavy-tailed, long-memory input?

To produce brain-like long-range dependence the *drive* itself should carry it.
`generate_fbm_increments(time_steps, H, key)` produces **fractional Gaussian noise**
(fGn) — the increments of fractional Brownian motion — with a tunable **Hurst
exponent** $H$:

- $H = 1/2$ → ordinary white noise (no memory);
- $H > 1/2$ → **persistent** (long-range positive correlation);
- $H < 1/2$ → anti-persistent.

The exact autocovariance is $r(k) = \tfrac12(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})$,
$r(0)=1$.

### 8.2 Exact synthesis by circulant embedding (Davies–Harte)

The synthesis is exact (not approximate) and $O(n\log n)$: embed the fGn
autocovariance in a circulant matrix, diagonalise it with an FFT, weight a complex
Gaussian spectrum by $\sqrt{\text{eigenvalues}}$, and transform back.

**A subtlety the code documents carefully** (and a recently-resolved open question):
the output is scaled by the *deterministic* Davies–Harte factor $\sqrt m$ ($m$ the
embedding length), **not** divided by the per-realisation empirical standard
deviation. Dividing by the empirical std forces every draw to have std exactly 1,
but that rescaling is correlated with the sample (worst at large $H$) and **flattens
the very long-range dependence the drive exists to inject**. With the exact scaling
the process has unit variance *in expectation* and the *exact* autocovariance, and
individual draws genuinely vary about 1 — which is correct. `tests/test_drivers.py`
checks the realised ensemble autocovariance against the analytic target for
$H\in\{0.3,0.5,0.7,0.9\}$.

**Run it:** `python examples/run_fbm_driven.py`.
**Check your understanding:** Why would normalising each draw to std 1 *bias* the
estimated long-range dependence? What does $H=1/2$ give, and why is that a good
sanity check?

---

<a name="part-9"></a>
## Part 9 — Function-space regularisation: Besov & Littlewood–Paley

### 9.1 The goal: control the *regularity* of the output (KB §4)

Brain signals are not arbitrarily rough or smooth; they live on a particular
**regularity manifold**. We encode that as a penalty on the **Besov norm**
$\lVert y\rVert_{B^s_{p,q}}$ of the generated trajectory, where $s$ is smoothness,
$p$ integrability, $q$ summability.

### 9.2 The strict bounds — the central claim, stated correctly (KB §4.2)

These bounds (and the careful distinction of which α is which) are the corrected
heart of the function-space story, implemented in `training.besov_indices`:

$$p < \alpha_S \qquad\text{(heavy-tail integrability; } \alpha_S = 2 \text{ for Gaussian fGn)},$$
$$s < \min\{H,\ 1/p\} \qquad\text{(fBm smoothness } \cap \text{ Besov embedding scale)},$$
$$q = p.$$

The code derives them with a small `margin` to stay strictly inside:
`p = alpha_stable - margin`, `s = min(H, 1/p) - margin`, `q = p`. **Note `p` is
bounded by $\alpha_S$ (the *drive's* stability index), not by $\alpha_D$.** This is
the exact point v1 got wrong, and the reason §0's two-α warning exists.

### 9.3 Why Littlewood–Paley (the efficiency trick) (KB §4.3)

Evaluating a Besov norm directly (the Gagliardo double integral) is $O(n^2)$.
Instead we use the equivalent **Littlewood–Paley characterisation**: split the
signal into **dyadic frequency bands** (band $j$ keeps FFT bins
$2^{j-1}\le|k|<2^j$), take the $L^p$ size of each band $\lVert\Delta_j Y\rVert_p$,
weight by $2^{js}$, and aggregate in $\ell^q$:

$$\lVert Y\rVert_{B^s_{p,q}} \approx \Big(\sum_j \big(2^{js}\lVert\Delta_j Y\rVert_p\big)^q\Big)^{1/q}.$$

All FFT-based, $O(n\log n)$. In code: `make_dyadic_masks` (the band-pass masks),
`dyadic_band_energies` (the per-band $L^p$ norms), `littlewood_paley_penalty` (the
weighted aggregate). For $s>0$ the $2^{js}$ weight grows with band index, so
**high-frequency content is penalised more** — pushing the output toward the
target smoothness.

> **Bonus diagnostic.** Because a signal of Besov smoothness $r$ has band energies
> decaying like $2^{-jr}$, the slope $-\,d\log_2\lVert\Delta_j Y\rVert_p/dj$
> *measures* the realised regularity — a direct way to check the regulariser is
> doing what you asked.

**Run it:** `python examples/besov_regularization.py`.
**Check your understanding:** If you increase $s$, do you expect the output
smoother or rougher? Why is `p` tied to the *drive's* tail index and not the
derivative order?

---

<a name="part-10"></a>
## Part 10 — Training the readout

Only $\mathbf W_{out}$ is ever trained. There are two routes (KB §6.2–6.3).

### 10.1 Closed-form ridge (the fast default)

`fit_ridge_readout(X, Y, beta, washout)` solves the Tikhonov problem in closed form,
$\mathbf W_{out}^\top = (\mathbf X^\top\mathbf X + \beta\mathbf I)^{-1}\mathbf X^\top\mathbf Y$,
discarding `washout` initial transient steps. `fit_readout_ridge(model, U, Y, ...)`
is the convenience wrapper: simulate the frozen reservoir, solve, and splice the
result back in via `equinox.tree_at`. Works for every `*PhantomBrain` (extra
`simulate` kwargs like the qSOC `dt` are forwarded).

### 10.2 Gradient descent with the Besov prior

When you want the topological regulariser, `train_step` does one optax step on the
loss `MSE + lambda_reg * littlewood_paley_penalty(...)`. The **freezing is enforced
structurally**, not by convention: `readout_filter_spec(model)` builds a boolean
pytree that is `True` only at `model.readout.W_out`; `equinox.partition` splits the
model so gradients touch the readout *only*. The reservoir and kernel weights can
never receive an update.

> **Why two routes?** Ridge is faster and more stable and is the right default when
> you don't need the regularity prior; gradient descent earns its cost only when the
> Besov penalty matters.

**Run it:** `python examples/train_readout.py` (gradient + Besov),
`python examples/fit_ridge_readout.py` (closed form).
**Check your understanding:** What does `washout` protect against? How does
`equinox.partition` guarantee the reservoir stays frozen?

---

<a name="part-11"></a>
## Part 11 — Measuring what you made: LRD & avalanche metrics

A generated trajectory is only interesting if it has the right statistics.
`metrics.py` provides two families (KB §4 and criticality).

### 11.1 Long-range dependence

- `hurst_dfa(x)` — **detrended fluctuation analysis**: the slope of
  $\log F(s)$ vs $\log s$, which equals $H$ for fGn. (The window bounds avoid DFA's
  small-scale crossover and high-variance large scales.)
- `spectral_exponent(x)` — the log-periodogram slope $\beta$ where
  $\mathrm{PSD}\sim f^{-\beta}$. For fGn, $\beta = 2H-1$. (Fit restricted to low
  frequencies — fGn is a power law only as $f\to0$ — and log-binned to tame
  variance.)
- `signal_metrics(x)` bundles both and maps the spectral estimate back onto the
  Hurst scale via $H=(\beta+1)/2$, so the two estimators (and the drive's known $H$)
  are directly comparable.

Cross-check table: $H=1/2$ → DFA $0.5$, $\beta=0$; $H=0.7$ → DFA $0.7$, $\beta=0.4$.

### 11.2 Criticality / neuronal avalanches

Near a critical point, event sizes and durations are power-law distributed (Beggs &
Plenz 2003).

- `detect_avalanches(activity)` — segment a non-negative trace into
  supra-threshold excursions (default threshold = median), returning each
  avalanche's size (summed excess) and duration.
- `power_law_exponent(values)` — the Clauset–Shalizi–Newman maximum-likelihood
  exponent $\mu = 1 + n/\sum_i\ln(x_i/x_{min})$.
- `avalanche_exponents(activity)` → `(tau, alpha, n_avalanches)`. **Critical cortex
  signature: $\tau\approx1.5$ (sizes), $\alpha\approx2.0$ (durations).** (Here
  `alpha` is the duration exponent — yet another α; not $\alpha_D$.)

**Run it:** `python examples/signal_metrics.py`.
**Check your understanding:** If you drive the reservoir with $H=0.8$ noise, what
DFA exponent should the *output* have if the reservoir preserved the LRD? What does
$\tau\approx1.5$ tell you about the operating point?

---

<a name="part-12"></a>
## Part 12 — Validating the kernels against ground truth

How do we know the fractional machinery is *correct*, not just plausible?
`validation.py` provides analytic references (KB §7).

- **Power-law derivative.** The fractional derivative of a power law has a closed
  form: $\mathcal D^{\alpha_D} t^\beta = \dfrac{\Gamma(\beta+1)}{\Gamma(\beta+1-\alpha_D)}\,t^{\beta-\alpha_D}$
  (`analytic_power_law_derivative`). The kernel's *operator view* (`kernel.apply`)
  is checked against this.
- **Convergence order.** `convergence_order(err_coarse, err_fine)` recovers the
  empirical order $p$ from errors at two grid spacings. The suite confirms the
  textbook rates: **GL is $O(h)$, L1 is $O(h^{2-\alpha_D})$.**
- **Mittag–Leffler function.** $E_{\alpha_D}(z)=\sum_k z^k/\Gamma(\alpha_D k+1)$
  (`mittag_leffler`) is the **eigenfunction of the Caputo derivative**:
  $x(t)=E_{\alpha_D}(\lambda t^{\alpha_D})$ solves $\mathcal D^{\alpha_D}_C x=\lambda x$,
  $x(0)=1$. It validates the kernels on a signal with non-zero initial value, which
  exercises the Caputo-vs-Riemann–Liouville distinction the power law cannot.

There is also `tests/test_hpfracc_crossref.py`, an *optional* cross-validation
against the separate `hpfracc` library (skipped unless installed) — the project
deliberately **cross-validates rather than couples** to it (see ROADMAP for the
reasoning).

**Run it:** `python examples/validate_kernels.py`; `pytest tests/test_validation.py`.
**Check your understanding:** Why is the Mittag–Leffler test stronger than the
power-law test for distinguishing Caputo from Riemann–Liouville? What empirical
convergence order would falsify the L1 implementation?

---

<a name="part-13"></a>
## Part 13 — Putting it together: the config system & a worked study

### 13.1 Reproducible experiments (`config.py`)

A whole experiment — kernel, model variant, drive, training — is captured by a
small tree of dataclasses (`ExperimentConfig` = `KernelConfig` + `ModelConfig` +
`DriveConfig` + `TrainingConfig`) that round-trips to a single YAML file. Variant-
specific reservoir kwargs go in `ModelConfig.params` and are forwarded to the
constructor. Factories (`build_kernel`, `build_model`, `build_drive`,
`build_experiment`) turn a spec into live objects, so a run is fully determined by
its config plus the integer `seed`.

```python
from fracres import load_config, build_experiment, fit_readout_ridge
config = load_config("configs/memory_task.yaml")
model, drive = build_experiment(config)        # deterministic in config.seed
```

**Run it:** `python examples/config_experiment.py`.

### 13.2 A worked example: the `h`/`λ` study (ROADMAP Open-Question #1)

The defaults `step_size` ($h$) and `decay` ($\lambda$) were long pinned at
conservative values $(0.1, 1.0)$. A sweep on the delayed-copy memory task (GL
$\alpha_D=0.8$, $N=300$, 4 seeds) showed three things — a good template for how to
interrogate this kind of model:

1. **The leak is load-bearing.** $\lambda=0$ gives ~0 recall at *every* $h$: with no
   state-dependent dissipation there is no usable fading memory. `decay` is not a
   minor trim (cf. Part 3.3).
2. **$h$ and $\lambda$ are not degenerate.** The effective gain
   $g_{\text{eff}} = \text{leading} - h^{\alpha_D}\lambda$ (Part 4.3) only *loosely*
   organises recall (correlation $\approx -0.68$): lower $g_{\text{eff}}$ tends to
   help, but it is not a single-variable collapse — because $h$ also scales the
   input drive while $\lambda$ does not.
3. **There is a stability edge.** Recall peaks on the low-$g_{\text{eff}}$ side and
   the reservoir **diverges** once $g_{\text{eff}}$ goes strongly negative
   ($h=0.4,\lambda=4\Rightarrow g_{\text{eff}}=-1.12$, NaN).

Net result: held-out recall rose from **0.64** to **0.95** simply by un-pinning
these defaults to $(h,\lambda)=(0.4, 2.0)$, now set in `configs/memory_task.yaml`.
The lesson generalises: every "magic default" in this codebase is a hypothesis you
can test with one sweep.

**Check your understanding:** Using $g_{\text{eff}}$, predict the sign of the change
in recall if you halve $h$ at fixed $\lambda$. Why can't $\lambda$ fully substitute
for $h$?

---

<a name="appendix-a"></a>
## Appendix A — Symbol glossary

| Symbol | Code | Meaning |
|---|---|---|
| $\alpha_D$ | `alpha`, `kernel.alpha` | fractional-derivative order, $(0,1)$ — the memory dial |
| $\alpha_S$ | `alpha_stable` | heavy-tail stability index of the drive ($=2$ for Gaussian) |
| $H$ | `hurst`, `H` | Hurst exponent of the fGn drive |
| $h$ | `step_size` | reservoir integration step |
| $\lambda$ | `decay` | node leak rate |
| $L$ | `history_length` | truncated memory-window length |
| $\rho$ | `spectral_radius` | spectral radius of $\mathbf W_{res}$ (ESP) |
| $\sigma_{\max}$ | `max_singular_value` | largest singular value (sufficient ESP) |
| `leading` | `kernel.leading` | coefficient on $x_{k-1}$ ($=\alpha_D$ for GL) |
| `forcing_factor` | `kernel.forcing_factor` | drive scale ($1$ for GL, $\Gamma(2-\alpha_D)$ for L1) |
| $g_{\text{eff}}$ | — | effective leading gain $\text{leading}-h^{\alpha_D}\lambda$ |
| $s,p,q$ | `s,p,q` | Besov smoothness / integrability / summability |
| $\tau,\beta,\mu$ | — | avalanche duration exp / spectral exp / power-law exp |

---

<a name="appendix-b"></a>
## Appendix B — Theory → code map

| Concept (this manual) | Module / function | KB section |
|---|---|---|
| Reservoir + readout | `reservoirs.FractionalReservoir`, `readout.TopologicalReadout` | §1 |
| GL / L1 kernels | `kernels.GLKernel`, `kernels.L1CaputoKernel` | §3.2 |
| Discrete update / recurrence | `FractionalReservoir.__call__` | §3.2, §6.1 |
| Wilson–Cowan E/I | `reservoirs.WilsonCowanReservoir` | §2.2 |
| Amari neural field | `reservoirs.NeuralFieldReservoir`, `mexican_hat_kernel` | §2.2 |
| Matignon stability | `stability.matignon_diagnostics`, `set_edge_of_chaos` | §3.4 |
| qSOC homeostasis | `reservoirs.qSOCFractionalReservoir` | §5 |
| fGn drive | `drivers.generate_fbm_increments` | §4 (drive) |
| Besov / Littlewood–Paley | `regularizers.littlewood_paley_penalty` | §4.3 |
| Readout training | `training.fit_ridge_readout`, `training.train_step` | §6.2–6.3 |
| LRD & avalanche metrics | `metrics.signal_metrics`, `metrics.avalanche_exponents` | §4, criticality |
| Kernel validation | `validation.analytic_power_law_derivative`, `mittag_leffler` | §7 |
| Reproducible configs | `config.ExperimentConfig`, `build_experiment` | §6 |

---

<a name="appendix-c"></a>
## Appendix C — Suggested exercises

1. **Memory dial.** Fix everything but $\alpha_D$ and sweep it from 0.3 to 0.95.
   Plot output `hurst_dfa` vs $\alpha_D$. Does longer node memory (smaller
   $\alpha_D$) yield stronger output LRD?
2. **Find the edge.** For a random connectome, use `matignon_diagnostics` to read
   off the margin, then `set_edge_of_chaos` with `safety_factor` $\in\{0.8,1.0,1.1\}$
   and observe stable / marginal / divergent behaviour.
3. **Reproduce the `h`/λ result.** Recreate Part 13's sweep with the config system
   and confirm the 0.64 → 0.95 jump. Then add a third axis: does it survive at
   $\alpha_D=0.5$?
4. **Criticality from homeostasis.** Drive `qSOCPhantomBrain` and vary $E_{crit}$;
   track `avalanche_exponents` of the output. Can you find a set-point giving
   $\tau\approx1.5$?
5. **Trust but verify.** Add a new analytic test case to the validation suite (e.g.
   $\mathcal D^{\alpha_D}$ of a different power law) and confirm the GL / L1
   convergence orders.

---

*This manual tracks the code as of the `step_size`/`decay` default study (ROADMAP
Open-Question #1). When the implementation moves, prefer the docstrings and
`docs/knowledge_base_v2.md` as the source of truth.*
