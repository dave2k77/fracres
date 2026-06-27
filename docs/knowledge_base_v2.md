# Knowledge Base v2: Fractional Reservoir Computing for Critical Brain Dynamics
**Theoretical Framework, Function Spaces, and High-Performance Neural-Mass Models**

> **Status / provenance.** This is a corrected and extended revision of
> `docs/knowledge_base.md`, reconciled against the working `fracres` source tree
> (`src/fracres/{kernels,reservoirs,models,regularizers,drivers,training}.py`) as
> of 2026-06-27. It was authored as a *separate* file to avoid clobbering
> concurrent edits; merge into the canonical KB when convenient. Sections marked
> **⚠ CORRECTION** change a substantive claim in v1; sections marked **➕ NEW**
> are additions aimed at a genuinely high-performance phantom-brain model.
>
> The single most important fix: the v1 discrete update (§3.2) and the v1
> Section 6 code are **numerically divergent** — the leading coefficient on
> `x_{k-1}` must be `α`, not `1`. The source code already encodes the corrected,
> gain-normalised form; this document now matches it.

---

## 0. Notation — and a warning about two different α's ➕ NEW

A recurring source of confusion in v1 is that the symbol α is overloaded. They
are **distinct quantities** and must not be cross-substituted:

| Symbol | Meaning | Range | Where it enters |
|---|---|---|---|
| $\alpha_D$ | Order of the fractional (Caputo/GL) **derivative** | $(0,1)$ | Node dynamics, kernels (§3) |
| $\alpha_S$ | **Stability index** of an α-stable *heavy-tailed* driving law | $(0,2]$ | Integrability of the drive (§4.2) |
| $H$ | **Hurst exponent** of the fractional Gaussian drive | $(0,1)$ | Long-range dependence of the drive (§4) |
| $\rho(\mathbf W)$ | Spectral radius (largest $|\lambda|$) of a matrix | — | Echo State Property (§1.3) |
| $\sigma_{\max}(\mathbf W)$ | Largest singular value | — | Sufficient ESP condition (§1.3) |
| $s,p,q$ | Besov smoothness / integrability / summability indices | $s\in\mathbb R$, $p,q\in(0,\infty]$ | Regulariser (§4) |
| $h$ | Reservoir integration step | $>0$ | Discretisation (§3.2) |
| $L$ | Truncated memory-window length | $\in\mathbb N$ | Short-memory principle (§3.3) |

Where v1 wrote "$p<\alpha$ (order of the fractional derivative)" it should have
referred to $\alpha_S$, the heavy-tail stability index — see §4.2.

---

## 1. Fundamentals of Reservoir Computing (RC)

Reservoir Computing is a recurrent-network framework — Echo State Networks
(ESNs, Jaeger 2001) and Liquid State Machines (LSMs, Maass et al. 2002) — that
processes temporal data without training the recurrent weights by
backpropagation through time. Only a linear readout is fitted.

### 1.1 Core Architecture
1. **Input layer:** projects the driving signal into the reservoir via a fixed
   random $\mathbf{W}_{in}$.
2. **Reservoir:** a large, sparse, randomly connected pool of non-linear nodes
   $\mathbf{W}_{res}$, initialised once and **frozen**. It acts as a
   high-dimensional temporal kernel over the input history.
3. **Readout:** a linear map $\mathbf{W}_{out}$ — **the only trained part.**

### 1.2 Mathematical Formulation
For a leaky-integrator ESN with leak rate $\alpha_{\text{leak}}\in(0,1]$:

$$\mathbf{x}(t) = (1 - \alpha_{\text{leak}})\mathbf{x}(t-1) + \alpha_{\text{leak}}\, f\big(\mathbf{W}_{in}\mathbf{u}(t) + \mathbf{W}_{res}\mathbf{x}(t-1) + \mathbf{b}\big)$$

$$\mathbf{y}(t) = \mathbf{W}_{out} [\mathbf{x}(t); \mathbf{u}(t)]$$

with $f=\tanh$ typically. (Note: $\alpha_{\text{leak}}$ here is the *leak rate*,
yet another distinct α; we keep the subscript to disambiguate from $\alpha_D$.)

### 1.3 The Echo State Property (ESP) ⚠ CORRECTION
The ESP requires the reservoir's asymptotic state to depend only on input
history, forgetting initial conditions. v1 stated "$\rho(\mathbf W_{res})<1$ is a
**sufficient** condition." That is **incorrect**. The standard results
(Jaeger 2001; Yildiz, Jaeger & Kiebel 2012) are:

- **Necessary (for zero input, linearisation about the origin):**
  $\rho(\mathbf{W}_{res}) < 1$. If $\rho\ge 1$ the autonomous system is generally
  not contracting and the ESP fails.
- **Sufficient (a contraction condition):** $\sigma_{\max}(\mathbf{W}_{res}) < 1$
  (largest singular value), since $\tanh$ is 1-Lipschitz. This is conservative;
  many ESP networks have $\sigma_{\max}>1$.

In practice one scales to $\rho \approx 0.9$–$1.0$ (the "edge of chaos") because
$\rho$ is the more informative diagnostic, while acknowledging neither bound is
tight. For the **fractional** reservoir the relevant stability statement is
*not* the tanh-ESN bound at all but the Matignon wedge condition — see §3.4.

### 1.4 Linear Readout Training
With internal weights fixed, stack states into $\mathbf{X}$ and solve ridge
(Tikhonov) regression in closed form:

$$\mathbf{W}_{out} = \mathbf{Y}_{target} \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \beta \mathbf{I})^{-1}$$

This closed form is *not yet in the source* (it is a ROADMAP "Now" item); the
current `training.py` fits the readout by gradient descent with a Besov penalty.
A reference implementation is given in §6.3.

---

## 2. Biological Analogues: From Spiking to Mean-Field Models

### 2.1 Liquid State Machines (LSMs)
LSMs use spiking neurons (typically Leaky Integrate-and-Fire) as the reservoir;
an input spike train produces high-dimensional transient spatio-temporal
patterns, read out from low-pass-filtered spike trains or membrane potentials.

### 2.2 Neural Mass and Field Models (NMMs / NFMs)
To model macroscopic signals (EEG/MEG/fMRI) at population level we embed
continuous neural-mass models (Wilson–Cowan 1972) or neural fields
(Amari 1977) into the reservoir. A fractional excitatory/inhibitory neural-mass
reservoir of $N$ coupled masses:

$$\tau_E^{\alpha_D} \mathcal{D}^{\alpha_D}_t E_i = -E_i + \mathcal{S}_E\!\left( \sum_j W_{ij}^{EE} E_j - \sum_j W_{ij}^{EI} I_j + W_{in}^{E} u_i \right)$$

$$\tau_I^{\alpha_D} \mathcal{D}^{\alpha_D}_t I_i = -I_i + \mathcal{S}_I\!\left( \sum_j W_{ij}^{IE} E_j - \sum_j W_{ij}^{II} I_j \right)$$

with $\mathcal{D}^{\alpha_D}_t$ a temporal fractional derivative, $\tau$ time
constants, and $\mathcal{S}$ a sigmoid firing rate. (The explicit E/I split is a
ROADMAP "Next" item; the current `FractionalReservoir` is the single-population
reduction.)

---

## 3. Mathematical Framework of the Fractional Reservoir (F-RC)

Embedding fractional dynamics into the *nodes* relocates the generation of
long-range dependence (LRD) from network topology to the integro-differential
operators themselves: a single fractional node already has **power-law** memory,
whereas an ordinary leaky node has only **exponential** memory.

### 3.1 Continuous Formulation
With Caputo derivative $\mathcal{D}^{\alpha_D}_t$ and node decays
$\mathbf{\Lambda}=\mathrm{diag}(\lambda_i)$:

$$\mathcal{D}^{\alpha_D}_t \mathbf{x}(t) = -\mathbf{\Lambda}\mathbf{x}(t) + f\big(\mathbf{W}_{res}\mathbf{x}(t) + \mathbf{W}_{in}\mathbf{u}(t)\big)$$

The $-\mathbf{\Lambda}\mathbf{x}$ leak is essential and must survive
discretisation — v1's Section 6 code dropped it, which (together with the
leading-coefficient error below) is why the naive update diverges.

### 3.2 Discrete-Time Approximation (Grünwald–Letnikov) ⚠ CORRECTION
The GL fractional difference is

$$\Delta^{\alpha_D} \mathbf{x}_k = \frac{1}{h^{\alpha_D}} \sum_{j=0}^{k} w_j\,\mathbf{x}_{k-j},\qquad w_j = (-1)^j \binom{\alpha_D}{j},$$

with the verified recursion $c_0=1,\ c_j=\big(1-\tfrac{1+\alpha_D}{j}\big)c_{j-1}$
(so $c_j\equiv w_j$; in particular $c_1=-\alpha_D$).

Setting $\Delta^{\alpha_D}\mathbf{x}_k = \mathbf{g}_k$ where
$\mathbf{g}_k=-\mathbf{\Lambda}\mathbf{x}_{k-1}+f(\mathbf{W}_{res}\mathbf{x}_{k-1}+\mathbf{W}_{in}\mathbf{u}_k)$,
and isolating $\mathbf{x}_k$ (recall $w_0=1$):

$$\boxed{\ \mathbf{x}_k = \underbrace{\alpha_D}_{=-w_1}\,\mathbf{x}_{k-1} \;-\; \sum_{j=2}^{k} c_j\,\mathbf{x}_{k-j} \;+\; h^{\alpha_D}\,\mathbf{g}_k\ }$$

**This is the correction.** v1 wrote the leading coefficient as `1`
($\mathbf{x}_{k-1}$ with no factor) and omitted the $h^{\alpha_D}$ scaling. With
coefficient `1` the linear memory gain is $1+\alpha_D$ and the trajectory
diverges; the correct coefficient is $\alpha_D=-w_1$, which makes the bare linear
gain exactly $1$, after which the leak $\mathbf{\Lambda}$ pulls it strictly below
$1$ (the discrete ESP analogue). This matches `src/fracres/kernels.py`
(`leading = -c_1 = alpha`) and `reservoirs.FractionalReservoir`.

Numerical check (α_D = 0.7): recursion reproduces
$w=[1,-0.7,-0.105,-0.0455,\dots]$ exactly; leading coefficient $-w_1=0.7=\alpha_D$. ✓

### 3.2b L1–Caputo scheme (the package default)
The L1 scheme is gentler than GL near $t=0$ and is the `fracres` default kernel.
With weights $b_m=(m+1)^{1-\alpha_D}-m^{1-\alpha_D}$ ($b_0=1$, monotone
decreasing), the explicit one-step form is

$$\mathbf{x}_k = (1-b_1)\,\mathbf{x}_{k-1} - \sum_{j=2}^{L}(b_j-b_{j-1})\,\mathbf{x}_{k-j} + \Gamma(2-\alpha_D)\,h^{\alpha_D}\,\mathbf{g}_k,$$

i.e. `leading = 1 - b_1`, strict-past weights `b_j - b_{j-1}`, forcing factor
$\Gamma(2-\alpha_D)$ — exactly `src/fracres/kernels.L1CaputoKernel`.

### 3.3 Short-Memory Principle and its high-performance alternative
Truncating the non-local sum to a window $L$ caps cost at $\mathcal{O}(L)$ per
step (rolling buffer), at the price of discarding tail memory $j>L$. The error
of truncation decays like $L^{-\alpha_D}$, so persistent ($\alpha_D$ small)
regimes need large $L$.

➕ **NEW — FFT/block convolution for *exact* long memory.** Because the update is
a convolution of the state history with a fixed power-law kernel, the full
(untruncated) memory over $T$ steps can be evaluated **off-line** in
$\mathcal{O}(T\log T)$ via FFT, or **on-line** in amortised $\mathcal{O}(\log T)$
per step using the Hairer–Lubich *fast convolution* (block/exponential-sum
decomposition of the GL kernel). For a generative phantom brain producing long
EEG-like records, this removes the truncation/memory trade-off entirely and is
the recommended path once kernels are validated (§7). See also §6.4.

### 3.4 Stability of the fractional reservoir — Matignon's theorem ➕ NEW
For the linearised autonomous system $\mathcal{D}^{\alpha_D}_t\mathbf{x}=\mathbf{A}\mathbf{x}$
with $\mathbf{A}=-\mathbf{\Lambda}+\mathbf{W}_{res}$ and $0<\alpha_D<1$, the
relevant stability result is **not** $\rho<1$ but Matignon (1996):

$$\text{asymptotically stable}\iff |\arg(\lambda_i(\mathbf{A}))| > \frac{\alpha_D\,\pi}{2}\quad\forall i.$$

The stable region is a wedge of half-angle $\alpha_D\pi/2$ around the negative
real axis. **Consequence:** as $\alpha_D\to 1$ the threshold tends to $\pi/2$
(classical $\mathrm{Re}(\lambda)<0$); for $\alpha_D<1$ the threshold is *smaller*,
so the stable region is **larger** — eigenvalues with positive real part can
remain stable provided their argument exceeds $\alpha_D\pi/2$. Verified thresholds:
$\alpha_D=0.5\Rightarrow45°$, $0.7\Rightarrow63°$, $0.9\Rightarrow81°$,
$1.0\Rightarrow90°$.

This is genuinely useful for a phantom brain: fractional nodes let the reservoir
sit near criticality with "hotter" connectomes (larger effective spectral
content) than a classical ESN could, buying richer, longer fading memory without
losing stability. It also tells us *how* to set the edge of chaos in the
fractional case — push eigenvalue arguments toward the wedge boundary
$\alpha_D\pi/2$ rather than pushing $\rho\to1$.

### 3.5 Memory capacity and the edge of chaos ➕ NEW
Two quantitative diagnostics should replace hand-tuning:

- **Linear memory capacity** $\mathrm{MC}=\sum_{k\ge1}\mathrm{corr}^2(\hat y_k(t),u(t-k))$,
  the reconstructability of delayed inputs by a linear readout. Bounded above by
  $N_x$; maximised near the stability boundary.
- **Power-law vs exponential memory.** A leaky node's memory kernel is
  $\sim e^{-\lambda k}$ (finite MC, short horizon); a fractional node's is
  $\sim k^{-(1+\alpha_D)}$ (heavy-tailed, long horizon). This is the mechanistic
  reason F-RC manufactures LRD at the node level. Report MC and the empirical
  memory-kernel slope as standard outputs (§7).

---

## 4. Function-Space Perspective: Besov Regularity

When the drive is a scale-free stochastic process, state trajectories live in
fractional function spaces, and the *right* loss geometry is Besov, not $L^2$.

### 4.1 From Sobolev–Slobodeckij to Besov
The Slobodeckij seminorm of $W^{s,p}(\Omega)$,

$$[V]_{s,p} = \left( \int_\Omega \int_\Omega \frac{|V(x)-V(y)|^p}{|x-y|^{d+sp}}\,dx\,dy \right)^{1/p},$$

costs $\mathcal{O}(n^2)$. The equivalent Besov norm $B^s_{p,q}$ via the
**Littlewood–Paley** decomposition turns this into dyadic frequency-band
filtering at $\mathcal{O}(n\log n)$ (FFT) — see §4.3 and `regularizers.py`.

### 4.2 Strict regularity bounds — the central claim, stated correctly ⚠ CORRECTION
v1 gave: "Integrability $p<\alpha$ (order of the fractional derivative);
smoothness $s<H$." The first clause is wrong on two counts, and the pair is a
strict weakening of the project's actual central claim.

**(i) The integrability bound is about heavy tails, not the derivative order.**
For an α-stable driving law the $p$-th moment is finite **iff** $p<\alpha_S$,
the *stability index* $\in(0,2]$ — a property of the noise, unrelated to
$\alpha_D$. So the correct statement is $p<\alpha_S$.

**(ii) The smoothness bound interacts with $p$.** fBm with Hurst $H$ has sample
paths in $B^s_{p,q}$ for $s<H$ (Ciesielski–Kerkyacharian–Roynette 1993). But to
control the trajectory in the $L^p$ geometry that the heavy tail forces, the
admissible smoothness is also capped by the Besov embedding scale $1/p$.
Combining the two gives the project's **central claim**:

$$\boxed{\ s < \min\{H,\ 1/p\},\qquad p < \alpha_S\ }$$

**Why $L^2$ fails (the motivating gap).** The plain $L^2$ MSE corresponds to
$B^0_{2,2}$ — $s=0$, $p=2$. When the drive is heavy-tailed ($\alpha_S<2$) the
second moment is infinite, so the $L^2$ objective is not even well defined in the
population limit and is dominated by rare large excursions in finite samples;
and when $H\le 1/2$ the $s=0$ choice is blind to the scale-free roughness that
defines the target dynamics. The Besov penalty with $s<\min\{H,1/p\}$, $p<\alpha_S$
is the remedy: it measures the trajectory in a norm that *is* finite and *does*
see the multi-scale structure. This is the theoretical spine of the thesis and
should be cited as such (see `spaces/.../central_claim.md`).

> **Action item for the code:** `regularizers.py` and `training.py` currently
> hard-code `p = alpha - margin`, `s = H - margin` using the *derivative* order.
> They should instead take the heavy-tail index $\alpha_S$ for $p$ and set
> $s = \min\{H,1/p\} - \text{margin}$. (Out of scope for this doc-only revision;
> flagged for the code agent.)

### 4.3 Littlewood–Paley characterisation
With smooth band-pass filters $\phi_j$ localised at $|\omega|\sim 2^j$ and a
low-pass $\psi$, the dyadic blocks $\Delta_j\hat y=\mathcal F^{-1}(\phi_j\mathcal F\hat y)$
give the inhomogeneous Besov norm

$$\|\hat y\|_{B^s_{p,q}} = \|\mathcal F^{-1}(\psi\,\mathcal F\hat y)\|_{L^p} + \left( \sum_{j\ge1} \big(2^{js}\|\Delta_j\hat y\|_{L^p}\big)^q \right)^{1/q}.$$

This is exactly `regularizers.littlewood_paley_penalty` + `make_dyadic_masks`.

**Numerical caveats (➕ NEW):** when $p<1$ (which happens whenever
$\alpha_S<1$) the "$L^p$ norm" is a quasi-norm and `(·)**(1/p)` strongly
amplifies small values — keep the `+1e-12` stabiliser, prefer `float64`
(`jax.config.update("jax_enable_x64", True)`) for the band sums, and consider
working with $\log$-norms to avoid overflow. The current code adds the
stabiliser *inside* the sum (`|Δ|^p + 1e-12`); adding it to the base
(`(|Δ|+ε)^p`) is gentler on the gradient near zero.

---

## 5. Quasi-Self-Organised Criticality (qSOC)

Biological cortex exhibits *quasi*-SOC: feedback continually nudges the system
back to the critical transition when drive pushes it toward hyper-excitability
(supercritical) or silence (subcritical).

### 5.1 Homeostatic threshold control ⚠ CORRECTION (consistency)
v1 defines a *windowed* macroscopic energy

$$E_{avg}(t) = \frac{1}{\tau_{soc}} \int_{t-\tau_{soc}}^{t} \|\mathbf{x}(\tau)\|^2 \, d\tau,$$

but the threshold ODE it then writes — and the code that implements it — uses the
**instantaneous** $\|\mathbf{x}(t)\|^2$:

$$\tau_b \frac{d\mathbf{b}}{dt} = -\mathbf{b} + \gamma\big(E_{crit} - \|\mathbf{x}(t)\|^2\big).$$

These are not the same controller. The instantaneous form is what
`qSOCFractionalReservoir` integrates with an explicit Euler step. Either
(a) drop $E_{avg}$ and present the instantaneous controller as the model, or
(b) actually low-pass the energy (an extra leaky-integrator state
$\dot E = (\|\mathbf x\|^2 - E)/\tau_{soc}$ carried through the scan) so the
written and implemented controllers agree. Option (b) is more faithful to
homeostatic plasticity and only adds one scalar to the carry.

➕ **NEW — stability of the controller.** The explicit-Euler threshold update is
conditionally stable: it requires $dt < 2\tau_b$, and becomes stiff as
$\tau_b\to0$ (ROADMAP open question). A semi-implicit step
$b_{k+1} = \big(b_k + \tfrac{dt}{\tau_b}\gamma(E_{crit}-E_k)\big)/(1+dt/\tau_b)$
is unconditionally stable and a drop-in replacement.

---

## 6. High-Performance Computational Architecture (JAX / Equinox)

> v1's Section 6 listing is superseded: it (a) used the divergent update of §3.2,
> (b) applied `sigmoid(...)` with no `h^α` scaling or leak, and (c) would have
> trained the entire reservoir — including the fixed fractional-kernel weights —
> because every `jnp.ndarray` leaf is differentiated. Use the `fracres` package
> instead; the corrected reference snippets below match it.

### 6.1 The corrected node update (matches `kernels.py` + `reservoirs.py`)
```python
# one reservoir step (GL or L1 kernel supplies leading / weights / forcing_factor)
fractional_memory = einsum("j,jk->k", kernel.weights, x_history[1:])  # strict past
current_state     = x_history[0]
h_alpha           = step_size ** kernel.alpha
g_k               = -decay * current_state + sigmoid(W_res @ current_state + W_in @ u_t)
x_next            = (kernel.leading * current_state
                     - fractional_memory
                     + kernel.forcing_factor * h_alpha * g_k)
```
The key differences from v1: `kernel.leading` (= `α` for GL, `1-b_1` for L1)
instead of an implicit `1`; the `h_alpha` and `forcing_factor` scalings; the
explicit `-decay` leak.

### 6.2 Freezing the reservoir correctly ➕ NEW
For *bona fide* reservoir computing only `readout.W_out` is trainable. The
current `training.py` calls `optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))`,
and `filter_value_and_grad` produces gradients for **all** array leaves —
including `W_res`, `W_in`, and the kernel `weights`. To truly freeze the
reservoir, partition explicitly:

```python
filter_spec = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(lambda m: m.readout.W_out, filter_spec, replace=True)
diff, static = eqx.partition(model, filter_spec)        # only W_out is differentiable

@eqx.filter_jit
def step(diff, static, opt_state, U, Yt, masks, ...):
    def loss_fn(diff):
        model = eqx.combine(diff, static)
        ...
    loss, grads = eqx.filter_value_and_grad(loss_fn)(diff)
    updates, opt_state = optimizer.update(grads, opt_state, diff)
    return eqx.apply_updates(diff, updates), opt_state, loss
```
This also guarantees the fractional kernel weights are never perturbed.

### 6.3 Closed-form ridge readout ➕ NEW (ROADMAP "Now")
```python
def fit_ridge_readout(X_states, Y_target, beta):
    # X_states: (T, N), Y_target: (T, out)
    X = X_states.T                                  # (N, T)
    G = X @ X.T + beta * jnp.eye(X.shape[0])        # (N, N)
    return jnp.linalg.solve(G, X @ Y_target).T      # W_out: (out, N)
```
Far faster and more stable than gradient descent for the unregularised-topology
case; use the Besov-penalised gradient path only when the topological prior is
actually needed.

### 6.4 Throughput checklist ➕ NEW
- **`jax.lax.scan`** for the time loop (already used) — never a Python loop.
- **`jax.checkpoint` (rematerialisation)** on the scan body when backpropagating
  through long sequences, to bound memory at the cost of recompute.
- **`x64`** for GL coefficient generation and the Besov band sums; the recursion
  $c_j=(1-\tfrac{1+\alpha_D}{j})c_{j-1}$ loses precision in float32 for large $L$.
- **`vmap`** over (i) ensemble seeds for uncertainty, (ii) a batch of drives.
- **`donate_argnums`** on the carry buffers to enable in-place XLA buffer reuse.
- **FFT memory (§3.3)** instead of a long truncated window when $L$ would
  otherwise dominate the per-step cost.
- Precompute and **mark kernel weights static** so they are neither traced as
  inputs nor differentiated.

---

## 7. Validation and Benchmarking Plan ➕ NEW

A high-performance model is worthless if the numerics are wrong; validate before
optimising.

1. **Kernel correctness.** Test against the analytic fractional derivative of a
   power law: $\mathcal{D}^{\alpha_D} t^{\beta} = \dfrac{\Gamma(\beta+1)}{\Gamma(\beta+1-\alpha_D)}\,t^{\beta-\alpha_D}$
   (e.g. $\beta=2,\alpha_D=0.7\Rightarrow$ coefficient $1.7142$, verified). Port
   the `hpfracc` validation harness (ROADMAP "Now").
2. **Recover the driving Hurst.** Drive with known $H$; estimate the output's
   $H$ via DFA / wavelet / GPH using the in-repo `benchmarking/lrdbench` toolkit;
   confirm the input→output $H$ relationship and its dependence on $\alpha_D$.
3. **Criticality diagnostics.** Neuronal-avalanche statistics: size and duration
   distributions should be power laws with a branching ratio $\sigma\approx1$ at
   criticality (Beggs & Plenz 2003); track distance-to-criticality as qSOC gains
   vary.
4. **Spectral realism.** The generated EEG/MEG-like signal should show a $1/f^{\beta}$
   power spectrum with $\beta$ in the physiological range.
5. **Regulariser efficacy.** Confirm the Besov penalty actually moves the
   estimated trajectory regularity toward the target $B^s_{p,q}$ (ROADMAP "Next").
6. **Drive fidelity.** `generate_fbm_increments` standardises by empirical std;
   verify the realised fGn autocovariance against the exact Davies–Harte target,
   or switch to exact scaling (ROADMAP open question).

---

## 8. Summary of changes from v1

| # | Location | v1 | v2 |
|---|---|---|---|
| 1 | §3.2 / §6 | leading coeff `1`, no $h^{\alpha}$, no leak → **divergent** | leading coeff $\alpha_D=-w_1$, $h^{\alpha_D}$ scaling, explicit leak (matches source) |
| 2 | §1.3 | $\rho<1$ "sufficient" for ESP | $\rho<1$ necessary; $\sigma_{\max}<1$ sufficient; fractional case → Matignon |
| 3 | §4.2 | $p<\alpha$ (derivative order), $s<H$ | $p<\alpha_S$ (heavy-tail index), $s<\min\{H,1/p\}$; $L^2$-failure rationale |
| 4 | §5.1 | windowed $E_{avg}$ defined, instantaneous energy used | inconsistency flagged; faithful windowed option + semi-implicit step |
| 5 | §6 | trains whole reservoir incl. kernels | explicit partition to freeze all but `W_out` |
| 6 | §0,§3.4,§3.5,§7 | — | two-α notation, Matignon stability, memory capacity, validation plan |

---

## 10. Outstanding source-code actions (for the code agent) ➕ NEW

> **Coordination note (2026-06-27).** Status of the items below:
>
> 1. **Freeze the reservoir properly** (§6.2) — ✅ **IMPLEMENTED** in
>    `training.py`. `train_step` now partitions the model with
>    `readout_filter_spec` so only `readout.W_out` is differentiated/updated;
>    `W_res`, `W_in` and the fractional-kernel `weights` stay exactly frozen.
>    Initialise the optimiser over the trainable partition:
>    `opt_state = optimizer.init(eqx.partition(model, readout_filter_spec(model))[0])`.
> 2. **Correct the Besov regularity indices** (§4.2) — ✅ **IMPLEMENTED**. New
>    `besov_indices(H, alpha_stable=2.0, margin)` sets `p = alpha_stable - margin`
>    (`p < alpha_S`, the heavy-tail index; 2.0 for Gaussian fGn) and
>    `s = min(H, 1/p) - margin`. `compute_loss`/`train_step` now take
>    `alpha_stable` instead of the (misused) derivative order; `regularizers.py`
>    docstring + stabiliser updated.
> 4. **Numerical hygiene** (§4.3) — ⚠️ **PARTIAL**: gentler Besov stabiliser
>    `(|Δ|+ε)^p` applied. `x64` for GL coefficients/band sums still TODO.
> 3. **Reconcile the qSOC controller** (§5.1) — ❌ **NOT DONE** (behavioural
>    change to `reservoirs.py`; left for the code agent).
>
> Items 5–6 (E/I neural-mass split, FFT memory) are enhancements — see ROADMAP.
>
> **Verification status / handoff.** The edited modules (`training.py`,
> `regularizers.py`, `__init__.py`) byte-compile cleanly and independently. The
> full `pytest` suite was **not** run here: the sandbox has no PyPI access to
> install JAX/Equinox, and the repo `.venv` is Windows-only. Additionally, a
> concurrent process was observed rewriting `src/fracres/reservoirs.py` during
> this session (it was left truncated mid-statement at one point), so files were
> changing underneath this edit. **Recommended:** the other agent runs `pytest`
> and a freeze smoke-test (assert only `readout.W_out` changes after one
> `train_step`) once `reservoirs.py` is settled.

---

## 9. References (verify before citation in the thesis)

- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.
- Maass, W., Natschläger, T., Markram, H. (2002). *Real-time computing without stable states.* Neural Computation 14(11).
- Lukoševičius, M., Jaeger, H. (2009). *Reservoir computing approaches to RNN training.* Computer Science Review 3(3).
- Yildiz, I. B., Jaeger, H., Kiebel, S. J. (2012). *Re-visiting the echo state property.* Neural Networks 35.
- Matignon, D. (1996). *Stability results for fractional differential equations with applications to control processing.* IMACS/IEEE-SMC.
- Podlubny, I. (1999). *Fractional Differential Equations.* Academic Press. (GL/L1 schemes, short-memory principle.)
- Ciesielski, Z., Kerkyacharian, G., Roynette, B. (1993). *Quelques espaces fonctionnels associés à des processus gaussiens.* Studia Mathematica 107(2). (fBm ∈ $B^s_{p,q}$, $s<H$.)
- Wilson, H. R., Cowan, J. D. (1972). *Excitatory and inhibitory interactions in localized populations of model neurons.* Biophysical Journal 12(1).
- Beggs, J. M., Plenz, D. (2003). *Neuronal avalanches in neocortical circuits.* Journal of Neuroscience 23(35).
- Davies, R. B., Harte, D. S. (1987). *Tests for Hurst effect.* Biometrika 74(1). (Exact fGn synthesis.)
- Hairer, E., Lubich, C., Schlichte, M. (1985). *Fast numerical solution of nonlinear Volterra convolution equations.* SIAM J. Sci. Stat. Comput. 6(3). (Fast power-law convolution.)

*(Author/year/title given for traceability; confirm exact pagination/DOIs against the `knowledge-base/` PDFs before final citation.)*
