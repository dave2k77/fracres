# Neural Mass and Neural Field Models — Literature Notes

Scope: classical neural mass/field models, pattern formation with lateral-inhibition
("Mexican-hat") connectivity, and fractional-order generalisations of neuronal and
population dynamics. All 16 entries verified against Crossref DOI metadata (2026-09-14).

## Classical neural mass and neural field models

### wilson1972excitatory — Wilson & Cowan (1972), *Biophysical Journal*
The founding neural mass model: spatially localized coarse-grained populations of
excitatory (E) and inhibitory (I) model neurons are described by two coupled nonlinear
integro-differential equations for the fraction of active cells, with sigmoidal response
functions and refractory dynamics. The E–I loop produces multistability, limit cycles and
hysteresis depending on coupling strengths. A fractional generalisation is motivated
because the coarse-grained averaging over heterogeneous neuron timescales naturally yields
power-law (rather than exponential) population memory kernels.

### wilson1973mathematical — Wilson & Cowan (1973), *Kybernetik*
Extends the 1972 mass equations to spatially distributed cortical and thalamic tissue,
deriving conditions for spatially inhomogeneous steady states and propagating activity
waves — an early neural field treatment. It formalises how localised E–I interactions plus
spatial spread generate standing and travelling patterns. Fractional-order temporal
operators would model the non-exponential relaxation observed when fast and slow synaptic
timescales coexist across tissue.

### amari1977dynamics — Amari (1977), *Biological Cybernetics*
Rigorous analysis of a one-dimensional continuum neural field with lateral-inhibition
(Mexican-hat) connectivity and a Heaviside firing nonlinearity. Amari proves existence and
stability of localised "bump" solutions and derives conditions for pattern formation —
the canonical reference for bump dynamics used throughout cortical field theory.
Fractional extensions target the heavy-tailed spatial kernels and anomalous diffusion of
activity that arise when axonal conduction delays and dendritic integration span many scales.

### jansen1995electroencephalogram — Jansen & Rit (1995), *Biological Cybernetics*
The standard neural mass model for EEG/VEP generation: a cortical column of pyramidal
cells coupled to excitatory and inhibitory interneuron populations, each population modelled
by a second-order linear synaptic filter (pulse-to-wave conversion) and a sigmoidal
wave-to-pulse nonlinearity; two coupled columns reproduce visual evoked potentials.
The linear alpha/beta band-pass filters are exact analogues of lumped RC dynamics, so
replacing them with fractional-order filters (constant-phase elements) is the natural way
to capture the 1/f power-law spectral background of real EEG.

### lopesdasilva1974model — Lopes da Silva et al. (1974), *Kybernetik*
The first lumped thalamic alpha-rhythm model: a linear feedback loop of excitatory and
inhibitory neuron populations with second-order postsynaptic potential kernels, whose
resonance near 10 Hz explains the thalamic alpha rhythm. It established that rhythmic EEG
emerges from population feedback rather than pacemaker cells. Fractional-order PSP kernels
are motivated by the experimentally observed power-law decay of synaptic responses and the
broadband nature of thalamocortical filtering.

## Reviews

### deco2008dynamic — Deco, Jirsa, Robinson, Breakspear & Friston (2008), *PLoS Comput. Biol.*
Comprehensive review of the model hierarchy from spiking neurons through neural masses to
cortical fields, emphasising how the same connectivity supports multiple dynamical regimes
and how mean-field reductions underlie EEG/MEG forward models and dynamic causal modelling.
Key message for this review: every rung of the reduction ladder assumes exponential memory
kernels, which fractional calculus generalises to power-law kernels with a single order parameter.

### coombes2005waves — Coombes (2005), *Biological Cybernetics*
Tutorial review of modern neural field theory: integral equation formulation with Mexican-hat
kernels, Evans-function stability of bumps, and interface methods for travelling fronts and
waves in one and two dimensions. Provides the analytical toolkit (and the notation) used when
fractional spatial/temporal derivatives are later inserted into the field equations.

### coombes2010large — Coombes (2010), *NeuroImage*
Companion review bridging "simple" analytically tractable field models and "complex"
large-scale simulations relevant to neuroimaging, including extensions to axonal delay
distributions and stochastic forcing. Fractional-order models can be read as a compact
closure for exactly these distributed-delay and long-memory effects.

## Mexican-hat connectivity and pattern formation

### murray2002mathematical — Murray (2002), *Mathematical Biology: I. An Introduction*, 3rd ed.
Standard text covering activator–inhibitor reaction–diffusion systems, Turing instability
analysis and dispersion relations — the mathematical machinery behind pattern formation with
short-range excitation / long-range inhibition. Fractional reaction–diffusion (space- or
time-fractional) variants are a direct generalisation covered in the same framework, linking
anomalous diffusion to altered Turing patterns.

### ermentrout1998neural — Ermentrout (1998), *Reports on Progress in Physics*
Review of neural networks as spatio-temporal pattern-forming systems, unifying Wilson–Cowan
type tissue models with wave, front and bump phenomena and their bifurcation structure.
Useful for situating neural fields among general pattern-forming media, where fractional
operators are known to change front speeds and selected wavenumbers.

## Fractional-order neuronal and population models

### lundstrom2008fractional — Lundstrom, Higgs, Spain & Fairhall (2008), *Nature Neuroscience*
Landmark experimental result: the spike-frequency adaptation of neocortical pyramidal neurons
to slowly varying inputs follows fractional-order differentiation — the neuron's firing rate
computes a fractional derivative of its input current, with adaptation timescales spanning
seconds to hundreds of seconds (a power-law, scale-free memory). Direct empirical
justification for replacing integer-order adaptation currents in mass models with
fractional-order operators.

### anastasio1994fractional — Anastasio (1994), *Biological Cybernetics*
Shows that the premotor brainstem vestibulo-oculomotor circuitry behaves as a fractional-order
integrator: the oculomotor neural integrator's "leakiness" and the vestibular system's
power-law dynamics are captured by fractional (rather than integer) differentiation/integration.
Early physiological evidence that fractional operators describe real neural computation,
predating and complementing the cortical result of Lundstrom et al.

### mondal2024emergent — Mondal et al. (2024), *Chaos, Solitons & Fractals*
Analyses a fractional-order Wilson–Cowan E–I network, showing that lowering the derivative
order enriches the bifurcation structure (equilibria, oscillations, chaotic and emergent
regimes) relative to the integer model. Demonstrates concretely that the memory order acts as
an extra bifurcation parameter controlling population dynamics — central to the fractional
reservoir-computing motivation.

### gonzalezramirez2022fractional — González-Ramírez (2022), *Front. Comput. Neurosci.*
Formulates a fractional-order neural field model (Caputo-type temporal derivative with
lateral-inhibition kernel) and derives approximate travelling-wave solutions, showing how the
fractional order modulates wave speed and profile. Direct fractional extension of the
Amari/Coombes field framework, relevant to slowly propagating, long-memory cortical activity.

### wang2014stability — Wang, Yu & Wen (2014), *Neural Networks*
Provides stability analysis (Mittag-Leffler stability conditions) for fractional-order
Hopfield neural networks with time delays. Representative of the well-developed theory of
fractional recurrent networks; useful background when fractional memory units are embedded
in reservoir architectures and their convergence must be guaranteed.

### brandibur2022stability — Brandibur & Kaslik (2022), *Fractal and Fractional*
Stability and bifurcation analysis of a fractional-order coupled FitzHugh–Nagumo-type
excitable neuronal model, deriving exact stability domains in the fractional-order/parameter
plane and showing fractional-order-induced stabilisation of oscillations. Links single-cell
excitability to the fractional mass/field level: fractional spiking units are the microscopic
substrate of fractional population kernels.

## Coverage notes and gaps
- All 16 entries verified via Crossref DOI resolution; Semantic Scholar API was rate-limited
  (HTTP 429) during the search session, so Crossref bibliographic search was used for discovery
  of the fractional-order items instead.
- No dedicated fractional-order Jansen–Rit / EEG neural mass paper was found in indexed
  literature during this search; the closest items are gonzalezramirez2022fractional
  (fractional neural field) and the general fractional RNN literature (wang2014stability).
  This is a genuine literature gap the review can position itself against.
- Murray (2002) author name and edition are standard bibliographic knowledge; the Crossref
  record for DOI 10.1007/b98868 confirms title, publisher, series and year but does not list
  the author field.
