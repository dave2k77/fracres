# Reservoir Computing — Verified Literature Notes

For: *Fractional Reservoir Computing for Critical Brain Dynamics*
Compiled 2026-09-14. Every entry below was verified against a live source:
CrossRef API (all DOI entries), doi.org resolution (302 → publisher), the
authors' report PDFs (HTTP 200, ai.rug.nl/minds mirror), and the NeurIPS
proceedings site + dblp. Semantic Scholar Graph API returned HTTP 429
(rate-limited) from this IP during compilation, so verification relied on
CrossRef and publisher sources. Nothing here is unverified; see "Flags" at
the end for residual caveats.

Recurring thread for the fractional-memory framing: classical RC encodes
memory entirely in **network topology and spectral tuning** (the recurrent
weight matrix), and the echo state property guarantees a **fading memory**
of the input history. A fractional-memory reservoir instead puts memory in
the **dynamics** (power-law kernels / fractional derivatives), so each
reference below is annotated with what that contrast buys or costs.

## (a) Foundations: echo state networks

**jaeger2001echo** — Jaeger, *The "echo state" approach to analysing and
training recurrent neural networks* (GMD Report 148, 2001).
Introduces echo state networks: a randomly connected RNN whose recurrent
weights are left untrained, with learning confined to a linear readout, and
defines the echo state property (network state uniquely determined by the
left-infinite input history). This is the founding statement that memory in
RC lives in the fixed recurrent topology — the very assumption a
fractional-memory reservoir relaxes by giving each node an intrinsic
power-law memory kernel.

**jaeger2002adaptive** — Jaeger, *Adaptive nonlinear system identification
with echo state networks* (NIPS 2002, pp. 593–600).
First peer-reviewed venue demonstration that ESNs identify highly nonlinear
dynamical systems with orders-of-magnitude better accuracy than prior RNN
methods. Establishes the train-the-readout-only pipeline that fractional
variants typically keep, changing only the state-update equation.

## (b) Liquid state machines

**maass2002liquid** — Maass, Natschläger & Markram, *Real-time computing
without stable states* (Neural Computation 14(11):2531–2560, 2002).
Independent, neuroscience-motivated formulation of reservoir computing: a
generic cortical microcircuit ("liquid") perturbed by input streams, read
out by simple linear units, with a fading-memory guarantee via the
separation property. Directly relevant to brain dynamics: it argues
biological circuits compute on *transients*, not fixed points — a viewpoint
compatible with critical, fractionally ordered neural dynamics.

**verstraeten2007unification** — Verstraeten et al., *An experimental
unification of reservoir computing methods* (Neural Networks
20(3):391–403, 2007).
Shows empirically that ESNs, LSMs, and backpropagation-decorrelation are
instances of one framework, and benchmarks reservoirs across memory and
nonlinearity regimes. Useful baseline evidence that the *reservoir/readout
split* — not the specific neuron model — is the essential object, which is
exactly what fractional RC modifies.

## (c) Survey

**lukosevicius2009survey** — Lukoševičius & Jaeger, *Reservoir computing
approaches to recurrent neural network training* (Computer Science Review
3(3):127–149, 2009).
The standard survey unifying ESN, LSM and related models; catalogues
reservoir generation recipes, spectral-radius tuning, and known failure
modes. Provides the canonical statement that good reservoirs need a
carefully tuned memory/nonlinearity trade-off — the tuning burden that
fractional dynamics is claimed to soften.

## (d) Echo state property theory

**buehner2006tighter** — Buehner & Young, *A tighter bound for the echo
state property* (IEEE TNN 17(3):820–824, 2006).
Derives a sufficient condition for the ESP based on the largest singular
value that is tighter and less conservative than Jaeger's original
spectral-radius condition. Anchors the mathematical meaning of "fading
memory": any fractional-memory reservoir must re-establish an ESP analogue
under power-law (non-geometric) decay, since these bounds assume contractive
geometric forgetting.

**yildiz2012echo** — Yildiz, Jaeger & Kiebel, *Re-visiting the echo state
property* (Neural Networks 35:1–9, 2012).
Shows the ESP is not purely a property of the weight matrix: it depends on
the input, and there exist inputs for which no scaling of a given reservoir
has echo states (and vice versa). Sharpens what a fractional design must
prove — input-dependent contractivity of a fractional state map, where
classical Lipschitz bounds no longer apply directly.

## (e) Edge of chaos / criticality

**bertschinger2004edge** — Bertschinger & Natschläger, *Real-time
computation at the edge of chaos in recurrent neural networks* (Neural
Computation 16(7):1413–1436, 2004).
Demonstrates with Lyapunov-exponent analysis that real-time computing
performance peaks at the transition between ordered and chaotic dynamics.
The core criticality result for reservoirs: in classical RC, criticality is
reached by *tuning spectral radius to the edge*; a fractional reservoir
instead has tunable memory order as a second axis to sit at criticality.

**legenstein2007edge** — Legenstein & Maass, *Edge of chaos and prediction
of computational performance for neural circuit models* (Neural Networks
20(3):323–334, 2007).
Extends edge-of-chaos analysis to more biologically realistic circuit
models and proposes predicting performance from the network's Jacobian
spectrum. Bridges RC criticality and brain-criticality literature — the
conceptual hinge for a review on critical brain dynamics.

## (f) Memory capacity

**jaeger2002memory** — Jaeger, *Short term memory in echo state networks*
(GMD Report 152, 2002).
Defines the memory-capacity curve MC(τ) for ESNs and proves a bound: linear
ESNs with N units have total memory capacity ≤ N, with characteristic
exponential decay of recall with delay. The canonical statement of
classical RC's *geometric* forgetting — the quantitative foil for
fractional reservoirs, whose power-law kernels promise algebraically (not
exponentially) decaying memory curves.

**dambre2012capacity** — Dambre, Verstraeten, Schrauwen & Massar,
*Information processing capacity of dynamical systems* (Scientific Reports
2:514, 2012).
Generalizes Jaeger's memory capacity to a full information-processing
capacity profile over all polynomial functionals of past inputs, proving
the total is bounded by the number of linearly independent reservoir
variables. Frames the key design question for fractional RC: can power-law
memory *redistribute* (rather than increase) capacity toward long delays,
given this conservation law?

## (g) Modern developments

**tanaka2019review** — Tanaka et al., *Recent advances in physical
reservoir computing: A review* (Neural Networks 115:100–123, 2019).
Comprehensive review of physical substrates (photonic, spintronic,
mechanical, biological) implementing reservoirs, including time-delay
(single-node) architectures. Delay-based reservoirs are the closest
classical analogue to fractional designs: both replace topological memory
with dynamical memory in the node itself.

**nakajima2021book** — Nakajima & Fischer (eds.), *Reservoir Computing:
Theory, Physical Implementations, and Applications* (Springer, Natural
Computing Series, 2021).
Edited volume covering theory, physical implementations, and applications
including neuromorphic and biological reservoirs. Useful for situating
fractional-memory reservoirs within current theory (ESP variants,
capacity) and for brain-relevant implementations.

**gauthier2021nextgen** — Gauthier, Bollt, Griffith & Barbosa, *Next
generation reservoir computing* (Nature Communications 12:5564, 2021).
Shows an RC-forecasting task can be matched by a nonlinear vector
autoregression ("next-generation RC") with far less data and no random
reservoir. A cautionary modern result: it strips the reservoir down to its
memory-functional content, which is precisely the component fractional
calculus modifies — making NG-RC both a competitor and a natural host for
fractional (long-memory) feature maps.

## Flags / caveats

- Jaeger 2001 and 2002 (GMD reports 148/152) have no DOI; verification is
  via the authors' PDF copies (HTTP 200 at ai.rug.nl/minds) plus
  corroborating bibliographic records (Scholarpedia, BibSonomy, Springer
  reference lists). Year/number confirmed; page counts not asserted.
- jaeger2002adaptive (NIPS 2002) has no DOI (NeurIPS proceedings of that
  era); verified via proceedings.neurips.cc abstract page and dblp
  (pp. 593–600). The MIT Press volume is dated 2003 in some records;
  cited here by conference year 2002 as is conventional.
- Semantic Scholar API was unreachable (HTTP 429) during compilation; all
  DOI metadata above comes from CrossRef responses captured live.
