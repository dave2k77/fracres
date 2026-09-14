# Fractional Calculus & Fractional-Order Dynamical Systems — Verified Literature Notes

Companion notes for `../bib/fractional.bib`. Every entry was verified against a live
API response (Crossref/doi.org, OpenAlex, OpenLibrary, arxiv.org abstract page) on
2026-09-14. Verification channel is stated per reference. Relevance is framed for
**fractional reservoir computing (FRC)** applied to critical brain dynamics.

---

## (a) Foundations

### oldham1974fractional — Oldham & Spanier, *The Fractional Calculus* (1974)
The classic first textbook treatment of differentiation and integration to arbitrary
order: Riemann–Liouville definitions, the Grünwald–Letnikov limit, and tables of
fractional derivatives of elementary functions. It supplies the discrete
memory-kernel viewpoint (fractional derivative as a weighted sum over history) that
fractional echo state networks exploit directly in their recurrent updates.
Verified via OpenLibrary API (work OL3941460W).

### podlubny1999fractional — Podlubny, *Fractional Differential Equations* (1999)
The standard modern monograph: Caputo vs. Riemann–Liouville derivatives, existence
theory, Laplace-transform solution methods, and the central role of Mittag-Leffler
functions in fractional dynamics. Its treatment of geometric/physical
interpretation and of Grünwald–Letnikov numerics is the canonical reference when
discretising a fractional-order reservoir. Verified via OpenLibrary API
(work OL7941236W, Mathematics in Science and Engineering vol. 198, Academic Press).

### kilbas2006theory — Kilbas, Srivastava & Trujillo (2006)
Encyclopaedic rigour: fractional integrals/derivatives on function spaces,
Mittag-Leffler-type special functions, and ODE/PDE theory. The go-to citation for
well-posedness claims about fractional-order neural dynamics and for the analytic
properties of the power-law memory kernels an FRC reservoir implements. Verified
via Crossref (North-Holland Mathematics Studies vol. 204, ISBN 9780444518323).

## (b) Mittag-Leffler functions

### gorenflo2014mittag — Gorenflo, Kilbas, Mainardi & Rogosin (2014)
The definitive monograph on Mittag-Leffler functions and their generalisations,
including asymptotics and completely monotone relaxation. It explains why the
impulse response of a fractional-order system decays as a Mittag-Leffler function
— fast initial decay crossing over to a power-law tail — which is precisely the
non-exponential fading-memory profile that fractional reservoirs aim to capture.
Verified via Crossref DOI 10.1007/978-3-662-43930-2.

### haubold2011mittag — Haubold, Mathai & Saxena (2011)
A widely cited open-access review of Mittag-Leffler functions and their
applications across physics. Useful as an accessible citation for ML relaxation,
its Laplace pairs, and its connection to anomalous diffusion — relevant when
linking fractional reservoirs to heavy-tailed, critical dynamics in neural data.
Verified via Crossref DOI 10.1155/2011/298628.

## (c) Matignon stability

### matignon1996stability — Matignon, CESA'96 IMACS Multiconference
The foundational stability result for commensurate fractional-order linear
systems: BIBO/asymptotic stability holds iff all eigenvalues λ of the system
matrix satisfy |arg(λ)| > απ/2, where α is the fractional order. For FRC this is
the exact analogue of the echo-state spectral-radius condition — it defines the
stability wedge in the complex plane that a fractional-order reservoir's Jacobian
must respect, and shows the stable region *widens* as α decreases. Verified via
OpenAlex API (work W56786142; ~1,440 citations).

### matignon1998stability — Matignon, ESAIM: Proceedings 5 (1998)
Extends the 1996 results to generalized (non-commensurate, multi-order) fractional
differential systems, with diffusive-representation methods for analysis and
simulation. Relevant to reservoirs whose nodes may run at different fractional
orders (heterogeneous α), and to bridging state-space stability analysis with
practical approximations. Verified via Crossref DOI 10.1051/proc:1998004.

## (d) Numerical schemes

### linxu2007l1 — Lin & Xu, *J. Comput. Phys.* 225 (2007)
Introduces and rigorously analyses the L1 finite-difference discretisation of the
Caputo derivative combined with spectral methods in space, proving convergence
order O(τ^{2−α}) in time. The L1 Caputo scheme is the workhorse for simulating
fractional-order reservoir dynamics node-by-node; this paper is the standard
error-analysis citation for it. Verified via Crossref DOI 10.1016/j.jcp.2007.02.001.

### diethelm2002predictor — Diethelm, Ford & Freed, *Nonlinear Dynamics* 29 (2002)
The Adams–Bashforth–Moulton predictor-corrector for nonlinear fractional ODEs
(the "fractional Adams method"), with error analysis. This is the practical
integrator of choice when an FRC reservoir has nonlinear node dynamics driven by
a Caputo derivative, where the L1 scheme's linear structure is insufficient.
Verified via Crossref DOI 10.1023/A:1016592219341.

### scherer2011grunwald — Scherer, Kalla, Tang & Huang, *Comput. Math. Appl.* 62 (2011)
A careful survey of the Grünwald–Letnikov discretisation: shifted weights, memory
truncation (short-memory principle), and consistency/stability. The GL weights
(−1)^k Γ(α+1)/(Γ(k+1)Γ(α−k+1)) are exactly the trainable-or-fixed memory kernel
used in discrete-time fractional echo state networks, so this paper anchors the
numerical-fidelity claims of that architecture. Verified via Crossref DOI
10.1016/j.camwa.2011.03.054.

## (e) Control / engineering

### monje2010fractional — Monje, Chen, Vinagre, Xue & Feliu (2010)
The standard engineering textbook on fractional-order systems and control:
frequency-domain characterisation, fractional PID tuning, realisation via rational
approximations (Oustaloup, Carlson), and hardware/software implementation.
Supplies the control-theoretic toolkit (CRONE/Oustaloup approximations) used to
implement fractional kernels in real reservoir hardware. Verified via Crossref
DOI 10.1007/978-1-84996-335-0.

### podlubny1999controllers — Podlubny, *IEEE Trans. Automat. Control* 44 (1999)
Introduces the PI^λD^μ controller and shows fractional-order feedback can
outperform integer-order PID. The seminal citation establishing that fractional
orders give strictly more expressive dynamical compensation — the same argument
that motivates fractional over integer-order reservoir memory. Verified via
Crossref DOI 10.1109/9.739144.

## (f) Fractional-order reservoir computing

### yao2020fractional — Yao & Wang, *Neural Processing Letters* 52 (2020)
The first explicit fractional-order echo state network: reservoir states updated
through a fractional (Grünwald–Letnikov-type) difference so the network carries
power-law memory instead of exponential forgetting. Demonstrates improved
time-series prediction (including chaotic benchmarks) over the integer-order ESN.
Direct proof-of-concept that fractional memory kernels help reservoirs — the
foundation for FRC in critical brain dynamics. Verified via Crossref DOI
10.1007/s11063-020-10267-y.

### mastin2026fractional — Mastin et al., *npj Unconventional Computing* (2026)
A review of fractional-order systems for neuromorphic computing covering both
software models and hardware candidates, including a dedicated treatment of
fractional-order reservoir computing and the memory-capacity/fading-memory
trade-off controlled by the fractional order. Positions FRC as "a general and
effective extension of the RC paradigm" — useful for situating our brain-dynamics
application within neuromorphic state of the art. Verified via Crossref DOI
10.1038/s44335-026-00070-8.

### teuscher2026hardware — Teuscher, arXiv:2609.10882 (2026)
A critical review of fractional-order hardware for neuromorphic computing:
derives that the retained history for truncation error ε scales as ε^{−1/α},
plus a fixed-point word-length limit on usable history, and surveys constant-phase
devices across orders. Tells us exactly what physical FRC implementations can and
cannot deliver — essential when mapping fractional reservoirs onto real
neuromorphic substrates for brain-signal modelling. Verified via arxiv.org
abstract page (submitted 9 Sep 2026; arXiv DOI 10.48550/arXiv.2609.10882).

### goswami2026longmemory — Goswami, Paul, Ghosh & Chakraborty, arXiv:2607.11272 (2026)
Proposes long-memory reservoir computing with a Fractional ESN (fESN) whose
reservoir embeds fractional-differencing dynamics; proves standard ESNs induce
short-memory processes while fESN/wESN reservoirs generate polynomially decaying
dependence consistent with statistical long memory. The theoretical guarantee —
polynomially decaying reservoir autocorrelation — is exactly the property needed
to model long-range dependence in critical neural time series. Verified via
arxiv.org abstract page (submitted 13 Jul 2026; arXiv DOI 10.48550/arXiv.2607.11272).

---

## Verification log & flags

| bibkey | Verified via | Notes |
|---|---|---|
| oldham1974fractional | OpenLibrary API (OL3941460W) | Book; no DOI exists. Page count not re-verified. |
| podlubny1999fractional | OpenLibrary API (OL7941236W) | Book; no DOI. ISBN 0125588402 from standard cataloguing. |
| kilbas2006theory | Crossref (book record via 10.1016/S0304-0208(06)80001-0) | That DOI is the book's front matter; ISBN 9780444518323 confirms the volume. |
| gorenflo2014mittag | Crossref DOI 10.1007/978-3-662-43930-2 | 2nd ed. (2014); 1st ed. 2004 exists. |
| haubold2011mittag | Crossref DOI 10.1155/2011/298628 | — |
| matignon1996stability | OpenAlex W56786142 | Conference paper; no DOI exists. Page range 963–968 from standard citations — **page numbers not independently re-verified by API**. |
| matignon1998stability | Crossref DOI 10.1051/proc:1998004 | — |
| linxu2007l1 | Crossref DOI 10.1016/j.jcp.2007.02.001 | NB: an earlier guess (…03.001) resolved to a different JCP paper — corrected via bibliographic search. |
| diethelm2002predictor | Crossref DOI 10.1023/A:1016592219341 | — |
| scherer2011grunwald | Crossref DOI 10.1016/j.camwa.2011.03.054 | — |
| monje2010fractional | Crossref DOI 10.1007/978-1-84996-335-0 | — |
| podlubny1999controllers | Crossref DOI 10.1109/9.739144 | — |
| yao2020fractional | Crossref DOI 10.1007/s11063-020-10267-y | — |
| mastin2026fractional | Crossref DOI 10.1038/s44335-026-00070-8 | Volume/issue/pages not yet assigned at verification time. |
| teuscher2026hardware | arxiv.org/abs/2609.10882 | arXiv API was rate-limited from this host; verified via the live abstract page + arXiv DataCite DOI. |
| goswami2026longmemory | arxiv.org/abs/2607.11272 | Same as above. |

**Could not verify / caveats**
- Matignon 1996 page numbers (963–968) come from secondary citations, not from an
  API response; everything else about the entry is OpenAlex-verified.
- arXiv full-text search API (export.arxiv.org) was unreachable from this host
  (persistent 429 "Rate exceeded"), so the arXiv landscape scan for fractional
  reservoir computing was done via web search + abstract-page verification rather
  than exhaustive API enumeration. The fractional-RC literature appears genuinely
  sparse (4 items found); a deeper arXiv API sweep is recommended when the API is
  reachable.
- Semantic Scholar API was also rate-limited (429) throughout; Crossref, OpenAlex
  and OpenLibrary were used instead.
