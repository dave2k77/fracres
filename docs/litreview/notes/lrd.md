# Long-Range Dependence Mathematics, Heavy Tails, and LRD in Neural Signals

Literature notes for *Fractional Reservoir Computing for Critical Brain Dynamics*.
All entries verified 2026-09-14 via doi.org CSL content negotiation, Crossref API, or Open Library ISBN API; BibTeX in `../bib/lrd.bib`.

## Key relations used throughout this review

- For fractional Gaussian noise (fGn) with Hurst exponent H, the **DFA fluctuation exponent α equals H** (α = H for 0 < H < 1); for fractional Brownian motion (fBm) the DFA exponent is α = H + 1. This is why DFA must be combined with a stationarity check (e.g., α > 1 indicates a nonstationary, fBm-like signal) before interpreting α as a Hurst exponent.
- The **power spectral density (PSD) of fGn decays as S(f) ∝ f^(−β) with β = 2H − 1** over the scaling range. Thus H = (β + 1)/2, and 1/f-like spectra (β ≈ 1) correspond to H ≈ 1, the edge of the fGn family. For fBm, β = 2H + 1.
- **Heavy tails (variance trap):** L2-based estimators — variance–time plots, R/S analysis, aggregated-variance, and DFA to second order — presuppose finite variance. For α-stable processes with tail index α_tail < 2 the variance is infinite, so sample variances grow with sample size and second-order LRD estimators are biased or meaningless; H estimates then confound heavy-tailedness with true long memory (Samorodnitsky & Taqqu 1994; Samorodnitsky 2007). Wavelet log-scale diagrams remain more robust but their confidence intervals still assume finite moments above the wavelet order used.

## (a) fBm / fGn

**mandelbrot1968fractional** — Mandelbrot & Van Ness, *Fractional Brownian Motions, Fractional Noises and Applications*, SIAM Review 10(4):422–437, 1968. Defines fractional Brownian motion B_H(t) as the self-similar Gaussian process with stationary increments parameterized by H ∈ (0,1), and its increment process, fractional Gaussian noise. Establishes that for H > 1/2 increments are positively correlated (persistent) with non-summable autocorrelation — the canonical mathematical model of long-range dependence. *Verified via doi.org CSL: 10.1137/1010093.*

## (b) The Hurst phenomenon

**hurst1951long** — Hurst, *Long-Term Storage Capacity of Reservoirs*, Trans. ASCE 116(1):770–799, 1951. Reports the empirical rescaled-range (R/S) analysis of Nile River minima and ~75 geophysical time series, finding R/S ∝ n^K with K ≈ 0.73 rather than the n^1/2 expected for independent increments. This empirical anomaly — the "Hurst phenomenon" — motivated all subsequent LRD modeling. *Verified via doi.org CSL: 10.1061/TACEAT.0006518.*

**mandelbrot1969noah** — Mandelbrot & Wallis, *Noah, Joseph, and Operational Hydrology*, Water Resources Research 4(5):909–918, 1968. Explains the Hurst phenomenon via fGn (the "Joseph effect": long-run persistence) and heavy-tailed marginals (the "Noah effect": extreme floods), and shows how both bias naive reservoir design. NOTE: Crossref metadata for DOI 10.1029/WR004i005p00909 gives vol. 4(5), Oct 1968, pp. 909–918; the paper is very widely (mis)cited as vol. 5(5), 1969 — we follow the verified publisher metadata. *Verified via doi.org CSL.*

## (c) Estimators and long-memory statistics

**taqqu1995estimators** — Taqqu, Teverovsky & Willinger, *Estimators for Long-Range Dependence: An Empirical Study*, Fractals 3(4):785–798, 1995. Systematic Monte Carlo comparison of nine H estimators (R/S, aggregated variance, periodogram, Whittle, wavelet, etc.) on synthetic fGn/fBm and FARIMA series of realistic lengths. Shows most estimators are strongly biased at short series and that agreement among several methods is required before claiming LRD — the standard cautionary reference for empirical H estimation. *Verified via doi.org CSL: 10.1142/S0218348X95000692.*

**beran1994statistics** — Beran, *Statistics for Long-Memory Processes*, Chapman & Hall/CRC, 1994 (Monographs on Statistics and Applied Probability 61, 315 pp.). The standard monograph on statistical inference for long-memory processes: spectral representations, fractional ARIMA, Whittle/GPH semiparametric estimation, and asymptotic theory. Defines LRD rigorously via non-summable autocovariance / spectral pole at zero frequency. *Verified via Open Library ISBN 0-412-04901-5 (1994, Chapman & Hall, 315 pp.); DOI 10.1201/9780203738481 resolves to the CRC digital reissue.*

**granger1980introduction** — Granger & Joyeux, *An Introduction to Long-Memory Time Series Models and Fractional Differencing*, J. Time Series Analysis 1(1):15–29, 1980. Introduces fractional differencing (1−B)^d and the ARFIMA framework, bridging fGn-type spectral behavior β = 2d (i.e., H = d + 1/2) with practical econometric time-series models. *Verified via doi.org CSL: 10.1111/j.1467-9892.1980.tb00297.x.*

## (d) Detrended fluctuation analysis

**peng1994mosaic** — Peng, Buldyrev, Havlin, Simons, Stanley & Goldberger, *Mosaic Organization of DNA Nucleotides*, Phys. Rev. E 49(2):1685–1689, 1994. Introduces detrended fluctuation analysis (DFA), which removes polynomial trends at each scale before computing fluctuation F(n) ∝ n^α, making scaling analysis robust to nonstationarity. For stationary fGn the DFA exponent satisfies **α = H** (and α = H+1 for fBm); the paper's α ≈ 0.6–0.7 for non-coding DNA demonstrated long-range correlations in patchy (mosaic) sequences. *Verified via doi.org CSL: 10.1103/PhysRevE.49.1685.*

## (e) Wavelet-based estimation

**abry1998wavelet** — Abry & Veitch, *Wavelet Analysis of Long-Range-Dependent Traffic*, IEEE Trans. Inf. Theory 44(1):2–15, 1998. Establishes the wavelet log-scale diagram: for an LRD process, E|d(j,k)|² ∝ 2^(j(2H)) across octaves j, so H is read from a linear fit of log-energy vs. scale. Shows wavelets naturally whiten fGn-like processes (decorrelation property) and provides unbiased, robust estimation — applied here to Internet traffic. *Verified via doi.org CSL: 10.1109/18.650984.*

**veitch1999wavelet** — Veitch & Abry, *A Wavelet-Based Joint Estimator of the Parameters of Long-Range Dependence*, IEEE Trans. Inf. Theory 45(3):878–897, 1999. Extends the wavelet framework to jointly estimate H and the power-law amplitude (c_f) with full covariance-weighted regression and confidence intervals. Supplies the practical estimator later adopted for neural scaling exponents. *Verified via doi.org CSL: 10.1109/18.761330.*

**abry2003self** — Abry, Flandrin, Taqqu & Veitch, *Self-Similarity and Long-Range Dependence through the Wavelet Lens*, in Doukhan, Oppenheim & Taqqu (eds.), *Theory and Applications of Long-Range Dependence*, Birkhäuser Boston, 2003, pp. 527–556. Review chapter unifying self-similarity, LRD, and wavelet estimation, including the Besov-space perspective on scaling regularity. **PARTIALLY VERIFIED / FLAGGED:** the container volume is verified (Open Library ISBN 0-8176-4168-8: Birkhäuser Boston, 736 pp.), but no chapter-level DOI is registered in Crossref and the chapter could not be independently confirmed via OpenAlex/Semantic Scholar (rate-limited) at verification time — treat page numbers 527–556 as unverified. 

## (f) Heavy tails and stable processes

**samorodnitsky1994stable** — Samorodnitsky & Taqqu, *Stable Non-Gaussian Random Processes: Stochastic Models with Infinite Variance*, Chapman & Hall, 1994 (632 pp.). The definitive treatment of α-stable (0 < α_tail < 2) processes, where variance is infinite and covariance is undefined. Establishes that for heavy-tailed LRD models the dependence structure must be characterized by codifference / alternative measures, since **all L2 (variance- and covariance-based) estimators break down — the variance trap**: sample variance diverges with n, corrupting R/S, variance–time, and DFA-2 estimates. *Verified via Open Library ISBN 0-412-05171-0 (1994, Chapman & Hall, 632 pp.); DOI 10.1201/9780203738818 resolves to the CRC digital reissue.*

**samorodnitsky2007long** — Samorodnitsky, *Long Range Dependence*, Foundations and Trends in Stochastic Systems 1(3):163–257, 2007 (Crossref online date 2006). Authoritative modern survey distinguishing LRD definitions (second-order, tail-based, spectral) and showing how the "right" definition depends on the marginal heaviness of the process. Explicitly catalogs where second-order characterizations fail under infinite variance and surveys LRD for stable and heavy-tailed renewal models. *Verified via doi.org CSL: 10.1561/0900000004.*

## (g) LRD in brain signals

**linkenkaer2001long** — Linkenkaer-Hansen, Nikouline, Palva & Ilmoniemi, *Long-Range Temporal Correlations and Scaling Behavior in Human Brain Oscillations*, J. Neurosci. 21(4):1370–1377, 2001. First demonstration that amplitude envelopes of spontaneous human MEG/EEG alpha, beta, and theta oscillations exhibit power-law temporal correlations over hundreds of seconds, with DFA exponents α ≈ 0.6–0.8 — i.e., genuine LRD in ongoing cortical dynamics, by the α = H relation for (stationary) envelope signals. *Verified via doi.org CSL: 10.1523/JNEUROSCI.21-04-01370.2001.*

**hardstone2012detrended** — Hardstone, Poil, Schiavone, Jansen, Nikulin, Mansvelder & Linkenkaer-Hansen, *Detrended Fluctuation Analysis: A Scale-Free View on Neuronal Oscillations*, Front. Physiol. 3:450, 2012. Methodological primer and review of DFA applied to neuronal oscillations, covering pitfalls (nonstationarity, crossover scales, α = H vs. α = H+1 ambiguity) and surveying DFA changes across sleep/wake states, development, and pathologies (e.g., Alzheimer's, depression). *Verified via doi.org CSL: 10.3389/fphys.2012.00450.*

**he2014scale** — He, *Scale-Free Brain Activity: Past, Present, and Future*, Trends Cogn. Sci. 18(9):480–487, 2014. Review of 1/f-type power spectra (PSD ∝ f^(−β)) in EEG/MEG/ECoG and their relation to excitation/inhibition balance; links spectral slope changes to task states and aging, and discusses scale-free activity as a possible signature of near-critical dynamics. *Verified via doi.org CSL: 10.1016/j.tics.2014.04.003.*

**voytek2015dynamic** — Voytek & Knight, *Dynamic Network Communication as a Unifying Neural Basis for Cognition, Development, Aging, and Disease*, Biological Psychiatry 77(12):1089–1097, 2015. Argues that aperiodic 1/f spectral slope and oscillatory dynamics jointly index network communication, with flattening of the slope across development/aging and disease — connecting the β = 2H−1 spectral perspective to clinical electrophysiology. NOTE: the task brief suggested a *Neuron* citation; the verified article is in *Biological Psychiatry* (DOI 10.1016/j.biopsych.2015.04.016). *Verified via doi.org CSL.*

## (h) Exact fGn simulation

**davies1987tests** — Davies & Harte, *Tests for Hurst Effect*, Biometrika 74(1):95–101, 1987. Best known for the embedded exact method for simulating fGn of arbitrary length via circulant embedding of the covariance matrix (the Davies–Harte algorithm), enabling O(n log n) exact synthesis — the standard generator used in estimator validation studies. Also develops score-based tests for the Hurst effect. *Verified via doi.org CSL: 10.1093/biomet/74.1.95.*

---

## Verification summary

- **16 / 17 entries fully verified** against publisher metadata via doi.org CSL JSON (12), Crossref API (2: correct DOIs found for `mandelbrot1969noah`, `abry1998wavelet`), or Open Library ISBN API (2: 1994 books).
- **1 entry partially verified (flagged):** `abry2003self` — container volume verified; chapter-level DOI/pages not independently confirmed (no Crossref chapter DOI; OpenAlex/dblp/Semantic Scholar lookups negative or rate-limited).
- **Corrections made during verification:** (i) Abry & Veitch 1998 DOI is 10.1109/18.650984, not ...650982; (ii) Mandelbrot & Wallis metadata gives WRR 4(5), 1968, DOI 10.1029/WR004i005p00909; (iii) Voytek & Knight 2015 is in *Biological Psychiatry* 77(12), DOI 10.1016/j.biopsych.2015.04.016 (the initially hypothesized DOI resolved to an unrelated Neuron article and was discarded); (iv) Hurst 1951 pages are 770–799 per Crossref.
