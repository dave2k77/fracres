# fracres

[![CI](https://github.com/dave2k77/fracres/actions/workflows/ci.yml/badge.svg)](https://github.com/dave2k77/fracres/actions/workflows/ci.yml)

**Fractional Reservoir Computing for Critical (Phantom) Brain Dynamics**

`fracres` is a small [JAX](https://github.com/google/jax) /
[Equinox](https://github.com/patrick-kidger/equinox) library for building
reservoir computers whose nodes carry *fractional* (non-Markovian, power-law)
memory. Driven by heavy-tailed stochastic noise and regularised toward critical
dynamics in a Besov / $L^p$ geometry, these "phantom brains" generate synthetic
macroscopic neural signals (EEG/MEG-like) with realistic long-range dependence.

The theory is documented in [`docs/knowledge_base.md`](docs/knowledge_base.md).

> Status: **pre-alpha / research code.** APIs will change.

---

## Why fractional?

A classical reservoir produces long-range dependence (LRD) through network
topology. `fracres` instead places LRD in the *nodes themselves* via fractional
integro-differential operators (Grünwald–Letnikov and L1-Caputo). Combined with:

- **fractional Gaussian noise** drive (tunable Hurst exponent $H$),
- a **Besov-space regulariser** enforcing the regularity bounds $p<\alpha$,
  $s<H$ via a Littlewood–Paley (FFT) decomposition, and
- a **quasi-self-organised-criticality (qSOC)** homeostatic threshold,

the model is pinned near the edge of chaos where critical brain dynamics live.

## Install

```bash
# from the repo root, into a fresh environment
pip install -e ".[dev]"
```

## Quickstart

```python
import jax
from fracres import L1CaputoKernel, PhantomBrain, generate_fbm_increments

key = jax.random.PRNGKey(0)
k_model, k_noise = jax.random.split(key)

kernel = L1CaputoKernel(alpha=0.75, history_length=200, dt=0.005)
model  = PhantomBrain(in_features=1, res_size=500, out_features=8,
                      fractional_operator=kernel, key=k_model)

drive  = generate_fbm_increments(2000, H=0.65, key=k_noise)[:, None]
X, Y_hat = model.simulate(drive)   # reservoir states, observed signal
```

Runnable scripts live in [`examples/`](examples):

```bash
python examples/run_fbm_driven.py
python examples/run_qsoc_simulation.py
```

## Repository layout

```
brain_models/
├── src/fracres/            # the importable package
│   ├── kernels.py          # fractional memory operators (GL, L1-Caputo)
│   ├── reservoirs.py       # FractionalReservoir, qSOCFractionalReservoir
│   ├── readout.py          # TopologicalReadout (the only trained part)
│   ├── models.py           # PhantomBrain, qSOCPhantomBrain
│   ├── drivers.py          # fBm / fGn stochastic drive
│   ├── regularizers.py     # Littlewood–Paley Besov-norm penalty
│   └── training.py         # Besov-regularised readout optimisation
├── examples/               # runnable demonstration scripts
├── tests/                  # pytest suite
├── docs/knowledge_base.md  # theoretical framework (the active knowledge base)
├── pyproject.toml
└── ROADMAP.md
```

## Map: theory → code

| Knowledge-base section            | Module                       |
|-----------------------------------|------------------------------|
| §3 Fractional difference (GL/L1)  | `kernels.py`                 |
| §2–3 Neural-mass reservoir        | `reservoirs.FractionalReservoir` |
| §5 quasi-SOC homeostasis          | `reservoirs.qSOCFractionalReservoir` |
| §4 Besov / Littlewood–Paley       | `regularizers.py`            |
| §1.4 / §6 readout + training      | `readout.py`, `training.py`  |
| fBm driving noise                 | `drivers.py`                 |

## Tests

```bash
pytest
```

## Citing

See [`CITATION.cff`](CITATION.cff). This project sits downstream of the
[`hpfracc`](https://github.com/dave2k77/hpfracc) fractional-calculus library and
the $L^p$/Besov long-range-dependence framework developed alongside it.

## License

MIT — see [`LICENSE`](LICENSE).
