# Legacy starter fragments

These are the original starter `.py` files, kept for reference. They were
fragments carved out of the knowledge-base pseudocode (Section 6) and did **not**
run standalone (missing imports, duplicated coefficients, an orphaned
`TopologicalReadout`, one empty file).

They have been fully refactored into the importable package under
`src/fracres/`. Mapping:

| Legacy fragment                          | Now lives in                          |
|------------------------------------------|---------------------------------------|
| `abstract_fractional_reservoir_brain.py` | `kernels.py`, `reservoirs.py`, `models.py` |
| `fractional_nm_reservoir_brain.py`       | `reservoirs.py`, `regularizers.py`, `training.py`, `readout.py` |
| `fractional_qsoc_reservoir_brain.py`     | `reservoirs.qSOCFractionalReservoir`, `models.qSOCPhantomBrain` |
| `fBm_noise_driver.py`                     | `drivers.py` + `examples/run_fbm_driven.py` |
| `fractional_nf_reservoir_brain.py` (empty) | `reservoirs.NeuralFieldReservoir` (stub) |

Safe to delete once you're comfortable with the new layout.
