"""fracres: Fractional Reservoir Computing for critical (phantom) brain dynamics.

A small, JAX/Equinox library for building reservoir computers whose nodes carry
*fractional* (non-Markovian, power-law) memory, driven by heavy-tailed stochastic
noise, and regularised toward critical dynamics in a Besov / L^p geometry.

See ``docs/knowledge_base.md`` for the theoretical framework.
"""
from __future__ import annotations

from fracres.drivers import generate_fbm_increments
from fracres.kernels import AbstractFractionalKernel, GLKernel, L1CaputoKernel
from fracres.models import PhantomBrain, qSOCPhantomBrain
from fracres.readout import TopologicalReadout
from fracres.regularizers import littlewood_paley_penalty, make_dyadic_masks
from fracres.reservoirs import FractionalReservoir, qSOCFractionalReservoir
from fracres.stability import (
    MatignonDiagnostics,
    matignon_diagnostics,
    matignon_edge_scale,
    set_edge_of_chaos,
    set_spectral_radius,
    system_matrix,
)
from fracres.training import (
    besov_indices,
    compute_loss,
    fit_readout_ridge,
    fit_ridge_readout,
    readout_filter_spec,
    train_step,
)
from fracres.validation import (
    analytic_power_law_derivative,
    convergence_order,
    mittag_leffler,
)

__version__ = "0.1.0a0"

__all__ = [
    "AbstractFractionalKernel",
    "GLKernel",
    "L1CaputoKernel",
    "FractionalReservoir",
    "qSOCFractionalReservoir",
    "TopologicalReadout",
    "PhantomBrain",
    "qSOCPhantomBrain",
    "generate_fbm_increments",
    "littlewood_paley_penalty",
    "make_dyadic_masks",
    "compute_loss",
    "train_step",
    "besov_indices",
    "readout_filter_spec",
    "fit_ridge_readout",
    "fit_readout_ridge",
    "MatignonDiagnostics",
    "matignon_diagnostics",
    "matignon_edge_scale",
    "set_spectral_radius",
    "set_edge_of_chaos",
    "system_matrix",
    "analytic_power_law_derivative",
    "convergence_order",
    "mittag_leffler",
    "__version__",
]
