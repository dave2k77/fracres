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
from fracres.training import (
    besov_indices,
    compute_loss,
    readout_filter_spec,
    train_step,
)
from fracres.validation import analytic_power_law_derivative, convergence_order

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
    "analytic_power_law_derivative",
    "convergence_order",
    "__version__",
]
