"""Experiment configuration: typed dataclasses with YAML (de)serialisation.

An *experiment spec* captures a reproducible run as a small tree of dataclasses
-- the memory kernel, the model/reservoir variant and its hyper-parameters, the
stochastic drive, and the training method -- that round-trips to a single
human-readable YAML file. Factory helpers (:func:`build_kernel`,
:func:`build_model`, :func:`build_drive`, :func:`build_experiment`) turn a spec
into the live ``fracres`` objects, so an experiment is fully determined by its
config plus the top-level integer ``seed``.

Variant-specific reservoir hyper-parameters (``spectral_scale``, ``decay``,
``tau_E``/``tau_I``, ``sigma_e`` ...) go in :attr:`ModelConfig.params`, a plain
mapping forwarded as constructor keyword arguments; an unknown key surfaces as a
``TypeError`` from the model constructor at build time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import yaml

from fracres.drivers import generate_fbm_increments
from fracres.kernels import GLKernel, L1CaputoKernel
from fracres.models import (
    NeuralFieldPhantomBrain,
    PhantomBrain,
    WilsonCowanPhantomBrain,
    qSOCPhantomBrain,
)

_KERNELS = {"gl": GLKernel, "l1": L1CaputoKernel}
_MODELS = {
    "phantom": PhantomBrain,
    "qsoc": qSOCPhantomBrain,
    "wilson_cowan": WilsonCowanPhantomBrain,
    "neural_field": NeuralFieldPhantomBrain,
}
_TRAIN_METHODS = ("ridge", "gradient")


@dataclass
class KernelConfig:
    """Fractional memory kernel: ``kind`` (``gl``/``l1``), order, history depth."""

    kind: str = "gl"
    alpha: float = 0.8
    history_length: int = 100

    def __post_init__(self):
        if self.kind not in _KERNELS:
            raise ValueError(
                f"unknown kernel kind {self.kind!r}; choose from {sorted(_KERNELS)}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.history_length < 2:
            raise ValueError(f"history_length must be >= 2, got {self.history_length}")


@dataclass
class ModelConfig:
    """Model/reservoir variant, its I/O sizes, and variant-specific ``params``."""

    kind: str = "phantom"
    in_features: int = 1
    res_size: int = 200
    out_features: int = 1
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _MODELS:
            raise ValueError(
                f"unknown model kind {self.kind!r}; choose from {sorted(_MODELS)}"
            )
        for name in ("in_features", "res_size", "out_features"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclass
class DriveConfig:
    """Stochastic fGn drive: number of steps and Hurst exponent."""

    time_steps: int = 2000
    hurst: float = 0.7

    def __post_init__(self):
        if self.time_steps < 1:
            raise ValueError(f"time_steps must be >= 1, got {self.time_steps}")
        if not 0.0 < self.hurst < 1.0:
            raise ValueError(f"hurst must be in (0, 1), got {self.hurst}")


@dataclass
class TrainingConfig:
    """Readout fitting: ``ridge`` (closed form) or ``gradient`` (optax + Besov)."""

    method: str = "ridge"
    washout: int = 200
    beta: float = 1e-2  # ridge regularisation
    lr: float = 5e-3  # gradient learning rate
    epochs: int = 500  # gradient epochs
    lambda_reg: float = 1e-4  # gradient Besov-penalty weight
    alpha_stable: float = 2.0  # heavy-tail index of the drive (2 for fGn)

    def __post_init__(self):
        if self.method not in _TRAIN_METHODS:
            raise ValueError(
                f"unknown training method {self.method!r}; "
                f"choose from {list(_TRAIN_METHODS)}"
            )
        if self.washout < 0:
            raise ValueError(f"washout must be >= 0, got {self.washout}")


@dataclass
class ExperimentConfig:
    """Top-level spec: a name, a seed, and the kernel/model/drive/training blocks."""

    name: str = "experiment"
    seed: int = 0
    kernel: KernelConfig = field(default_factory=KernelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# --- serialisation ------------------------------------------------------------

def to_dict(config: ExperimentConfig) -> dict:
    """Plain nested-``dict`` view of a config (YAML/JSON ready)."""
    return asdict(config)


def from_dict(data: dict) -> ExperimentConfig:
    """Reconstruct an :class:`ExperimentConfig` from a nested dict (missing keys
    fall back to defaults). Validation runs via each block's ``__post_init__``.
    """
    data = data or {}
    return ExperimentConfig(
        name=data.get("name", "experiment"),
        seed=data.get("seed", 0),
        kernel=KernelConfig(**data.get("kernel", {})),
        model=ModelConfig(**data.get("model", {})),
        drive=DriveConfig(**data.get("drive", {})),
        training=TrainingConfig(**data.get("training", {})),
    )


def save_config(config: ExperimentConfig, path) -> None:
    """Write ``config`` to ``path`` as YAML (block style, declaration order)."""
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def load_config(path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file."""
    with open(path) as f:
        return from_dict(yaml.safe_load(f))


# --- factories ----------------------------------------------------------------

def build_kernel(cfg: KernelConfig):
    """Instantiate the fractional kernel described by ``cfg``."""
    return _KERNELS[cfg.kind](cfg.alpha, cfg.history_length)


def build_model(config: ExperimentConfig, key=None):
    """Instantiate the model (kernel + reservoir + readout) for ``config``.

    ``key`` defaults to ``PRNGKey(config.seed)``; ``model.params`` are forwarded
    to the model constructor as keyword arguments.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    kernel = build_kernel(config.kernel)
    m = config.model
    return _MODELS[m.kind](
        m.in_features, m.res_size, m.out_features, kernel, key=key, **m.params
    )


def build_drive(config: ExperimentConfig, key=None) -> jnp.ndarray:
    """Generate the ``(time_steps, in_features)`` fGn drive for ``config``.

    Each input feature is an independent fGn realisation of the configured Hurst
    exponent. ``key`` defaults to ``PRNGKey(config.seed + 1)`` (distinct from the
    model-init key).
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed + 1)
    f = config.model.in_features
    cols = [
        generate_fbm_increments(config.drive.time_steps, config.drive.hurst, k)
        for k in jax.random.split(key, f)
    ]
    return jnp.stack(cols, axis=1)


def build_experiment(config: ExperimentConfig):
    """Convenience: return ``(model, drive)`` built from ``config``."""
    return build_model(config), build_drive(config)
