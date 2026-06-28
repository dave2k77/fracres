"""Experiment configuration: dataclasses, YAML round-trip, and factories."""
import jax.numpy as jnp
import pytest

from fracres import (
    DriveConfig,
    ExperimentConfig,
    GLKernel,
    KernelConfig,
    L1CaputoKernel,
    ModelConfig,
    PhantomBrain,
    TrainingConfig,
    WilsonCowanPhantomBrain,
    build_drive,
    build_experiment,
    build_kernel,
    build_model,
    from_dict,
    load_config,
    save_config,
    to_dict,
)

# --- construction & validation ------------------------------------------------

def test_defaults_construct():
    c = ExperimentConfig()
    assert c.kernel.kind == "gl" and c.model.kind == "phantom"
    assert c.training.method == "ridge"


@pytest.mark.parametrize("block,kwargs", [
    (KernelConfig, {"kind": "bogus"}),
    (KernelConfig, {"alpha": 1.5}),
    (KernelConfig, {"alpha": 0.0}),
    (KernelConfig, {"history_length": 1}),
    (ModelConfig, {"kind": "bogus"}),
    (ModelConfig, {"res_size": 0}),
    (DriveConfig, {"hurst": 1.0}),
    (DriveConfig, {"time_steps": 0}),
    (TrainingConfig, {"method": "bogus"}),
    (TrainingConfig, {"washout": -1}),
])
def test_validation_rejects_bad_values(block, kwargs):
    with pytest.raises(ValueError):
        block(**kwargs)


# --- serialisation ------------------------------------------------------------

def test_yaml_roundtrip(tmp_path):
    c = ExperimentConfig(
        name="demo", seed=7,
        kernel=KernelConfig(kind="l1", alpha=0.6, history_length=40),
        model=ModelConfig(kind="wilson_cowan", res_size=32, params={"tau_I": 4.0}),
        drive=DriveConfig(time_steps=500, hurst=0.65),
        training=TrainingConfig(method="gradient", lr=1e-2, epochs=100),
    )
    path = tmp_path / "exp.yaml"
    save_config(c, path)
    assert load_config(path) == c  # dataclasses compare field-by-field


def test_from_to_dict_roundtrip():
    c = ExperimentConfig(name="x", seed=2, model=ModelConfig(res_size=64))
    assert from_dict(to_dict(c)) == c


def test_from_dict_partial_fills_defaults():
    c = from_dict({"model": {"res_size": 50}})
    assert c.model.res_size == 50
    assert c.kernel.kind == "gl"  # default block
    assert c.seed == 0


def test_load_validates(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("kernel:\n  alpha: 1.5\n")
    with pytest.raises(ValueError):
        load_config(path)


# --- factories ----------------------------------------------------------------

def test_build_kernel_types():
    assert isinstance(build_kernel(KernelConfig(kind="gl")), GLKernel)
    k = build_kernel(KernelConfig(kind="l1", alpha=0.6, history_length=30))
    assert isinstance(k, L1CaputoKernel)
    assert k.alpha == 0.6 and k.history_length == 30


@pytest.mark.parametrize("kind,cls", [
    ("phantom", PhantomBrain),
    ("wilson_cowan", WilsonCowanPhantomBrain),
])
def test_build_model_variants(kind, cls):
    c = ExperimentConfig(model=ModelConfig(kind=kind, res_size=16))
    assert isinstance(build_model(c), cls)


def test_build_model_forwards_params():
    c = ExperimentConfig(model=ModelConfig(res_size=16, params={"decay": 2.0}))
    assert float(build_model(c).reservoir.decay) == 2.0


def test_build_model_unknown_param_raises():
    c = ExperimentConfig(model=ModelConfig(res_size=16, params={"bogus": 1.0}))
    with pytest.raises(TypeError):
        build_model(c)


def test_build_drive_shape_multifeature():
    c = ExperimentConfig(
        model=ModelConfig(in_features=3, res_size=16),
        drive=DriveConfig(time_steps=128, hurst=0.7),
    )
    assert build_drive(c).shape == (128, 3)


def test_build_experiment_runs():
    c = ExperimentConfig(
        model=ModelConfig(res_size=24), drive=DriveConfig(time_steps=150)
    )
    model, drive = build_experiment(c)
    X, Y = model.simulate(drive)
    assert X.shape == (150, 24) and Y.shape == (150, 1)
    assert jnp.all(jnp.isfinite(X))


def test_seed_controls_reproducibility():
    same = [build_model(ExperimentConfig(seed=5, model=ModelConfig(res_size=16)))
            for _ in range(2)]
    assert jnp.allclose(same[0].reservoir.W_res, same[1].reservoir.W_res)
    other = build_model(ExperimentConfig(seed=6, model=ModelConfig(res_size=16)))
    assert not jnp.allclose(same[0].reservoir.W_res, other.reservoir.W_res)
