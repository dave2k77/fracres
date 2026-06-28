"""Run a fracres experiment from a YAML config (KB reproducibility).

A whole experiment -- kernel, reservoir variant, drive, and training -- is
captured by ``configs/memory_task.yaml`` and rebuilt with the
:mod:`fracres.config` factories, so a run is reproducible from one file plus its
integer ``seed``. This script loads the spec, builds the model and drive, fits
the readout per the training block, reports held-out recall, and shows the config
round-trips back to YAML unchanged.

Run:  python examples/config_experiment.py
"""
from pathlib import Path

import jax.numpy as jnp

from fracres import (
    build_experiment,
    fit_readout_ridge,
    load_config,
    save_config,
    to_dict,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "memory_task.yaml"
DELAY = 3


def main():
    config = load_config(CONFIG_PATH)
    print(f"Loaded experiment '{config.name}' (seed={config.seed}) from")
    print(f"  {CONFIG_PATH}")
    print(f"  kernel={config.kernel.kind}(alpha={config.kernel.alpha})  "
          f"model={config.model.kind}(N={config.model.res_size})  "
          f"drive=fGn(H={config.drive.hurst}, T={config.drive.time_steps})")

    # Build the live objects straight from the spec.
    model, drive = build_experiment(config)
    split = int(0.7 * config.drive.time_steps)
    target = jnp.concatenate([jnp.zeros((DELAY, 1)), drive[:-DELAY]], axis=0)

    # Train per the config's training block (ridge here).
    t = config.training
    fitted = fit_readout_ridge(
        model, drive[:split], target[:split], beta=t.beta, washout=t.washout
    )
    pred = fitted(drive)
    corr = float(jnp.corrcoef(pred[split:, 0], target[split:, 0])[0, 1])
    print(f"\nTrained readout ({t.method}, beta={t.beta}): "
          f"held-out recall at delay {DELAY} = {corr:.3f}")

    # The spec round-trips back to YAML unchanged -- a faithful experiment record.
    out_path = Path(__file__).resolve().parent.parent / "configs" / "_roundtrip.yaml"
    save_config(config, out_path)
    reloaded = load_config(out_path)
    out_path.unlink()
    print(f"YAML round-trip identical: {reloaded == config}")
    print(f"config keys: {list(to_dict(config))}")


if __name__ == "__main__":
    main()
