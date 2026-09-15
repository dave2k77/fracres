"""Load real resting-state EEG (OpenNeuro ds002778) and take a first look.

Demonstrates the Next-1 ingestion path: participants metadata, session-aware
loading (PD subjects have medication OFF/ON sessions), and a first LRD
statistic on the loaded traces using fracres's existing metrics -- the data
side of the comparison the forward-fit workflow will formalise.

Run:  python examples/load_eeg.py
(needs the optional 'data' extra and the local mirror at data/ds002778)
"""
from __future__ import annotations

import numpy as np

from fracres.data import dataset_root, load_participants, load_subject
from fracres.metrics import hurst_dfa, spectral_exponent


def channel_stats(rec, names=("Cz", "O1")):
    """DFA Hurst + spectral exponent for a few channels of one recording."""
    out = {}
    for name in names:
        trace = rec.channel(name)
        # Demean; BioSemi recordings are unreferenced (large DC offsets).
        trace = trace - trace.mean()
        out[name] = (hurst_dfa(trace), spectral_exponent(trace))
    return out


def main():
    root = dataset_root()
    participants = load_participants(root)
    n_pd = sum(p.is_pd for p in participants)
    n_hc = len(participants) - n_pd
    print(f"{len(participants)} participants ({n_pd} PD, {n_hc} HC)")

    pd_sub = next(p for p in participants if p.is_pd).subject
    hc_sub = next(p for p in participants if not p.is_pd).subject

    recordings = {
        f"{hc_sub} (HC)": load_subject(root, hc_sub, resample=256.0),
        f"{pd_sub} OFF meds": load_subject(root, pd_sub, session="off", resample=256.0),
        f"{pd_sub} ON meds": load_subject(root, pd_sub, session="on", resample=256.0),
    }
    for label, rec in recordings.items():
        stats = channel_stats(rec)
        line = "  ".join(
            f"{ch}: DFA-H={h:.3f}, beta={b:.2f}" for ch, (h, b) in stats.items()
        )
        print(
            f"{label:18s} {rec.n_channels}ch {rec.duration:.0f}s "
            f"@{rec.sfreq:.0f}Hz | {line}"
        )

    # Positions are ready for a spatially-embedded readout later.
    rec = next(iter(recordings.values()))
    data_jax, pos_jax = rec.to_jax()
    print(f"\nJAX arrays: data{tuple(data_jax.shape)}, positions{tuple(pos_jax.shape)}")
    assert np.isfinite(rec.data).all()


if __name__ == "__main__":
    main()
