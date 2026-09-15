"""Mechanism inference on real EEG: the OFF/ON medication contrast.

For one Parkinson's subject, asks which generative mechanism -- phantom
fractional order alpha, drive Hurst H, reservoir leak -- reproduces the Cz
channel's LRD statistics, once OFF and once ON dopaminergic medication. The
forward-fit workflow showed no single (alpha, H) matches both sessions; this
grid search quantifies the shift directly.

Run:  python examples/inverse_problem_eeg.py
(needs the optional 'data' extra and the local mirror at data/ds002778;
a few minutes of compute: 2 sessions x 50 grid points x 2 seeds)
"""
from __future__ import annotations

from fracres.data import dataset_root, load_subject
from fracres.inverse import grid_search

SUBJECT = "sub-pd3"
CHANNEL = "Cz"
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
HURSTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DECAYS = [1.0, 2.0]
DRIVE_KIND = "fbm"  # raw EEG is fBm-like (beta > 1); fGn cannot reach it
N_SEEDS = 2


def main():
    root = dataset_root()
    bests = {}
    for session in ("off", "on"):
        rec = load_subject(root, SUBJECT, session=session, resample=64.0)
        print(f"\n=== {SUBJECT} ses-{session}, channel {CHANNEL} "
              f"({rec.duration:.0f}s @ {rec.sfreq:.0f}Hz) ===")
        result = grid_search(
            rec, CHANNEL, ALPHAS, HURSTS, DECAYS, drive_kind=DRIVE_KIND,
            n_seeds=N_SEEDS, res_size=200,
        )
        print(result.leaderboard(5))
        bests[session] = result.best

    off, on = bests["off"], bests["on"]
    print("\n=== medication contrast (best-fit mechanism) ===")
    print(f"OFF: alpha={off.alpha:.2f}  H={off.hurst:.2f}  decay={off.decay:.1f}"
          f"  (obj={off.objective:.3f})")
    print(f"ON : alpha={on.alpha:.2f}  H={on.hurst:.2f}  decay={on.decay:.1f}"
          f"  (obj={on.objective:.3f})")
    print(
        "\nReading: the best-fit fractional order / drive Hurst moving with\n"
        "medication state is the candidate mechanism-level description of the\n"
        "dopaminergic effect on cortical LRD. A single subject is a demo, not\n"
        "a result -- the study runs this per subject x channel."
    )


if __name__ == "__main__":
    main()
