"""Forward-fit on real EEG: does the phantom reproduce the data's statistics?

Drives a frozen fractional reservoir with one real channel (Fz) of the ds002778
Parkinson's recording and fits the readout to reconstruct the other channels --
once with the patient OFF dopaminergic medication, once ON. For each session it
reports held-out predictive skill (mean correlation) and the per-channel LRD
statistics of the model output against the recorded channels: agreement of
DFA-H / spectral-beta -- not pointwise prediction -- is the success criterion
(ROADMAP Next 1).

Run:  python examples/forward_fit_eeg.py
(needs the optional 'data' extra and the local mirror at data/ds002778)
"""
from __future__ import annotations

import jax

from fracres.data import dataset_root, load_subject
from fracres.forward_fit import forward_fit
from fracres.kernels import GLKernel
from fracres.models import PhantomBrain

DRIVE = ["Fz"]
TARGETS = ["Cz", "Pz", "O1", "F3", "F4", "T7", "T8"]
RES_SIZE = 300
SUBJECT = "sub-pd3"


def fit_session(root, session, seed=0):
    rec = load_subject(root, SUBJECT, session=session, resample=128.0)
    kernel = GLKernel(alpha=0.8, history_length=100)
    model = PhantomBrain(
        len(DRIVE), RES_SIZE, len(TARGETS), kernel,
        key=jax.random.PRNGKey(seed), step_size=0.4, decay=2.0,
    )
    return rec, forward_fit(rec, DRIVE, TARGETS, model)


def main():
    root = dataset_root()
    for session in ("off", "on"):
        rec, result = fit_session(root, session)
        print(
            f"\n=== {SUBJECT} ses-{session} "
            f"({rec.duration:.0f}s @ {rec.sfreq:.0f}Hz, drive={DRIVE[0]}) ==="
        )
        print(
            f"held-out reconstruction: mean corr = {result.mean_correlation:.3f} "
            f"over {len(TARGETS)} channels"
        )
        print(result.metrics_table())

    print(
        "\nSuccess = model columns track data columns per channel and session;\n"
        "the OFF->ON shift is the medication contrast the inverse problem will\n"
        "later attribute to a change in (alpha, H) of the underlying dynamics."
    )


if __name__ == "__main__":
    main()
