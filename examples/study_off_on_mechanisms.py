"""Study: OFF/ON medication contrast of the inferred mechanism, all subjects.

Population-level version of examples/inverse_problem_eeg.py: for every ds002778
subject and session, grid-search the best-fit generative mechanism
(alpha, H, decay) for each analysed channel, then summarise the medication
contrast at the group level (paired across the 15 PD subjects, HC as reference).

Refinements over the single-subject demo, per the lessons note:
- alpha axis extended to the boundary region {0.85, 0.9, 0.95, 1.0}
  (the single-subject best sat at the grid edge);
- H at 0.05 spacing over [0.45, 0.80] (the region both sessions landed in);
- 3 seeds per grid point (common random numbers across the grid);
- every recording cropped to the same 120 s so estimator finite-sample biases
  are comparable across subjects *and* sessions.

Output: outputs/off_on_mechanisms.csv (one row per subject x session x channel;
incremental + resumable -- already-written rows are skipped), then a group
summary with a Wilcoxon signed-rank test on per-subject best-fit H, OFF vs ON.

Run:  python examples/study_off_on_mechanisms.py   (~45 min; safe to re-run)
"""
from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path

import numpy as np

from fracres.data import dataset_root, list_sessions, load_participants, load_subject
from fracres.inverse import grid_search

CHANNELS = ["Fz", "Cz", "Pz"]
ALPHAS = [0.85, 0.9, 0.95, 1.0]
HURSTS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
DECAYS = [1.0]
N_SEEDS = 3
RESAMPLE = 64.0
T_CROP = int(120 * RESAMPLE)  # 7680 samples: uniform length for every recording

OUT = Path(__file__).resolve().parents[1] / "outputs" / "off_on_mechanisms.csv"
FIELDS = [
    "subject", "group", "session", "channel",
    "alpha", "hurst", "decay", "objective",
    "data_dfa", "data_beta", "model_dfa", "model_beta",
]


def done_rows(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["subject"], r["session"], r["channel"]) for r in csv.DictReader(f)}


def main():
    root = dataset_root()
    participants = load_participants(root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_rows(OUT)
    new_file = not OUT.exists()

    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for p in participants:
            for ses in list_sessions(root, p.subject):
                needed = [c for c in CHANNELS if (p.subject, ses, c) not in done]
                if not needed:
                    continue
                t0 = time.time()
                rec = load_subject(root, p.subject, session=ses, resample=RESAMPLE)
                rec = dataclasses.replace(rec, data=rec.data[:, :T_CROP])
                for ch in needed:
                    result = grid_search(
                        rec, ch, ALPHAS, HURSTS, DECAYS,
                        drive_kind="fbm", n_seeds=N_SEEDS, res_size=200,
                    )
                    best, dm = result.best, result.data_metrics
                    writer.writerow({
                        "subject": p.subject, "group": p.group, "session": ses,
                        "channel": ch, "alpha": best.alpha, "hurst": best.hurst,
                        "decay": best.decay, "objective": f"{best.objective:.4f}",
                        "data_dfa": f"{dm.hurst_dfa:.4f}",
                        "data_beta": f"{dm.spectral_beta:.4f}",
                        "model_dfa": f"{best.model_metrics.hurst_dfa:.4f}",
                        "model_beta": f"{best.model_metrics.spectral_beta:.4f}",
                    })
                    f.flush()
                print(f"{p.subject} {ses}: {len(needed)} channels in "
                      f"{time.time() - t0:.0f}s", flush=True)

    summarise()


def summarise():
    from scipy import stats

    with open(OUT, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for ch in CHANNELS:
        print(f"\n=== {ch} ===")
        for group, ses in (("hc", "hc"), ("pd", "off"), ("pd", "on")):
            sel = [r for r in rows if r["group"] == group and r["session"] == ses
                   and r["channel"] == ch]
            if not sel:
                continue
            h = [float(r["hurst"]) for r in sel]
            a = [float(r["alpha"]) for r in sel]
            o = [float(r["objective"]) for r in sel]
            print(f"  {group}/{ses:3s} n={len(sel):2d}  "
                  f"H={np.mean(h):.3f}+-{np.std(h):.3f}  "
                  f"alpha={np.mean(a):.3f}+-{np.std(a):.3f}  "
                  f"obj={np.mean(o):.3f}")
        # Paired medication contrast across PD subjects.
        off = {r["subject"]: float(r["hurst"]) for r in rows
               if r["group"] == "pd" and r["session"] == "off" and r["channel"] == ch}
        on = {r["subject"]: float(r["hurst"]) for r in rows
              if r["group"] == "pd" and r["session"] == "on" and r["channel"] == ch}
        paired = [(off[s], on[s]) for s in off if s in on]
        if len(paired) >= 6:
            d = np.array([o - f for f, o in paired])
            w = stats.wilcoxon([f for f, _ in paired], [o for _, o in paired])
            print(f"  PD OFF->ON: dH={d.mean():+.3f} (median {np.median(d):+.3f}), "
                  f"Wilcoxon p={w.pvalue:.4f}, n={len(paired)}")


if __name__ == "__main__":
    main()
