#!/usr/bin/env python3
"""Is FNO's error concentrated at the window endpoints, as J-90 predicted?

THE PREDICTION, MADE BEFORE THE NUMBER EXISTED. PORT_LOG J-90 recorded, when FNO
was implemented and before any of it had trained, that the FFT treats the 12 h
window as periodic while `theta_TO(0) != theta_TO(T)`, so the spectral branch
sees a discontinuity at the window edge and pays Gibbs ringing there which the
local term `W` must cancel. Two consequences were stated:

    1. error should be worst at the window endpoints;
    2. the burden on `W` should grow with the endpoint mismatch.

Both are falsifiable and this measures them.

WHY IT MUST RUN BEFORE THE FACTORIAL CELLS EXIST. `33_fno_spectral_precheck.py`
found that adding the IEC baseline cuts the endpoint mismatch from 11.12 degC to
0.47 degC, and predicted that if the Gibbs mechanism is real then
FNO-with-baseline should improve by more than the other architectures gain from
the same change. This script establishes the **baseline condition** for that
prediction. Running it now means it cannot be influenced by knowing how the new
cells turn out.

THE CONTROL THAT MAKES IT A TEST RATHER THAN AN OBSERVATION. Every model here
satisfies the initial condition exactly, so error at `t = 0` is zero by
construction for all of them and the raw profile shape is not diagnostic. Two
things fix that:

  * each model's error profile is normalised by its own mean, so what is compared
    is **shape**, not size;
  * MIONet and S-DeepONet have no FFT. If endpoint concentration is a property of
    the problem — later times are simply harder — all three show it. **Only an
    excess specific to FNO is evidence for the spectral mechanism.**

The same applies to the correlation with endpoint mismatch: a larger mismatch
means a larger swing, which is harder for any architecture, so a positive
correlation is expected everywhere. Only FNO's correlation being *stronger* is
evidence.

Run:  python audit_port/scripts/34_endpoint_error.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "audit_port" / "ENDPOINT_ERROR.md"
CELLS = [
    ("FNO in-cascade", "fno2", True),
    ("MIONet in-cascade", "mionet", False),
    ("S-DeepONet in-cascade", "sdon", False),
    ("COD", "o5", False),
]
#: Last 10% of the window against the middle 50%, both on the normalised profile.
TAIL_FRAC = 0.10


def load(stem):
    p = ROOT / "artifacts" / stem / "predictions.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return d["pred"].astype(float), d["gt"].astype(float), d["t_eval"].astype(float)


def main() -> int:
    rows, profiles = [], {}
    for label, stem, has_fft in CELLS:
        got = load(stem)
        if got is None:
            print(f"[skip] {label}: no predictions.npz")
            continue
        pred, gt, t = got
        err = np.abs(pred[:, :, 0] - gt[:, :, 0])          # (n_cases, n_t)
        # Shape, not size: normalise each model by its own mean error.
        prof = err.mean(axis=0)
        prof_n = prof / prof.mean()
        profiles[label] = (t, prof_n)

        nt = len(t)
        k = max(1, int(TAIL_FRAC * nt))
        mid = slice(int(0.25 * nt), int(0.75 * nt))
        tail_ratio = float(prof_n[-k:].mean() / prof_n[mid].mean())
        head_ratio = float(prof_n[:k].mean() / prof_n[mid].mean())

        # Correlation of end-of-window error with the endpoint mismatch the FFT
        # would have to jump across.
        mismatch = np.abs(gt[:, -1, 0] - gt[:, 0, 0])
        end_err = err[:, -k:].mean(axis=1)
        r = float(np.corrcoef(mismatch, end_err)[0, 1])
        # The confound: a larger endpoint mismatch means a larger swing, and a
        # larger swing is harder for any architecture. Note also that
        # |theta(T) - theta(0)| <= swing always, so the two predictors are
        # correlated by construction and the raw correlations cannot be compared
        # directly.
        swing = gt[:, :, 0].max(axis=1) - gt[:, :, 0].min(axis=1)
        r_sw = float(np.corrcoef(swing, end_err)[0, 1])
        r_ms = float(np.corrcoef(mismatch, swing)[0, 1])
        # Partial correlation of end-of-window error with endpoint mismatch,
        # holding swing fixed. This is the statistic that discriminates: it asks
        # whether the periodicity violation predicts error BEYOND what ordinary
        # difficulty explains. The Gibbs mechanism predicts it is positive for a
        # spectral model and near zero for the others.
        den = np.sqrt(max((1 - r_sw ** 2) * (1 - r_ms ** 2), 1e-12))
        r_partial = float((r - r_sw * r_ms) / den)

        rows.append({"label": label, "fft": has_fft, "mae": float(err.mean()),
                     "tail_ratio": tail_ratio, "head_ratio": head_ratio,
                     "r_mismatch": r, "r_swing": r_sw,
                     "r_partial": r_partial, "n": err.shape[0]})

    if not rows:
        print("No predictions found.")
        return 1

    print(f"{'cell':24s} {'FFT':>4s} {'MAE':>8s} {'last10%/mid':>12s} "
          f"{'r(mism)':>9s} {'r(swing)':>9s} {'PARTIAL':>9s}")
    for r in rows:
        print(f"{r['label']:24s} {'yes' if r['fft'] else 'no':>4s} "
              f"{r['mae']:8.3f} {r['tail_ratio']:12.3f} "
              f"{r['r_mismatch']:9.3f} {r['r_swing']:9.3f} "
              f"{r['r_partial']:9.3f}")

    fno = next((r for r in rows if r["fft"]), None)
    others = [r for r in rows if not r["fft"]]
    verdict = "inconclusive"
    if fno and others:
        excess_tail = fno["tail_ratio"] - max(o["tail_ratio"] for o in others)
        # The PARTIAL correlation is the discriminating statistic: the raw one
        # cannot separate "the periodicity violation hurts this model" from
        # "harder cases hurt every model", because mismatch and swing are
        # correlated by construction.
        excess_r = fno["r_partial"] - max(o["r_partial"] for o in others)
        print(f"\nFNO tail concentration minus the largest non-FFT cell : "
              f"{excess_tail:+.3f}")
        print(f"FNO partial r minus the largest non-FFT cell          : "
              f"{excess_r:+.3f}")
        if excess_tail > 0.10 and excess_r > 0.10:
            verdict = "supported"
        elif excess_tail <= 0.0 and excess_r <= 0.0:
            verdict = "refuted"
        else:
            verdict = "partial"
        print(f"\nJ-90 endpoint mechanism: {verdict.upper()}")

    md = ["# Is FNO's error concentrated at the window endpoints?\n",
          "Generated by `audit_port/scripts/34_endpoint_error.py`, **before the "
          "factorial cells existed**, so it cannot have been influenced by "
          "knowing how they turn out.\n",
          "## The prediction under test\n",
          "PORT_LOG J-90, written when FNO was implemented and before it had "
          "trained: the FFT treats the 12 h window as periodic while "
          "`theta_TO(0) != theta_TO(T)`, so the spectral branch pays Gibbs "
          "ringing at the edge which `W` must cancel. Therefore error should be "
          "worst at the endpoints, and should grow with the endpoint mismatch.\n",
          "## The control\n",
          "Every model satisfies the IC exactly, so error at `t = 0` is zero by "
          "construction and raw profile shape is not diagnostic. Each profile is "
          "normalised by its own mean, so **shape** is compared rather than size; "
          "and MIONet, S-DeepONet and COD have no FFT, so if endpoint "
          "concentration is a property of the problem all of them show it. Only "
          "an **excess specific to FNO** is evidence. The same logic applies to "
          "the correlation: a larger endpoint mismatch means a larger swing, "
          "which is harder for everyone.\n",
          "| cell | FFT | thermal MAE | last 10% / middle | "
          "r(mismatch) | r(swing) | **partial r, swing held fixed** |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {'yes' if r['fft'] else 'no'} | "
                  f"{r['mae']:.3f} | {r['tail_ratio']:.3f} | "
                  f"{r['r_mismatch']:.3f} | {r['r_swing']:.3f} | "
                  f"**{r['r_partial']:.3f}** |")
    md.append("")
    md.append("## Error profile across the window, normalised by each cell's mean\n")
    md.append("| t/T | " + " | ".join(lbl for lbl in profiles) + " |")
    md.append("|---" * (1 + len(profiles)) + "|")
    any_t = next(iter(profiles.values()))[0]
    for j in range(0, len(any_t), max(1, len(any_t) // 10)):
        cells = " | ".join(f"{profiles[l][1][j]:.2f}" for l in profiles)
        md.append(f"| {any_t[j] / any_t[-1]:.2f} | {cells} |")
    md.append("")
    md.append("## Verdict\n")
    if verdict == "supported":
        md.append("**Supported.** FNO shows endpoint concentration and "
                  "mismatch-correlation in excess of every cell without an FFT, "
                  "which is what the spectral mechanism predicts and what "
                  "'later times are harder' does not.\n")
    elif verdict == "refuted":
        md.append("**Refuted.** FNO's endpoint concentration and "
                  "mismatch-correlation do not exceed the cells without an FFT. "
                  "Whatever costs FNO its accuracy, it is not the Gibbs mechanism "
                  "J-90 predicted, and that prediction is withdrawn rather than "
                  "reinterpreted.\n")
    else:
        md.append("**Partial.** One of the two signatures exceeds the non-FFT "
                  "cells and the other does not, so the mechanism is neither "
                  "established nor excluded on this evidence. The factorial's "
                  "with-baseline cells are the sharper test: "
                  "`33_fno_spectral_precheck.py` measured that the baseline cuts "
                  "the endpoint mismatch 24-fold, so if the mechanism is real "
                  "FNO should gain more from it than MIONet and S-DeepONet do.\n")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
