#!/usr/bin/env python3
"""Is gas NMAE still a meaningful number on the realistic T1 benchmark?

The O-5 rerun reports the `eval/benchmark.py` denominator floor binding on 75% of
`c_C2H2` cases, against 38% on the v57 benchmark, with a median denominator of
exactly 1.0e-4 — the floor itself. A median sitting *on* the floor means more than
half the cases are being divided by a constant that has nothing to do with the
case, so the resulting percentage is not an error measure at all: it is
`MAE / 1e-4`, a rescaled absolute error wearing a percent sign.

This is a property of the **ground truth alone**, so it needs no model and no
checkpoint. The gas states barely move over a 12 h window — that is physics, not
a modelling failure — and the realistic sampler made it more pronounced, because
consistent ICs put each unit near its own gas equilibrium instead of 30 degC away
from it.

DECISIONS C-9 already says physical units are primary and NMAE secondary with the
floor-hit rate attached. This measures whether "secondary" is still too generous
for the gases on this benchmark.

Run:  python audit_port/scripts/28_gas_nmae_floor.py
Exit: 0 if every state's NMAE is usable on both benchmarks, 1 if any gas NMAE is
      degenerate — which is the expected outcome and the point of recording it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import (  # noqa: E402
    build_realistic_test_set, build_test_set, rk45_ground_truth,
)
from cod.data.physics import STATE_NAMES_FAST, TW  # noqa: E402
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.data.steady_state import formula_A  # noqa: E402
from cod.eval.metrics import TRANSFORMER_STATES  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "GAS_NMAE_FLOOR.md"

BENCH_FLOOR = 1e-4      # eval/benchmark.py, the notebook NMAE floor
N_EVAL = 50
DEGENERATE_FRAC = 0.5   # floor binding on more than half the cases


def variations(cases, T=TW):
    """Per-case peak-to-peak of the ground truth, (n_cases, 6)."""
    t_eval = np.linspace(0, T, N_EVAL)
    out = np.zeros((len(cases), 6))
    for i, c in enumerate(cases):
        gt = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval, T=T)
        out[i] = gt.max(axis=0) - gt.min(axis=0)
    return out


def main() -> int:
    params = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])

    print("[data] building both benchmarks (100 cases each)")
    sets = {
        "realistic T1 (fix-7 sampler, seed 999)":
            build_realistic_test_set(n_test=100, seed=999, params=params),
        "v57 benchmark (seed 999, formula_A)":
            build_test_set(n_test=100, seed=999, steady_state=formula_A),
    }

    results = {}
    for label, cases in sets.items():
        print(f"[gt]  {label}")
        results[label] = variations(cases)

    print(f"\n=== denominator floor ({BENCH_FLOOR:g}, eval/benchmark.py) ===")
    hdr = f"{'state':>10} " + " ".join(f"{l.split(' (')[0]:>26}" for l in sets)
    print(hdr)
    failures = []
    rows = {}
    for i, name in enumerate(STATE_NAMES_FAST[:6]):
        cells, rows[name] = [], {}
        spec = TRANSFORMER_STATES[i]
        thr = spec.engineering_threshold
        for label, var in results.items():
            v = var[:, i]
            frac = float((v < BENCH_FLOOR).mean())
            med = float(np.median(v))
            # Second criterion, independent of the floor. Even where the
            # denominator clears 1e-4 it can still be a physically negligible
            # excursion, and normalising by that produces a ratio whose numerator
            # and denominator are both far below anything an instrument resolves.
            # `engineering_threshold` is what a practitioner would need to
            # measure, so the denominator as a fraction of it says whether the
            # normalisation has any physical referent at all.
            rel_thr = (med / thr) if thr else float("nan")
            rows[name][label] = {"floor_frac": frac, "median": med,
                                 "median_on_floor": med <= BENCH_FLOOR,
                                 "median_over_threshold": rel_thr}
            cells.append(f"{100 * frac:5.1f}% floor, med {med:.3g}")
            if name == "theta_TO":
                continue
            if frac > DEGENERATE_FRAC:
                failures.append(
                    f"{name} on {label.split(' (')[0]}: floor binds on "
                    f"{100 * frac:.0f}% of cases, median variation {med:.3g} "
                    f"({med / BENCH_FLOOR:.2f}x the floor)")
            elif thr and rel_thr < 0.01:
                failures.append(
                    f"{name} on {label.split(' (')[0]}: floor rarely binds "
                    f"({100 * frac:.0f}%) but the median variation {med:.3g} "
                    f"{spec.unit} is {100 * rel_thr:.2f}% of the "
                    f"{thr:g} {spec.unit} engineering threshold — a well-defined "
                    "ratio to a physically negligible excursion")
        print(f"{name:>10} " + " ".join(f"{c:>26}" for c in cells))

    # ── Report ─────────────────────────────────────────────────────────────
    md = ["# Is gas NMAE usable on the realistic benchmark?\n",
          "Generated by `audit_port/scripts/28_gas_nmae_floor.py`. Ground truth "
          f"by RK45 over the {TW:g} min window at {N_EVAL} points, 100 cases per "
          "benchmark. **No model is involved** — the normalisation denominator is "
          "a property of the ground truth, so this holds for every model scored "
          "on these sets.\n",
          f"`eval/benchmark.py` normalises by per-case peak-to-peak with a floor "
          f"of {BENCH_FLOOR:g}. When the floor binds, the reported percentage is "
          f"`MAE / {BENCH_FLOOR:g}` — a rescaled absolute error wearing a percent "
          "sign, carrying no information about the case it came from.\n",
          "## Fraction of cases where the floor binds\n",
          "| state | " + " | ".join(l for l in sets) + " |",
          "|---|" + "---|" * len(sets)]
    for name in STATE_NAMES_FAST[:6]:
        cells = []
        for label in sets:
            r = rows[name][label]
            mark = " **(below the floor)**" if r["median_on_floor"] else ""
            cells.append(f"{100 * r['floor_frac']:.0f}%, median variation "
                         f"{r['median']:.4g}{mark}")
        md.append(f"| `{name}` | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## The denominator against what a practitioner can measure\n")
    md.append("The floor is not the only way a normalisation loses meaning. Even "
              "where the denominator clears 1e-4 it can still be an excursion far "
              "below anything an instrument resolves, and a ratio between two such "
              "numbers is well defined and physically empty. Median 12 h variation "
              "on the realistic T1 set, against the engineering threshold each "
              "state carries in `cod/eval/metrics.py`:\n")
    md.append("| state | median variation | engineering threshold | variation as "
              "% of threshold |")
    md.append("|---|---|---|---|")
    for i, name in enumerate(STATE_NAMES_FAST[:6]):
        spec = TRANSFORMER_STATES[i]
        r = rows[name][list(sets)[0]]
        if spec.engineering_threshold is None:
            continue
        md.append(f"| `{name}` | {r['median']:.4g} {spec.unit} | "
                  f"{spec.engineering_threshold:g} {spec.unit} | "
                  f"**{100 * r['median_over_threshold']:.3f}%** |")
    md.append("")

    md.append("## Reading\n")
    md.append("The realistic sampler did not break the gas metric; it exposed "
              "how little the gases move. Consistent initial conditions start "
              "each unit near its own gas equilibrium instead of up to 30 degC "
              "away from it (audit M-9), so the 12 h transient that the old "
              "sampler's mismatched ICs produced is gone. The gases genuinely "
              "barely change over half a day — that is the physics of "
              "`k_gen V_arr / k_dis` against a 12 h window, not a modelling "
              "failure — and normalising by a quantity that is almost zero is "
              "what manufactures the large percentages.\n")
    md.append("This is the same defect the audit found at 34,558% on acetylene, "
              "now arriving from the other direction: there it inflated a "
              "monolithic baseline's error, here it would flatter or distort any "
              "model's. Neither number means anything.\n")

    md.append("## Consequence\n")
    if failures:
        md.append("**Gas NMAE is not reportable from this benchmark.** Quote "
                  "absolute ppm against the IEC 60599 thresholds in "
                  "`cod/eval/metrics.py`, which is what C-9 already makes the "
                  "primary metric. The percentage columns for the five gases "
                  "should not appear in the paper at all — not as a secondary "
                  "figure, not in an appendix — because a reader cannot tell "
                  "which entries are ratios and which are `MAE / 1e-4`.\n")
        md.append("`theta_TO` is unaffected: its variation is of order 10 degC "
                  "and the floor never binds, so thermal NMAE remains "
                  "meaningful.\n")
        for f in failures:
            md.append(f"* {f}")
        md.append("")
    else:
        md.append("Every state's denominator stays clear of the floor on both "
                  "benchmarks; NMAE remains usable as a secondary metric.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    if failures:
        print("\nDEGENERATE (expected — this is the finding):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
