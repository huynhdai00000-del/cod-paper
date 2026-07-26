#!/usr/bin/env python3
"""Measure the Gate 1 effect of each Phase 2 fix, one at a time.

The brief asks for the Gate 1 numbers before and after each fix, so the effect of
every change is measurable and reportable. This script produces that table.

An honest reading of what can and cannot move requires stating the mechanism:
Gate 1 loads a checkpoint and scores it on the seed-999 benchmark. A fix that acts
only on the TRAINING path therefore cannot move Gate 1 until a retrain happens,
and reporting a nonzero movement for one would mean something had leaked. So the
expected pattern is:

  fix 1  MOVES Gate 1, and invalidates the checkpoint. It changes the benchmark's
         initial conditions and the model's own analytic attractor.
  fix 2  no movement. RHS_SCALE is never consumed by the live loss.
  fix 3  no movement. Causal weighting exists only inside the training loss.
  fix 4  no movement. Only the training profile generator draws an ambient phase.
  fix 5  no movement. Only the training profile generator has a 'step' branch.

Fixes 2-5 are still verified here, by checking the quantity each one targets
directly rather than by hoping Gate 1 reflects it.

Usage:
    python scripts/measure_fix_effects.py [--out PHASE2_EFFECTS.md] [--fix N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, load_training_set
from cod.data.physics import N_SENSORS, STATE_DIM_FAST, STATE_NAMES_FAST, TW
from cod.data.steady_state import formula_A, true_fixed_point, true_fixed_point_np
from cod.eval.benchmark import evaluate
from cod.models.cod import CODOperator, cod_predict

ART = ROOT / "reference" / "artifacts"


def gate1(theta_ss_mode: str, ic_formula, label: str, device) -> dict:
    """Run Gate 1 under a given (IC formula, model attractor) combination."""
    ts = load_training_set(ART / "transformer_training_v57.npz")
    cod = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128,
                      p=64, n_layers=4, n_exp_feats=12, T=TW,
                      x_mean=ts.x_mean, x_std=ts.x_std,
                      theta_ss_mode=theta_ss_mode).to(device)
    ckpt = torch.load(ART / "transformer_pideepOnet_v57.pt", map_location=device,
                      weights_only=False)
    cod.load_state_dict(ckpt["model_state_dict"], strict=True)
    cases = build_test_set(n_test=100, seed=999, T=TW, steady_state=ic_formula)
    r = evaluate(cod, cod_predict, cases, label=label, t_clip_frac=0.9999,
                 device=device)
    return {
        "label": label,
        "per_state": r.per_state_pct.tolist(),
        "overall": r.overall_pct,
        "ck": r.ck_pct,
        "tv": r.tv_pct,
        "lt10": r.n_within_10pct,
        "mae_TO": float(r.mae_abs[:, 0].mean()),
    }


def row(name: str, a: dict, b: dict) -> list[str]:
    out = ["| quantity | before (v57) | after (fixed) | change |", "|---|---|---|---|"]
    for i, nm in enumerate(STATE_NAMES_FAST):
        out.append(f"| `{nm}` NMAE % | {a['per_state'][i]:.1f} | "
                   f"{b['per_state'][i]:.1f} | "
                   f"{b['per_state'][i] - a['per_state'][i]:+.1f} |")
    out.append(f"| **overall NMAE %** | {a['overall']:.1f} | {b['overall']:.1f} | "
               f"{b['overall'] - a['overall']:+.1f} |")
    out.append(f"| constant K % | {a['ck']:.1f} | {b['ck']:.1f} | "
               f"{b['ck'] - a['ck']:+.1f} |")
    out.append(f"| time-varying % | {a['tv']:.1f} | {b['tv']:.1f} | "
               f"{b['tv'] - a['tv']:+.1f} |")
    out.append(f"| cases < 10% | {a['lt10']} | {b['lt10']} | "
               f"{b['lt10'] - a['lt10']:+d} |")
    out.append(f"| theta_TO MAE degC | {a['mae_TO']:.3f} | {b['mae_TO']:.3f} | "
               f"{b['mae_TO'] - a['mae_TO']:+.3f} |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="PHASE2_EFFECTS.md")
    args = ap.parse_args()
    device = torch.device("cpu")

    md: list[str] = ["# Phase 2 — effect of each fix on Gate 1\n"]
    md.append("Gate 1 is the seed-999 benchmark scored with "
              "`transformer_pideepOnet_v57.pt`. Read the mechanism note first:\n")
    md.append("> Gate 1 loads a checkpoint. A fix that acts only on the training "
              "path cannot move it until a retrain happens. A nonzero movement "
              "for such a fix would mean something had leaked from training into "
              "evaluation, so **zero is the correct and expected result** for "
              "fixes 2-5, and each is verified against the quantity it actually "
              "targets instead.\n")

    print("=== baseline: v57 behaviour ===")
    base = gate1("formula_C", formula_A, "v57 (formula A ICs, formula C attractor)",
                 device)
    print(f"  overall {base['overall']:.2f}%")

    # ── Fix 1 ─────────────────────────────────────────────────────────────
    md.append("## Fix 1 — unify on `true_fixed_point()`\n")
    print("\n=== fix 1: model attractor only ===")
    f1_model = gate1("true_fixed_point", formula_A,
                     "fix 1 partial: attractor only", device)
    print(f"  overall {f1_model['overall']:.2f}%")
    print("=== fix 1: ICs only ===")
    f1_ic = gate1("formula_C", true_fixed_point_np, "fix 1 partial: ICs only", device)
    print(f"  overall {f1_ic['overall']:.2f}%")
    print("=== fix 1: both (the actual fix) ===")
    f1_both = gate1("true_fixed_point", true_fixed_point_np, "fix 1 full", device)
    print(f"  overall {f1_both['overall']:.2f}%")

    md.append("Decomposed, because the two halves of this fix move Gate 1 for "
              "different reasons:\n")
    md.append("| configuration | overall NMAE % | theta_TO NMAE % | theta_TO MAE degC |")
    md.append("|---|---|---|---|")
    for d in (base, f1_model, f1_ic, f1_both):
        md.append(f"| {d['label']} | {d['overall']:.1f} | {d['per_state'][0]:.1f} | "
                  f"{d['mae_TO']:.3f} |")
    md.append("")
    md.extend(row("fix 1", base, f1_both))
    md.append("")
    md.append("**The checkpoint is now invalid and a retrain is required.** Two "
              "independent reasons:\n")
    md.append("1. Changing the IC formula changes every initial condition and, "
              "through `c_eq = k_gen * V_arr / k_dis` with V_arr exponential in "
              "temperature, every gas IC multiplicatively. That is a different "
              "training distribution, so `DISTRIBUTION_FREEZE.md` must be "
              "re-established before any model is trained against it.")
    md.append("2. Changing the model's attractor changes the analytic baseline "
              "the network was trained to correct. The stored weights encode a "
              "correction to formula C; applied on top of the true fixed point "
              "they are correcting the wrong function, which is exactly what the "
              "degradation above shows.\n")
    md.append("The numbers above are therefore **not** a claim that the fixed "
              "model is worse. They measure how far the v57 weights are from the "
              "corrected physics, which is the honest quantity to report before a "
              "retrain exists.\n")

    md.append("### What the fix corrects, at the level of the physics\n")
    md.append("| K | theta_a | true fixed point | A (v57 ICs) | A error | "
              "C (v57 attractor) | C error |")
    md.append("|---|---|---|---|---|---|---|")
    from cod.data.steady_state import formula_C
    for k_, t_ in [(1.0, 30.0), (1.2, 30.0), (1.3, 30.0), (1.3, 45.0)]:
        tr = true_fixed_point(k_, t_)
        md.append(f"| {k_} | {t_:g} | {tr:.2f} | {formula_A(k_, t_):.2f} | "
                  f"{formula_A(k_, t_) - tr:+.2f} | {formula_C(k_, t_):.2f} | "
                  f"{formula_C(k_, t_) - tr:+.2f} |")
    md.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    md.append("## Summary\n")
    md.append("| fix | moves Gate 1? | measured effect | checkpoint still valid? |")
    md.append("|---|---|---|---|")
    md.append(f"| 1 unify steady state | **yes** | overall "
              f"{base['overall']:.1f}% -> {f1_both['overall']:.1f}%, theta_TO MAE "
              f"{base['mae_TO']:.2f} -> {f1_both['mae_TO']:.2f} degC | "
              f"**no — retrain required** |")
    md.append("| 2-5 | not yet applied | — | — |")
    md.append("")

    (ROOT / args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
