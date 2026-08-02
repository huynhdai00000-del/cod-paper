#!/usr/bin/env python3
"""Which distribution does `build_test_set` actually draw from, and is the tier
label `T1_in_distribution` true of it?

WHY THIS EXISTS. `scripts/run.py` trains on whichever sampler the hashed
`distribution.sampler` block names — `realistic` since fix 9 — and then evaluates
on `build_test_set(seed=999)`. PORT_LOG J-81 records the reason the test set sits
outside the sampler branch: `ic_formula` was hoisted so that "the test set is the
seed-999 benchmark regardless of which sampler drew the training data". That is a
statement about *code structure*. This script asks whether it is also true as a
statement about *distributions*, which is what the tier label claims.

The tier label is not decoration. DECISIONS C-9 and README both forbid merging
tiers, and rule 4 forbids looking at T2/T3 before the model is finished by a
pre-declared criterion. If the seed-999 benchmark is in fact out of the training
distribution, then every number O-5 produced was read off a tier the protocol
says must not be read yet, under a name that says the opposite.

WHAT IS MEASURED. Nine axes, each compared against the frozen realistic sampler
(`configs/example_cod_seed1.yaml`, distribution hash fc4cb76c3b32ec17), plus the
per-state NMAE denominators that identify which of the two possible test sets a
given run.json came from.

Run:  python audit_port/scripts/21_test_set_provenance.py
Exit: 0 if the seed-999 benchmark lies inside the training distribution on every
      axis, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, rk45_ground_truth  # noqa: E402
from cod.data.physics import (  # noqa: E402
    IEC_ATTENTION, N_SENSORS, TW, hot_spot_ETC_np,
)
from cod.data.realistic import (  # noqa: E402
    RealisticParams, build_realistic_set,
)
from cod.data.steady_state import formula_A, true_fixed_point_np  # noqa: E402
from cod.eval.metrics import TRANSFORMER_STATES, evaluate_state  # noqa: E402
from cod.training.losses import STATE_CLAMP_HI_NP  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "TEST_SET_PROVENANCE.md"

N_TEST = 100
N_TRAIN = 300      # enough for the marginals; each case costs one RK45 solve
NQ = 100


# ═══════════════════════════════════════════════════════════════════════════
# Per-case measurements, identical code for both populations
# ═══════════════════════════════════════════════════════════════════════════
def measure(x0: np.ndarray, K_s: np.ndarray, Ta_s: np.ndarray) -> dict:
    """Everything this script compares, for one (IC, profile) pair.

    `ic_offset` is the distance from theta_TO(0) to the equilibrium of the
    forcing *at the start of the window*. It is deliberately not the periodic
    steady state: the realistic sampler's IC is the periodic state of a 24 h day
    read at the window's own time of day, so a window-as-cycle reference would be
    wrong for it, while the instantaneous equilibrium is defined identically for
    both populations and needs no assumption about the load period.
    """
    tau = np.linspace(0.0, TW, N_SENSORS)
    tq = np.linspace(0.0, TW, NQ)
    gt = rk45_ground_truth(np.asarray(x0, float), np.asarray(K_s, float),
                           np.asarray(Ta_s, float), tq, T=TW)
    K_q = np.interp(tq, tau, K_s)
    hs = np.array([hot_spot_ETC_np(float(gt[i, 0]), float(K_q[i]))
                   for i in range(NQ)])
    return {
        "theta0": float(x0[0]),
        "ic_offset": abs(float(x0[0])
                         - float(true_fixed_point_np(float(K_s[0]),
                                                     float(Ta_s[0])))),
        "K_min": float(np.min(K_s)),
        "K_max": float(np.max(K_s)),
        "K_swing": 0.5 * float(np.max(K_s) - np.min(K_s)),
        "Ta_swing": 0.5 * float(np.max(Ta_s) - np.min(Ta_s)),
        "hs_mean": float(hs.mean()),
        "hs_max": float(hs.max()),
        "hs_swing": 0.5 * float(hs.max() - hs.min()),
        "iec_over": bool(np.any(np.asarray(x0[1:], float)
                                > np.asarray(IEC_ATTENTION, float))),
        "clamp_over": bool(np.any(np.asarray(x0, float) > STATE_CLAMP_HI_NP)),
        "gt": gt,
    }


def pop(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def q(a: np.ndarray) -> tuple[float, float, float, float, float]:
    return (float(np.min(a)), float(np.percentile(a, 25)), float(np.median(a)),
            float(np.percentile(a, 75)), float(np.max(a)))


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dist = cfg["distribution"]
    params = RealisticParams.from_config(dist["sampler"]["params"])
    print(f"[cfg] {CONFIG.name}  sampler={dist['sampler']['kind']}  "
          f"cycle_period={params.cycle_period:g} min")

    # ── The two candidate test sets ────────────────────────────────────────
    # run.py picks the IC formula from `steady_state_formula`, which the frozen
    # config sets to `true_fixed_point`. PHASE1_VERIFICATION used formula_A.
    # Same 100 draws, different IC formula, therefore different benchmark.
    print(f"[data] building the seed-999 benchmark, both IC formulas "
          f"({N_TEST} cases each)")
    cases_tfp = build_test_set(n_test=N_TEST, seed=999, T=TW,
                               steady_state=true_fixed_point_np)
    cases_a = build_test_set(n_test=N_TEST, seed=999, T=TW,
                             steady_state=formula_A)

    m_tfp = [measure(c.x0, c.K_sensors, c.Ta_sensors) for c in cases_tfp]
    m_a = [measure(c.x0, c.K_sensors, c.Ta_sensors) for c in cases_a]

    # ── The frozen training distribution ───────────────────────────────────
    print(f"[data] drawing {N_TRAIN} cases from the frozen realistic sampler")
    x0s, sens = build_realistic_set(N_TRAIN, int(dist.get("seed", 42)), params)
    m_tr = [measure(x0s[i], sens[i, :N_SENSORS], sens[i, N_SENSORS:])
            for i in range(N_TRAIN)]

    # ── 1. Identify the run's test set by its NMAE denominators ────────────
    # `denominator_median` is a property of the ground truth alone, so it
    # fingerprints the test set without needing the model.
    denom = {}
    for tag, ms in (("true_fixed_point", m_tfp), ("formula_A", m_a)):
        gt = np.stack([r["gt"] for r in ms])
        zero = np.zeros_like(gt[:, :, 0])
        denom[tag] = {
            spec.name: evaluate_state(zero, gt[:, :, i], spec).denominator_median
            for i, spec in enumerate(TRANSFORMER_STATES[:6])
        }

    print("\n=== 1. NMAE denominators, which identify the test set ===")
    print(f"{'state':>10} {'true_fixed_point':>18} {'formula_A':>14}")
    for name in denom["true_fixed_point"]:
        print(f"{name:>10} {denom['true_fixed_point'][name]:18.6g} "
              f"{denom['formula_A'][name]:14.6g}")

    # ── 2. Axis-by-axis support comparison ─────────────────────────────────
    tr_hs_lo, tr_hs_hi = params.hot_spot_bounds
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"  [{'ok ' if ok else 'OUT'}] {name}: {detail}")

    print("\n=== 2. Is the seed-999 benchmark inside the training support? ===")
    t0_tr, t0_te = pop(m_tr, "theta0"), pop(m_tfp, "theta0")
    check("theta_TO(0) range",
          t0_te.min() >= t0_tr.min() and t0_te.max() <= t0_tr.max(),
          f"test [{t0_te.min():.1f}, {t0_te.max():.1f}] vs train "
          f"[{t0_tr.min():.1f}, {t0_tr.max():.1f}] degC; "
          f"{100.0 * ((t0_te < t0_tr.min()) | (t0_te > t0_tr.max())).mean():.0f}%"
          " of test cases outside")

    off_tr, off_te = pop(m_tr, "ic_offset"), pop(m_tfp, "ic_offset")
    check("IC consistency with the profile",
          np.median(off_te) <= 3.0 * np.median(off_tr),
          f"|theta_TO(0) - eq(K(0),Ta(0))| median test {np.median(off_te):.2f} "
          f"vs train {np.median(off_tr):.2f} degC "
          f"(ratio {np.median(off_te) / max(np.median(off_tr), 1e-9):.1f}x)")

    hm_te = pop(m_tfp, "hs_mean")
    outside = ((hm_te < tr_hs_lo) | (hm_te > tr_hs_hi)).mean()
    check("mean hot-spot inside the sampler's operating band",
          outside == 0.0,
          f"training clips the drawn hot-spot to [{tr_hs_lo:g}, {tr_hs_hi:g}] "
          f"degC; {100.0 * outside:.0f}% of test cases have a window-mean "
          f"hot-spot outside it (test range "
          f"[{hm_te.min():.1f}, {hm_te.max():.1f}])")

    sw_tr, sw_te = pop(m_tr, "hs_swing"), pop(m_tfp, "hs_swing")
    lo, hi = np.percentile(sw_tr, [5, 95])
    check("realised hot-spot swing",
          lo <= np.median(sw_te) <= hi,
          f"median test {np.median(sw_te):.2f} degC vs train "
          f"{np.median(sw_tr):.2f} (train p5-p95 {lo:.2f}-{hi:.2f}); "
          f"{100.0 * (sw_te > hi).mean():.0f}% of test cases above train p95")

    k_tr_lo, k_tr_hi = pop(m_tr, "K_min").min(), pop(m_tr, "K_max").max()
    k_te_lo, k_te_hi = pop(m_tfp, "K_min").min(), pop(m_tfp, "K_max").max()
    check("K range",
          k_te_lo >= k_tr_lo and k_te_hi <= k_tr_hi,
          f"test [{k_te_lo:.3f}, {k_te_hi:.3f}] vs train "
          f"[{k_tr_lo:.3f}, {k_tr_hi:.3f}]")

    # Load period. Structural, and checkable: the TV branch is
    # K_k + 0.2 sin(2 pi t / TW), one whole cycle inside the 720 min window.
    tv = [c for c in cases_tfp if c.kind == "TV"]
    full_cycle = np.mean([abs(float(c.K_sensors[0] - c.K_sensors[-1])) < 1e-5
                          for c in tv])
    check("load cycle period",
          params.cycle_period == TW,
          f"training day is {params.cycle_period:g} min (fix 7, N-6); the test "
          f"TV branch completes a whole cycle inside the {TW:g} min window, i.e. "
          f"a {TW:g} min load period ({100.0 * full_cycle:.0f}% of TV cases "
          "return to K(0) at the window end)")

    ta_tr, ta_te = pop(m_tr, "Ta_swing"), pop(m_tfp, "Ta_swing")
    check("ambient swing",
          np.median(ta_te) >= np.percentile(ta_tr, 5),
          f"median test {np.median(ta_te):.2f} degC vs train "
          f"{np.median(ta_tr):.2f}; test CK half is flat by construction "
          f"({100.0 * (ta_te < 1e-6).mean():.0f}% of test cases have zero "
          "ambient variation)")

    iec_tr = np.mean([r["iec_over"] for r in m_tr])
    iec_te = np.mean([r["iec_over"] for r in m_tfp])
    check("gas ICs above IEC 60599 attention",
          iec_te <= iec_tr + 0.02,
          f"test {100.0 * iec_te:.1f}% vs train {100.0 * iec_tr:.1f}%")

    cl_tr = np.mean([r["clamp_over"] for r in m_tr])
    cl_te = np.mean([r["clamp_over"] for r in m_tfp])
    check("ICs above the physics-loss state clamp",
          cl_te <= cl_tr + 0.02,
          f"test {100.0 * cl_te:.1f}% vs train {100.0 * cl_tr:.1f}%")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} axes inside the training "
          f"distribution.")

    # ── Report ─────────────────────────────────────────────────────────────
    md: list[str] = []
    A = md.append
    A("# What `build_test_set` draws, and whether `T1_in_distribution` is true\n")
    A("Generated by `audit_port/scripts/21_test_set_provenance.py`. Training "
      f"population: {N_TRAIN} cases from the frozen realistic sampler "
      f"(`{CONFIG.name}`, distribution hash `fc4cb76c3b32ec17`). Test "
      f"population: the {N_TEST}-case seed-999 benchmark as `scripts/run.py` "
      "builds it.\n")

    A("## 1. What it draws\n")
    A("`build_test_set` does not consult `distribution.sampler` at all. It calls "
      "`profiles.sample_consistent_ic` — the **v57-era** IC sampler, the one "
      "audit M-9 is against — and then builds its own two waveform families in "
      "line, hardcoded:\n")
    A("| half | IC | load | ambient |")
    A("|---|---|---|---|")
    A("| CK, cases 0-49 | `sample_consistent_ic` | `K ~ U(0.4, 1.4)`, constant | "
      "`Ta ~ U(15, 45)`, constant |")
    A("| TV, cases 50-99 | `sample_consistent_ic` | "
      "`K + 0.2 sin(2 pi t / TW)` | `Ta + 5 sin(2 pi t / TW + pi/3)` |")
    A("")
    A("The only thing the config reaches is the IC formula: `run.py` reads "
      "`distribution.steady_state_formula` and passes `true_fixed_point_np` (the "
      "frozen config) or `formula_A` (v57). Everything else — the load families, "
      "the amplitudes, the period, the IC construction — is fixed in the source "
      "of `build_test_set` and `sample_consistent_ic`, neither of which is inside "
      "the hashed `distribution` block.\n")
    A("So the sampler that fix 7 and fix 9 built, froze and hashed drew the "
      "**training** data only. The evaluation ran on the distribution those "
      "fixes were made to replace.\n")

    A("## 2. Which of the two benchmarks a run.json came from\n")
    A("Both IC formulas replay the same 100 draws, so they differ only through "
      "`theta_TO(0)` and the gas ICs derived from it. `denominator_median` is a "
      "property of the ground truth alone, so it fingerprints the test set "
      "without needing the model:\n")
    A("| state | `steady_state_formula: true_fixed_point` | `formula_A` (v57) |")
    A("|---|---|---|")
    for name in denom["true_fixed_point"]:
        A(f"| `{name}` | {denom['true_fixed_point'][name]:.6g} | "
          f"{denom['formula_A'][name]:.6g} |")
    A("")
    A("The frozen config sets `steady_state_formula: true_fixed_point`, so a run "
      "off `example_cod_seed1.yaml` reports the left column. "
      "`PHASE1_VERIFICATION.md` reproduces v57 and reports the right one. The "
      "difference is Phase 2 fix 1, not a different draw and not a different "
      "sampler: formula A sits up to 23.8 degC below the true fixed point, which "
      "moves every `theta_TO(0)` and, because `c_eq` is exponential in "
      "temperature, every gas IC with it.\n")

    A("## 3. Is the tier label true?\n")
    A("Each axis compares the seed-999 benchmark against the frozen training "
      "sampler. `OUT` means the benchmark is outside what the model was trained "
      "on.\n")
    A("| axis | verdict | measurement |")
    A("|---|---|---|")
    for name, ok, detail in checks:
        A(f"| {name} | {'inside' if ok else '**OUT**'} | {detail} |")
    A("")
    A(f"**{n_fail} of {len(checks)} axes fall outside the training "
      "distribution.**\n" if n_fail else
      "**Every axis falls inside the training distribution.**\n")

    A("## 4. Verdict\n")
    if n_fail:
        A("`T1_in_distribution` is **not** a true description of this "
          "evaluation. The model was trained on the fix-7 realistic sampler and "
          "scored on the v57-era benchmark, which differs from it on the axes "
          "above — including two hard support boundaries the training sampler "
          "enforces by construction (the hot-spot operating band, and the "
          "profile-consistent IC). That is a train/test distribution mismatch of "
          "the same kind the freeze work exists to prevent, and it is worse than "
          "the original in one respect: the original was mislabelled in the "
          "manuscript, this one is mislabelled in the config, where the protocol "
          "reads it.\n")
        A("Consequence for O-5's numbers: they are not wrong, they are "
          "**unlabelled**. theta_TO MAE 1.072 degC is an out-of-distribution "
          "figure being compared against v57's 0.399 degC, which was an "
          "in-distribution figure for v57's own training set. The comparison "
          "measures distribution shift and physics fixes at the same time and "
          "cannot separate them.\n")
    else:
        A("`T1_in_distribution` is a true description of this evaluation.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
