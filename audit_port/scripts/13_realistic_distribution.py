#!/usr/bin/env python3
"""Old sampler vs realistic sampler, side by side, and the Jensen gap on each.

Writes audit_port/REALISTIC_DISTRIBUTION.md. Nothing is frozen: this exists so the
distributions can be looked at before anything is committed to
DISTRIBUTION_FREEZE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, rk45_ground_truth
from cod.data.physics import (
    GAS_NAMES,
    IEC_ATTENTION,
    N_SENSORS,
    TW,
    hot_spot_ETC_np,
)
from cod.data.realistic import DEFAULTS, build_realistic_set, iec_exceedance
from cod.data.steady_state import formula_A
from cod.models.daily_mean import (
    activation_energies_kJ,
    jensen_gap_from_trajectory,
    jensen_gap_sinusoidal,
)

REPORT = ROOT / "audit_port" / "REALISTIC_DISTRIBUTION.md"
NAMES = list(GAS_NAMES) + ["DP"]
N = 100
N_T = 241
GUARD = 0.9999


def measure(x0s, sensors, label):
    """Realised swing, mean hot-spot and Jensen gap for a set of (IC, profile)."""
    t = np.linspace(0.0, TW, N_T)
    grid = np.linspace(0.0, TW, N_SENSORS)
    n = len(x0s)
    swing = np.zeros(n)
    hs_mean = np.zeros(n)
    hs_max = np.zeros(n)
    theta0 = np.zeros(n)
    gap = np.zeros((n, 6))

    for i in range(n):
        K = sensors[i][:N_SENSORS].astype(float)
        Ta = sensors[i][N_SENSORS:2 * N_SENSORS].astype(float)
        xt = rk45_ground_truth(x0s[i], K, Ta, t, T=TW, t_clip_frac=GUARD)
        K_at = np.interp(t, grid, K)
        hs = np.array([hot_spot_ETC_np(float(xt[j, 0]), float(K_at[j]))
                       for j in range(N_T)])
        swing[i] = 0.5 * (hs.max() - hs.min())
        hs_max[i] = hs.max()
        theta0[i] = float(x0s[i][0])
        gap[i], hs_mean[i] = jensen_gap_from_trajectory(hs, t)
        if (i + 1) % 25 == 0:
            print(f"  {label} {i + 1}/{n}")
    return dict(swing=swing, hs_mean=hs_mean, hs_max=hs_max, theta0=theta0,
                gap=gap, x0s=x0s)


def q(a):
    return (a.mean(), np.median(a), np.percentile(a, 10),
            np.percentile(a, 90), a.max())


def main() -> int:
    print("=== old sampler (seed-999 test set) ===")
    cases = build_test_set(n_test=N, seed=999, T=TW, steady_state=formula_A)
    old_x0 = np.array([c.x0 for c in cases])
    old_sens = np.array([np.concatenate([c.K_sensors, c.Ta_sensors])
                         for c in cases])
    old = measure(old_x0, old_sens, "old")

    print("=== realistic sampler ===")
    new_x0, new_sens = build_realistic_set(N, seed=999)
    new = measure(new_x0, new_sens, "new")

    ea = activation_energies_kJ()
    eo, en = iec_exceedance(old["x0s"]), iec_exceedance(new["x0s"])

    L: list[str] = []
    A = L.append
    A("# An operationally realistic distribution\n")
    A(f"Both samplers, N = {N}, measured the same way. **Nothing here is frozen** — "
      "this is for looking at before anything goes into "
      "`DISTRIBUTION_FREEZE.md`.\n")

    A("## 1. Why the old one is not operationally realistic\n")
    A("`profiles.sample_consistent_ic` draws `theta_TO(0)` from "
      "`steady_state(K, theta_a) + U(-30, 30)` where `K` and `theta_a` are drawn "
      "**independently of the profile that will drive the window**. So the unit "
      "starts at a temperature unrelated to its load and spends the window "
      "relaxing. Decomposed on the seed-999 test set "
      "(`audit_port/scripts/12_swing_decomposition.py`):\n")
    A("| hot-spot swing, degC | mean | median | p90 | max | above 15 |")
    A("|---|---|---|---|---|---|")
    A("| as drawn | 21.71 | 21.44 | 38.52 | 52.97 | 68% |")
    A("| with a profile-consistent IC | 10.85 | 5.11 | 31.11 | 38.69 | 38% |")
    A("| the theta_ss forcing alone | 11.23 | 6.14 | 30.74 | 35.86 | 41% |")
    A("")
    A("The IC offset from the profile-consistent value has a median of **28.3 degC** "
      "and a maximum of 74.3. Fixing the IC alone removes 76% of the median swing.\n")
    A("But fixing the IC alone is not enough, and the same table says why. Split by "
      "case type, a consistent IC gives the constant-K cases a swing of **exactly "
      "zero** — constant forcing, constant trajectory — while the time-varying "
      "cases stay at 20.3 degC because their own forcing is 21.3 degC. The result "
      "is bimodal at 0 and 20, not centred at 10-15. Both the IC **and** the "
      "profile amplitudes have to change.\n")

    A("## 2. What the realistic sampler does\n")
    A("**Initial condition.** `theta_TO(0)` is the periodic steady state of the "
      "profile itself — the state a unit running this daily pattern would actually "
      "be in — plus a recent-history offset `N(0, 3)` clipped to +-8 degC and "
      "sensor noise `N(0, 0.5)`. The offset matters: without it every constant-load "
      "window would start exactly at its steady state and never move, which is no "
      "more realistic than +-30 degC and would make the Jensen gap identically zero "
      "on those cases.\n")
    A("**Operating point.** Load is not drawn directly. A fleet is loaded so that "
      "temperature stays in band, so the intended mean hot-spot is drawn "
      "(`N(86, 11)` clipped to [62, 122] degC, against IEC 60076-7's 98 degC rated "
      "and 120 degC normal-cyclic ceiling) and `solve_K_for_hot_spot` inverts for "
      "the load factor that achieves it at that site's ambient. **This is what "
      "removes the IEC exceedance**: `c_eq = k_gen V_arr / k_dis` is exponential in "
      "temperature, so the 37% was the 150 degC initial conditions, not the gas "
      "model.\n")
    A("**Gases.** Long-run equilibrium at the unit's own mean hot-spot times a "
      "service factor `U(0.45, 1.35)`, with 8% of units carrying an incipient fault "
      "(H2, C2H2, C2H4 multiplied by 2-7). A fleet where nothing is ever elevated "
      "would make the DGA benchmark trivial.\n")

    A("## 3. Realised hot-spot swing, side by side\n")
    A("Half peak-to-peak of the true hot-spot trajectory, the same quantity "
      "C-10's table is indexed by.\n")
    A("| | mean | median | p10 | p90 | max | in 8-18 degC | above 25 degC |")
    A("|---|---|---|---|---|---|---|---|")
    for lbl, d in [("old sampler", old), ("realistic", new)]:
        s = d["swing"]
        m, md, p10, p90, mx = q(s)
        A(f"| {lbl} | {m:.2f} | **{md:.2f}** | {p10:.2f} | {p90:.2f} | {mx:.2f} | "
          f"{((s >= 8) & (s <= 18)).mean():.0%} | {(s > 25).mean():.0%} |")
    A("")
    A(f"Median moves from {np.median(old['swing']):.1f} to "
      f"{np.median(new['swing']):.1f} degC, into the 10-15 target band, and the "
      f"share above 25 degC falls from {(old['swing'] > 25).mean():.0%} to "
      f"{(new['swing'] > 25).mean():.0%}.\n")
    A("Underlying temperatures:\n")
    A("| | mean hot-spot | max hot-spot reached | theta_TO(0) mean | theta_TO(0) max |")
    A("|---|---|---|---|---|")
    for lbl, d in [("old sampler", old), ("realistic", new)]:
        A(f"| {lbl} | {d['hs_mean'].mean():.1f} | {d['hs_max'].max():.1f} | "
          f"{d['theta0'].mean():.1f} | {d['theta0'].max():.1f} |")
    A("")
    A(f"The old set reaches a hot-spot of {old['hs_max'].max():.1f} degC, past the "
      f"187.2 degC where the model's `V_arr` clamp parts company with the reference "
      f"(DECISIONS N-1). The realistic set peaks at {new['hs_max'].max():.1f} degC, "
      f"so that failure mode does not arise "
      f"({(new['hs_max'] > 187.2).mean():.0%} of cases against "
      f"{(old['hs_max'] > 187.2).mean():.0%}).\n")

    A("## 4. Gas initial conditions against IEC 60599 attention levels\n")
    A(f"Both columns are the N = {N} evaluation ICs. Audit M-9's 37.0% is the "
      "8000-IC *training* set; the same old sampler gives a different figure on "
      "this smaller draw, so both are reported and a large-sample estimate "
      "follows.\n")
    A("| gas | attention ppm | old: above | realistic: above | old median ppm | "
      "realistic median ppm |")
    A("|---|---|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        A(f"| `{g}` | {IEC_ATTENTION[i]} | {eo[g]:.0%} | {en[g]:.0%} | "
          f"{np.median(old['x0s'][:, i + 1]):.3g} | "
          f"{np.median(new['x0s'][:, i + 1]):.3g} |")
    A(f"| **any gas** | | **{eo['any_gas']:.0%}** | **{en['any_gas']:.0%}** | | |")
    A("")
    big_x0, _ = build_realistic_set(2000, seed=7)
    eb = iec_exceedance(big_x0)
    from cod.data.generate import load_training_set
    tr = load_training_set(ROOT / "reference" / "artifacts"
                           / "transformer_training_v57.npz")
    att = np.asarray(IEC_ATTENTION, dtype=float)
    old_train = float((tr.x0s[:, 1:] > att).any(axis=1).mean())
    A("On a stable sample (realistic sampler, n = 2000):\n")
    A("| gas | above attention |")
    A("|---|---|")
    for g in GAS_NAMES:
        A(f"| `{g}` | {eb[g]:.2%} |")
    A(f"| **any gas** | **{eb['any_gas']:.2%}** |")
    A("")
    A(f"So exceedance falls from **{old_train:.1%}** on the old 8000-IC training set "
      f"(audit M-9's figure) to **{eb['any_gas']:.1%}**, a factor of "
      f"{old_train / eb['any_gas']:.1f}.\n")
    A("**The residue is almost entirely H2, and that is a kinetics problem rather "
      "than a sampler problem.** `c_eq = k_gen/k_dis * V_arr` for H2 is 76 ppm at a "
      "110 degC hot-spot against an attention level of 100 ppm, so a transformer in "
      "long-run equilibrium at the IEEE reference temperature sits at 76% of the H2 "
      "attention level and anything slightly hotter exceeds it. Field practice puts "
      "a healthy unit at 5-50 ppm. The sampler could be tuned to hide this by "
      "lowering the operating temperature, and deliberately is not: it is further "
      "evidence for O-3, that `k_gen` and `k_dis` have no stated source. The other "
      f"four gases sit at "
      f"{max(eb[g] for g in GAS_NAMES if g != 'c_H2'):.1%} or below.\n")
    A(f"The remainder is the deliberate {DEFAULTS.fault_prob:.0%} incipient-fault "
      "subpopulation, which is a feature: a benchmark where no unit is ever flagged "
      "does not test anything a practitioner cares about.\n")
    A("Worst-case magnitudes tell the same story:\n")
    A("| gas | old max ppm | as x attention | realistic max ppm | as x attention |")
    A("|---|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        om, nm = old['x0s'][:, i + 1].max(), new['x0s'][:, i + 1].max()
        A(f"| `{g}` | {om:.4g} | {om / IEC_ATTENTION[i]:.0f}x | {nm:.4g} | "
          f"{nm / IEC_ATTENTION[i]:.1f}x |")
    A("")

    A("## 5. The Jensen gap on each distribution\n")
    A("Computed from the true hot-spot trajectory, so no model error is in it.\n")
    ana15 = dict(zip(NAMES, jensen_gap_sinusoidal(15.0, 100.0)))
    A("**Read the medians.** The gap is exponential in swing, so for the high-Ea "
      "states a handful of large-swing cases dominate any mean: C2H2's realistic "
      f"mean is {new['gap'][:, 1].mean():.1f} against a median of "
      f"{np.median(new['gap'][:, 1]):.2f}. The mean describes the tail, not a "
      "typical unit.\n")
    A("| state | Ea kJ/mol | old median | realistic median | "
      "C-10 analytical at +-15 degC |")
    A("|---|---|---|---|---|")
    for i, nm in enumerate(NAMES):
        A(f"| `{nm}` | {ea[nm]:.1f} | {np.median(old['gap'][:, i]):.3f} | "
          f"**{np.median(new['gap'][:, i]):.3f}** | {ana15[nm]:.3f} |")
    A("")
    A("Means, and note how far the tail moves them:\n")
    A("| state | old mean | realistic mean |")
    A("|---|---|---|")
    for i, nm in enumerate(NAMES):
        A(f"| `{nm}` | {old['gap'][:, i].mean():.3f} | "
          f"{new['gap'][:, i].mean():.3f} |")
    A("")
    A(f"On medians, DP moves from {np.median(old['gap'][:, 5]):.2f} to "
      f"{np.median(new['gap'][:, 5]):.2f} and C2H2 from "
      f"{np.median(old['gap'][:, 1]):.2f} to {np.median(new['gap'][:, 1]):.2f}, "
      "against C-10's analytical 1.70 and 2.59 at the +-15 degC reference. The "
      "realistic medians now sit just **below** that reference rather than far "
      f"above it, which is what a median swing of {np.median(new['swing']):.1f} degC "
      "should give: C-10's reference is +-15 and this distribution is centred a "
      "little under it.\n")
    A("That is the honest headline. Quoting the old set's medians instead would "
      f"have overstated the gap by "
      f"{np.median(old['gap'][:, 5]) / np.median(new['gap'][:, 5]) - 1:.0%} on DP "
      f"and {np.median(old['gap'][:, 1]) / np.median(new['gap'][:, 1]) - 1:.0%} on "
      "C2H2.\n")
    A("Stratified by swing, to show the mechanism is unchanged and only the "
      "distribution moved:\n")
    A("| swing band degC | old n | old DP gap | realistic n | realistic DP gap | "
      "analytical DP |")
    A("|---|---|---|---|---|---|")
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 25), (25, 200)]:
        mo = (old["swing"] >= lo) & (old["swing"] < hi)
        mn = (new["swing"] >= lo) & (new["swing"] < hi)
        mid = 0.5 * (lo + min(hi, 40))
        ana = dict(zip(NAMES, jensen_gap_sinusoidal(mid, 95.0)))
        A(f"| {lo}-{hi} | {int(mo.sum())} | "
          f"{old['gap'][mo, 5].mean() if mo.any() else float('nan'):.3f} | "
          f"{int(mn.sum())} | "
          f"{new['gap'][mn, 5].mean() if mn.any() else float('nan'):.3f} | "
          f"{ana['DP']:.3f} |")
    A("")
    A("Same gap at the same swing in both columns. The old distribution was not "
      "producing a *different* physics, it was sampling a different place on the "
      "same curve.\n")

    A("## 6. What is still open\n")
    A("1. **The load amplitude is calibrated to the target, not measured from a "
      "fleet.** Reaching a 10-15 degC hot-spot swing needs a daily load swing of "
      "+-12-28%, which is at the upper end of what a real feeder does. That is "
      "stated in `RealisticParams` rather than hidden. The honest reading: the "
      "Jensen gap matters for cycled units, and a genuinely base-loaded transformer "
      "has almost no gap for any method to exploit.")
    A("2. **The gas kinetics still have no provenance (O-3).** The IEC exceedance "
      "fell because temperatures became realistic, not because `k_gen` and `k_dis` "
      "were justified. `c_eq` at a 110 degC hot-spot is 76 ppm of H2 against a "
      "100 ppm attention level, and nothing in the repository says why.")
    A("3. **The 8% fault subpopulation is invented.** It is a modelling choice to "
      "keep the benchmark non-trivial, not a measured fleet statistic.")
    A("4. **Not frozen.** No hash recorded, no test tiers defined. T2 and T3 still "
      "need designing on top of this, and the whole point of the freeze protocol is "
      "that it happens before the first model is trained against it.\n")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez(ROOT / "audit_port" / "realistic_comparison.npz",
             old_swing=old["swing"], new_swing=new["swing"],
             old_gap=old["gap"], new_gap=new["gap"],
             old_x0=old["x0s"], new_x0=new["x0s"],
             old_hs_max=old["hs_max"], new_hs_max=new["hs_max"])
    print(f"\nWrote {REPORT}")
    print(f"  swing median  old {np.median(old['swing']):.2f} -> "
          f"new {np.median(new['swing']):.2f} degC")
    print(f"  IEC any-gas   old {eo['any_gas']:.0%} -> new {en['any_gas']:.0%}")
    print(f"  DP gap mean   old {old['gap'][:, 5].mean():.2f} -> "
          f"new {new['gap'][:, 5].mean():.2f}  (C-10 at +-15: 1.70)")
    print(f"  C2H2 gap mean old {old['gap'][:, 1].mean():.2f} -> "
          f"new {new['gap'][:, 1].mean():.2f}  (C-10 at +-15: 2.59)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
