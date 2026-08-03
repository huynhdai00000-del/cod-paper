#!/usr/bin/env python3
"""Settle N-11 against the real pre-fix-7 sampler, then measure the Jensen gap at
ETT-calibrated load swing.

PART 1 — THE ARM THAT WAS MISSING. `23_swing_multiseed.py` compared the fix-7
sampler against *the current sampler with* `cycle_period=720`, which confounds the
period with event duration: the event families scale their widths with `P`
(`0.04-0.10 P` for an overload spike, `0.09-0.15 P` for an evening peak), so at
P=1440 they reproduce the pre-fix-7 absolute widths of 58-144 and 130-216 min
while at P=720 they are half of them. Shorter events heat the oil less, so that
arm is not the old sampler and its 0.965 ratio was not a verdict.

This script builds the arm properly: `cod/data/realistic.py` as it stood at
`727d77c^` — the commit before fix 7 — loaded straight out of git so the
comparison cannot drift from what was actually there. Everything downstream of the
sampler (physics, steady state, `DailyMeanArrhenius`) is the current code in both
arms, because `727d77c^` already carries fixes 1-6 and 8; the sampler is the only
thing that differs.

What hangs on it, from PERIOD_FIX.md §2: the claim that the sampler's apparent
over-assumption of load swing was a **period** error rather than an **amplitude**
error, and the K_amp range of 10.2-23.8% derived by scaling 12-28% by
11.20/13.18 = 0.850. If the uplift is near 1 the scale factor is near 1, K_amp
stays near 12-28%, and that sits above both feeders ETT measures — which makes it
an amplitude assumption after all and reopens O-10's scope decision.

PART 2 — TWO HONEST OPERATING POINTS. C-13 already commits to presenting the
Jensen gap as a curve against amplitude with the real amplitude distribution laid
over it, rather than as one number. So rather than argue about a single contested
figure, this measures the sampler at the two amplitudes ETT actually gives
(`ETT_LOAD_CALIBRATION.md`): ETTh2's median of 8.7% of rated, and ETTh1's
non-back-feeding median of 17.8%. Reported: realised hot-spot swing, and the
resulting gap for DP and C2H2 — the aging indicator and the arc-fault indicator,
the two states C-10 puts at the ends of the activation-energy range.

Run:  python audit_port/scripts/26_prefix7_arm_and_ett_gap.py
Exit: 0 if N-7's 1.177 uplift is inside the per-seed range measured here, 1 if not.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import rk45_ground_truth  # noqa: E402
from cod.data.physics import N_SENSORS, TW, hot_spot_ETC_np  # noqa: E402
from cod.data.realistic import RealisticParams, build_realistic_set  # noqa: E402
from cod.models.daily_mean import jensen_gap_from_trajectory  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "PREFIX7_ARM_AND_ETT_GAP.md"
PREFIX7_REV = "727d77c^"        # the commit before fix 7

N = 500
SEEDS = (42, 999, 7, 123, 2024, 31337)
N_GAP = 400                     # per ETT operating point

# The recorded figures under test.
PERIOD_FIX_OLD = 11.20          # PERIOD_FIX.md §2, old sampler, N=100
PERIOD_FIX_NEW = 13.18          # PERIOD_FIX.md §2, fix 7, N=200 seed 999
N7_UPLIFT = 1.177               # DECISIONS N-7, = 13.18 / 11.20

# ETT_LOAD_CALIBRATION.md medians, as a fraction of rated.
# The back-feeding point matters most and was the one missing: C-13 identifies
# PV back-feeding days as *the* operating mode the paper opens on, and at 29.7%
# it sits above the frozen sampler's own 12-28% range, so no measurement inside
# the sampler's band speaks for it.
ETT_POINTS = [
    ("ETTh2, all days", 0.087),
    ("ETTh1, non-back-feeding days", 0.178),
    ("ETTh1, back-feeding days", 0.297),
]

# C-10's analytic table, for checking the measured gap against the curve. The
# zero node is not in C-10 and is not a measurement: a constant trajectory has a
# gap of exactly 1 by convexity, which `jensen_gap_from_trajectory` documents.
# Without it `np.interp` clamps everything below 5 degC to 1.07, and ETTh2's
# operating point is expected to land in that region — so the cross-check would
# read a floor of its own making as agreement.
C10_AMPLITUDE = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
C10_DP = np.array([1.00, 1.07, 1.29, 1.70, 2.37])
C10_C2H2 = np.array([1.00, 1.14, 1.62, 2.59, 4.42])

GAP_DP, GAP_C2H2 = 5, 1         # index into jensen_gap_from_trajectory


def load_prefix7_module():
    """`cod/data/realistic.py` as of `727d77c^`, imported from git.

    Read out of the repository rather than kept as a copy in the tree: a
    hand-maintained snapshot of old code is exactly the thing that silently stops
    matching what it claims to be.
    """
    # encoding is explicit: the file carries box-drawing characters and Windows
    # would otherwise decode git's output as cp1252 and fail.
    src = subprocess.run(
        ["git", "show", f"{PREFIX7_REV}:cod/data/realistic.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "realistic_prefix7.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("realistic_prefix7", tmp)
    mod = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so
    # the module has to be registered before its body executes or the decorator
    # fails on a module that is not there yet.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prefix7_params(mod, p_now: RealisticParams):
    """The old `RealisticParams` carrying the frozen config's shared values.

    The old dataclass has no `cycle_period` — that field is what fix 7 added — so
    the shared fields are copied across by name and nothing is defaulted silently.
    """
    old_names = {f.name for f in fields(mod.RealisticParams)}
    now_names = {f.name for f in fields(RealisticParams)}
    shared = old_names & now_names
    missing = old_names - now_names
    if missing:
        raise SystemExit(f"pre-fix-7 params have fields the current one lacks: "
                         f"{sorted(missing)}")
    kw = {n: getattr(p_now, n) for n in shared}
    return mod.RealisticParams(**kw), sorted(now_names - old_names)


def swings_and_gaps(x0s, sens, want_gap=False):
    """Half peak-to-peak hot-spot swing per case, and optionally the Jensen gap."""
    n = len(x0s)
    tq = np.linspace(0.0, TW, N_SENSORS)
    sw = np.empty(n)
    gaps = np.empty((n, 6)) if want_gap else None
    for i in range(n):
        K_s = sens[i, :N_SENSORS].astype(float)
        Ta_s = sens[i, N_SENSORS:].astype(float)
        gt = rk45_ground_truth(x0s[i].astype(float), K_s, Ta_s, tq, T=TW)
        hs = np.array([hot_spot_ETC_np(float(gt[j, 0]), float(K_s[j]))
                       for j in range(N_SENSORS)])
        sw[i] = 0.5 * (hs.max() - hs.min())
        if want_gap:
            gaps[i] = jensen_gap_from_trajectory(hs, tq)[0]
    return sw, gaps


def arm(label, builder, p, seeds=SEEDS, n=N):
    print(f"\n--- {label} ---")
    print(f"{'seed':>7} {'median':>9} {'p25':>8} {'p75':>8}")
    meds, allsw = [], []
    for s in seeds:
        x0s, sens = builder(n, s, p)
        sw, _ = swings_and_gaps(x0s, sens)
        meds.append(float(np.median(sw)))
        allsw.append(sw)
        print(f"{s:7d} {np.median(sw):9.3f} {np.percentile(sw, 25):8.3f} "
              f"{np.percentile(sw, 75):8.3f}")
    meds = np.array(meds)
    pooled = np.concatenate(allsw)
    print(f"between-seed: min {meds.min():.3f} max {meds.max():.3f} "
          f"sd {meds.std(ddof=1):.3f} | pooled median {np.median(pooled):.3f} "
          f"(N = {len(pooled)})")
    return meds, pooled


def gap_only(p_now) -> int:
    """Part 2 alone: the Jensen gap at each ETT operating point.

    Part 1's two sampler arms cost ~45 min and are already settled (N-11), so
    re-running them to add an operating point would be pure waste.
    """
    print("=== Jensen gap at ETT-calibrated load swing (part 2 only) ===")
    for lbl, amp in [("frozen sampler (12-28%)", None)] + list(ETT_POINTS):
        p = p_now if amp is None else replace(p_now, K_amp=(amp, amp))
        x0s, sens = build_realistic_set(N_GAP, 999, p)
        sw, gaps = swings_and_gaps(x0s, sens, want_gap=True)
        med = float(np.median(sw))
        print(f"  {lbl:34s} K_amp {str(p.K_amp):16s} "
              f"swing p25/med/p75 {np.percentile(sw, 25):5.2f}/{med:5.2f}/"
              f"{np.percentile(sw, 75):5.2f}  "
              f"DP {np.median(gaps[:, GAP_DP]):.3f}  "
              f"C2H2 {np.median(gaps[:, GAP_C2H2]):.3f}  "
              f"| C-10 curve at that swing: DP "
              f"{np.interp(med, C10_AMPLITUDE, C10_DP):.3f} "
              f"C2H2 {np.interp(med, C10_AMPLITUDE, C10_C2H2):.3f}")
    return 0


def main() -> int:
    p_now = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])
    if "--gap-only" in sys.argv:
        return gap_only(p_now)
    mod = load_prefix7_module()
    p_old, added = prefix7_params(mod, p_now)
    print(f"[git] pre-fix-7 sampler from {PREFIX7_REV}")
    print(f"[git] fields fix 7 added: {added}")
    print(f"K_amp = {p_now.K_amp} in both arms; the sampler is the only difference.")

    m_old, pool_old = arm("pre-fix-7 sampler (727d77c^), 12 h window-local",
                          mod.build_realistic_set, p_old)
    m_new, pool_new = arm("fix-7 sampler, 24 h day windowed at random phase",
                          build_realistic_set, p_now)

    med_old, med_new = float(np.median(pool_old)), float(np.median(pool_new))
    ratios = m_new / m_old
    print("\n=== uplift from fix 7, per seed ===")
    print("  ".join(f"{s}:{r:.3f}" for s, r in zip(SEEDS, ratios)))
    print(f"pooled {med_old:.3f} -> {med_new:.3f} degC, ratio {med_new / med_old:.4f}")
    print(f"per-seed: median {np.median(ratios):.4f}  min {ratios.min():.4f}  "
          f"max {ratios.max():.4f}")

    scale = med_old / med_new
    lo, hi = p_now.K_amp
    print(f"\nK_amp rescale to restore the old median: {scale:.4f} "
          f"-> {100 * lo * scale:.1f}-{100 * hi * scale:.1f}%")

    ok_uplift = ratios.min() <= N7_UPLIFT <= ratios.max()
    ok_old = m_old.min() <= PERIOD_FIX_OLD <= m_old.max()
    ok_new = m_new.min() <= PERIOD_FIX_NEW <= m_new.max()

    # ── Part 2: Jensen gap at ETT-calibrated amplitude ─────────────────────
    print("\n=== Jensen gap at ETT-calibrated load swing ===")
    gap_rows = []
    points = ([("frozen sampler (12-28%)", None)]
              + [(lbl, v) for lbl, v in ETT_POINTS])
    for lbl, amp in points:
        p = p_now if amp is None else replace(p_now, K_amp=(amp, amp))
        x0s, sens = build_realistic_set(N_GAP, 999, p)
        sw, gaps = swings_and_gaps(x0s, sens, want_gap=True)
        row = {
            "label": lbl, "K_amp": p.K_amp,
            "sw_med": float(np.median(sw)),
            "sw_p25": float(np.percentile(sw, 25)),
            "sw_p75": float(np.percentile(sw, 75)),
            "dp_med": float(np.median(gaps[:, GAP_DP])),
            "c2h2_med": float(np.median(gaps[:, GAP_C2H2])),
            "dp_p75": float(np.percentile(gaps[:, GAP_DP], 75)),
            "c2h2_p75": float(np.percentile(gaps[:, GAP_C2H2], 75)),
        }
        # C-10's analytic curve evaluated at the realised swing, as a cross-check
        # that the measurement sits on the curve the paper publishes.
        row["dp_c10"] = float(np.interp(row["sw_med"], C10_AMPLITUDE, C10_DP))
        row["c2h2_c10"] = float(np.interp(row["sw_med"], C10_AMPLITUDE, C10_C2H2))
        gap_rows.append(row)
        print(f"  {lbl:32s} K_amp {str(p.K_amp):16s} swing "
              f"{row['sw_med']:6.2f} degC  DP {row['dp_med']:.3f}  "
              f"C2H2 {row['c2h2_med']:.3f}")

    # ── Report ─────────────────────────────────────────────────────────────
    md: list[str] = []
    A = md.append
    A("# The pre-fix-7 arm, and the Jensen gap at ETT-calibrated swing\n")
    A(f"Generated by `audit_port/scripts/26_prefix7_arm_and_ett_gap.py`. "
      f"{len(SEEDS)} seeds x N={N} per arm; half peak-to-peak of the RK45 "
      "hot-spot trajectory, the quantity C-10's table is indexed by.\n")

    A("## 1. The comparison N-11 was missing\n")
    A(f"The pre-fix-7 sampler is `cod/data/realistic.py` at `{PREFIX7_REV}`, read "
      "out of git rather than reconstructed. Both arms run the current physics, "
      "steady state and `DailyMeanArrhenius`, since that commit already carries "
      f"fixes 1-6 and 8; the sampler is the only difference. Fields fix 7 added: "
      f"`{'`, `'.join(added)}`. `K_amp` is identical in both arms.\n")
    A("| seed | pre-fix-7 | fix 7 | ratio |")
    A("|---|---|---|---|")
    for s, a, b in zip(SEEDS, m_old, m_new):
        A(f"| {s} | {a:.3f} | {b:.3f} | {b / a:.4f} |")
    A(f"| **pooled** | **{med_old:.3f}** | **{med_new:.3f}** | "
      f"**{med_new / med_old:.4f}** |")
    A("")
    A(f"Between-seed sd is {m_old.std(ddof=1):.3f} degC on the old arm and "
      f"{m_new.std(ddof=1):.3f} on the new one, against a difference in pooled "
      f"medians of {abs(med_new - med_old):.3f} degC. The recorded figures:\n")
    A("| figure | recorded | measured here | reproducible |")
    A("|---|---|---|---|")
    A(f"| old-sampler median | {PERIOD_FIX_OLD:.2f} degC | {med_old:.3f} pooled, "
      f"per-seed {m_old.min():.2f}-{m_old.max():.2f} | "
      f"{'yes' if ok_old else '**no**'} |")
    A(f"| fix-7 median | {PERIOD_FIX_NEW:.2f} degC | {med_new:.3f} pooled, "
      f"per-seed {m_new.min():.2f}-{m_new.max():.2f} | "
      f"{'yes' if ok_new else '**no**'} |")
    A(f"| uplift (N-7) | {N7_UPLIFT:.3f} | {med_new / med_old:.4f} pooled, "
      f"per-seed {ratios.min():.3f}-{ratios.max():.3f} | "
      f"{'yes' if ok_uplift else '**no**'} |")
    A("")

    A("## 2. What follows for K_amp\n")
    A("PERIOD_FIX §2 derives its K_amp range by scaling 12-28% by the factor that "
      "restores the old median swing. With the arms measured properly that factor "
      f"is **{scale:.4f}** (recorded: 0.850), giving "
      f"**{100 * lo * scale:.1f}-{100 * hi * scale:.1f}%** of rated "
      "(recorded: 10.2-23.8%).\n")
    A("Against the feeders `ETT_LOAD_CALIBRATION.md` measures — ETTh2 median 8.7%, "
      "ETTh1 non-back-feeding 17.8%, ETTh1 back-feeding 29.7%:\n")
    A(f"* rescaled range **{100 * lo * scale:.1f}-{100 * hi * scale:.1f}%**")
    A("* ETTh2 at 8.7% sits "
      + ("inside" if lo * scale <= 0.087 <= hi * scale else "**below**")
      + " it")
    A("* ETTh1 non-back-feeding at 17.8% sits "
      + ("inside" if lo * scale <= 0.178 <= hi * scale else "**outside**")
      + " it")
    A("")

    A("## 3. The Jensen gap at ETT-calibrated swing\n")
    A(f"C-13 commits to presenting the gap as a curve against amplitude with the "
      f"real amplitude distribution over it, not as a single number. These are two "
      f"operating points taken straight from the ETT medians: `K_amp` set to that "
      f"value for every unit, N={N_GAP}, seed 999. DP and C2H2 are the ends of "
      "C-10's activation-energy range — the aging indicator and the arc-fault "
      "indicator.\n")
    A("| sampler setting | K_amp | realised swing degC (p25 / median / p75) | "
      "gap DP (median / p75) | gap C2H2 (median / p75) | C-10 curve at the median "
      "swing, DP / C2H2 |")
    A("|---|---|---|---|---|---|")
    for r in gap_rows:
        A(f"| {r['label']} | {r['K_amp'][0]:.3f}-{r['K_amp'][1]:.3f} | "
          f"{r['sw_p25']:.2f} / **{r['sw_med']:.2f}** / {r['sw_p75']:.2f} | "
          f"**{r['dp_med']:.3f}** / {r['dp_p75']:.3f} | "
          f"**{r['c2h2_med']:.3f}** / {r['c2h2_p75']:.3f} | "
          f"{r['dp_c10']:.3f} / {r['c2h2_c10']:.3f} |")
    A("")
    A("The last column is the cross-check that matters for C-13: the measured gap "
      "against C-10's analytic curve evaluated at the realised swing. Agreement "
      "means the published curve predicts the measurement and the paper can lead "
      "with the curve; disagreement means the curve and the sampler are saying "
      "different things and the discrepancy has to be explained before either is "
      "quoted.\n")
    A("Two things that column is not. It is a **linear** read of a curve that is "
      "convex in amplitude, so between nodes it overestimates, most noticeably "
      "below 5 degC where the only nodes are the analytic 1.00 at zero swing and "
      "C-10's 5 degC entry. And C-10 is tabulated for a pure sinusoid about "
      "100 degC, while these trajectories carry the family mix and their own "
      "operating temperatures. Read it as a consistency check on the order of "
      "magnitude, not as a second measurement.\n")
    A("Note that a degenerate `K_amp` gives every unit the same load amplitude, "
      "so the spread in realised swing here comes from the family mix, the "
      "operating point and the window phase alone. That is the point of an "
      "operating point rather than a population, but it is not the fleet "
      "distribution and should not be quoted as one.\n")

    A("## 4. Verdict\n")
    if ok_uplift:
        A(f"N-7's uplift of {N7_UPLIFT:.3f} is inside the per-seed range measured "
          "here, so the period argument survives and PERIOD_FIX §2 stands with a "
          "restated precision.\n")
    else:
        A(f"**N-7's uplift of {N7_UPLIFT:.3f} is not reproducible.** Measured "
          f"pooled uplift is {med_new / med_old:.4f} over {len(SEEDS)} seeds at "
          f"N={N} per arm, against a faithful pre-fix-7 sampler rather than the "
          "current one with its period reset. The three figures that derive from "
          f"it — {PERIOD_FIX_NEW:.2f} degC, the {N7_UPLIFT:.3f} uplift and the "
          "10.2-23.8% K_amp range — have to be restated from the pooled "
          "measurements above.\n")
        A("Whether that reopens O-10's scope decision is §2: if the rescaled "
          "K_amp range still sits above both measured feeders, the sampler is "
          "making an amplitude assumption and not only a period one, and saying "
          "so is a scope statement about which feeder population the benchmark "
          "describes — which is what O-10 was left open to decide.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    print("\nPASS" if ok_uplift else
          f"\nFAIL: N-7's uplift {N7_UPLIFT:.3f} is outside the measured "
          f"per-seed range {ratios.min():.3f}-{ratios.max():.3f}")
    return 0 if ok_uplift else 1


if __name__ == "__main__":
    raise SystemExit(main())
