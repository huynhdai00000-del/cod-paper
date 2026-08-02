#!/usr/bin/env python3
"""How much does the fix-7 sampler's median hot-spot swing move with the seed,
and does the 1.177x period uplift survive being measured properly?

`audit_port/PERIOD_FIX.md` §2 reports **13.18 degC** for the fix-7 sampler at
N=200 seed 999, against **11.20** for the old 12 h sampler at N=100, and
DECISIONS N-7 quotes the ratio 1.177 as the measured uplift. Both medians are
single-seed estimates at small N, and `22_swing_seed_check.py` found seed 42
giving 11.53 at N=800 — a 1.65 degC spread between two seeds of the same sampler.

A ratio of two noisy medians is noisier than either. This script measures both
periods over the same six seeds at the same N, so the uplift is a comparison of
two estimates of the same precision rather than two lucky draws. `cycle_period` is
the only parameter that differs between the two arms; `K_amp` is untouched, which
is the whole point of the period argument (N-6, N-7).

Run:  python audit_port/scripts/23_swing_multiseed.py
Exit: 0 if PERIOD_FIX's 13.18 and N-7's 1.177 are both inside the between-seed
      spread measured here, 1 if either is not.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.realistic import RealisticParams, build_realistic_set  # noqa: E402
from cod.data.generate import rk45_ground_truth  # noqa: E402
from cod.data.physics import N_SENSORS, TW, hot_spot_ETC_np  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
N = 500
SEEDS = (42, 999, 7, 123, 2024, 31337)

# The two recorded figures under test.
PERIOD_FIX_MEDIAN = 13.18    # PERIOD_FIX.md §2, N=200 seed 999
N7_UPLIFT = 1.177            # DECISIONS N-7, computed as 13.18 / 11.20


def swings(n, seed, p):
    x0s, sens = build_realistic_set(n, seed, p)
    tq = np.linspace(0.0, TW, N_SENSORS)
    out = np.empty(n)
    for i in range(n):
        K_s = sens[i, :N_SENSORS].astype(float)
        Ta_s = sens[i, N_SENSORS:].astype(float)
        gt = rk45_ground_truth(x0s[i].astype(float), K_s, Ta_s, tq, T=TW)
        hs = np.array([hot_spot_ETC_np(float(gt[j, 0]), float(K_s[j]))
                       for j in range(N_SENSORS)])
        out[i] = 0.5 * (hs.max() - hs.min())
    return out


def arm(period: float, p0: RealisticParams, label: str):
    """One period arm: per-seed medians and the pooled distribution."""
    p = replace(p0, cycle_period=period)
    print(f"\n--- {label}: cycle_period = {period:g} min ---")
    print(f"{'seed':>7} {'median':>9} {'p25':>8} {'p75':>8}")
    meds, allsw = [], []
    for s in SEEDS:
        a = swings(N, s, p)
        meds.append(float(np.median(a)))
        allsw.append(a)
        print(f"{s:7d} {np.median(a):9.3f} {np.percentile(a, 25):8.3f} "
              f"{np.percentile(a, 75):8.3f}")
    meds = np.array(meds)
    pooled = np.concatenate(allsw)
    print(f"between-seed medians: min {meds.min():.3f} max {meds.max():.3f} "
          f"sd {meds.std(ddof=1):.3f} | pooled median {np.median(pooled):.3f} "
          f"(N = {len(pooled)})")
    return meds, pooled


def main() -> int:
    p0 = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])
    print(f"N = {N} per seed, {len(SEEDS)} seeds, half peak-to-peak of the "
          "RK45 hot-spot trajectory.")
    print(f"K_amp = {p0.K_amp} in both arms — only cycle_period differs.")

    m24, pooled24 = arm(1440.0, p0, "fix 7, 24 h day")
    m12, pooled12 = arm(720.0, p0, "pre-fix-7, 12 h period (N-6 defect)")

    med24, med12 = float(np.median(pooled24)), float(np.median(pooled12))
    # Per-seed uplift, so each ratio pairs two estimates from the same draw.
    ratios = m24 / m12
    print(f"\n=== uplift from correcting the period ===")
    print("  ".join(f"{s}:{r:.3f}" for s, r in zip(SEEDS, ratios)))
    print(f"pooled medians {med12:.3f} -> {med24:.3f} degC, "
          f"ratio {med24 / med12:.4f}")
    print(f"per-seed ratio: median {np.median(ratios):.4f}  "
          f"min {ratios.min():.4f}  max {ratios.max():.4f}")

    ok_med = m24.min() <= PERIOD_FIX_MEDIAN <= m24.max()
    ok_ratio = ratios.min() <= N7_UPLIFT <= ratios.max()
    print(f"\nPERIOD_FIX reports {PERIOD_FIX_MEDIAN:.2f} degC (N=200, seed 999): "
          + ("inside" if ok_med else "OUTSIDE")
          + " the between-seed range measured here")
    print(f"N-7 reports an uplift of {N7_UPLIFT:.3f}: "
          + ("inside" if ok_ratio else "OUTSIDE")
          + " the between-seed range measured here")
    print("\nVERDICT: " + (
        "both recorded figures are consistent with this sampler."
        if ok_med and ok_ratio else
        f"at least one recorded figure is not reproducible. The pooled "
        f"estimates are {med24:.2f} degC and an uplift of "
        f"{med24 / med12:.3f}; quote those."))
    return 0 if (ok_med and ok_ratio) else 1


if __name__ == "__main__":
    raise SystemExit(main())
