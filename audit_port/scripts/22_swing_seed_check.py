#!/usr/bin/env python3
"""Is the realised hot-spot swing of the frozen sampler 13.18 or 11.20 degC?

`audit_port/PERIOD_FIX.md` §2 reports a median of **13.18 degC** for the fix-7
sampler, measured at N=200, seed 999. `21_test_set_provenance.py` measured
**11.20** at N=300, seed 42 — which is, awkwardly, the exact figure PERIOD_FIX
attributes to the *old* 12 h sampler. Two possibilities, and they have very
different consequences:

  * sampling noise / seed, in which case both are estimates of one number and the
    reported precision of "13.18" is overstated;
  * a real difference between what PERIOD_FIX measured and what
    `build_realistic_set` now produces from the frozen config, in which case the
    freeze certifies a distribution PERIOD_FIX does not describe.

The training set is drawn at `distribution.seed: 42`, so seed 42 is the one that
matters regardless of which explanation wins.

Run:  python audit_port/scripts/22_swing_seed_check.py
Exit: 0 if the two agree within the bootstrap spread of the estimator, 1 if the
      sampler and PERIOD_FIX genuinely disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import rk45_ground_truth  # noqa: E402
from cod.data.physics import N_SENSORS, TW, hot_spot_ETC_np  # noqa: E402
from cod.data.realistic import RealisticParams, build_realistic_set  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
PERIOD_FIX_MEDIAN = 13.18


def swings(n: int, seed: int, p: RealisticParams) -> np.ndarray:
    """Half peak-to-peak of the true hot-spot trajectory, per case.

    The same quantity PERIOD_FIX §2 tabulates and C-10's table is indexed by:
    RK45 on `fast_rhs_np`, so no model error enters.
    """
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


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p = RealisticParams.from_config(cfg["distribution"]["sampler"]["params"])
    print(f"[cfg] cycle_period={p.cycle_period:g}  K_amp={p.K_amp}  "
          f"hot_spot_mean={p.hot_spot_mean:g}")
    print("      config params identical to RealisticParams defaults: "
          f"{p == RealisticParams()}")

    runs = [(200, 999), (300, 42), (200, 42), (300, 999), (800, 42)]
    meds = {}
    print(f"\n{'N':>5} {'seed':>6} {'median':>9} {'p25':>8} {'p75':>8} {'mean':>8}")
    for n, s in runs:
        a = swings(n, s, p)
        meds[(n, s)] = float(np.median(a))
        print(f"{n:5d} {s:6d} {np.median(a):9.3f} {np.percentile(a, 25):8.3f} "
              f"{np.percentile(a, 75):8.3f} {a.mean():8.3f}")

    # Bootstrap the median at the size PERIOD_FIX used, from the largest sample,
    # to see whether 11.20 and 13.18 are two draws of one estimator.
    big = swings(800, 42, p)
    rng = np.random.RandomState(0)
    boot = np.array([np.median(rng.choice(big, 200, replace=True))
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nmedian of the frozen sampler (N=800, seed 42): {np.median(big):.3f} degC")
    print(f"bootstrap 95% interval for an N=200 median:   [{lo:.3f}, {hi:.3f}]")
    print(f"PERIOD_FIX reports {PERIOD_FIX_MEDIAN:.2f} degC at N=200, seed 999")

    inside = lo <= PERIOD_FIX_MEDIAN <= hi
    print("\nVERDICT: " + (
        "seed/sample noise — PERIOD_FIX's 13.18 is inside the spread of an "
        "N=200 median from this sampler. Quote the large-sample figure "
        f"({np.median(big):.2f} degC at N=800), not 13.18."
        if inside else
        "REAL DISAGREEMENT — 13.18 is outside the bootstrap spread. The frozen "
        "sampler does not produce what PERIOD_FIX describes."))
    return 0 if inside else 1


if __name__ == "__main__":
    raise SystemExit(main())
