#!/usr/bin/env python3
"""How much of the seed-999 hot-spot swing is the IC mismatch, and how much the profile?

The realistic sampler can only fix the first. If the profiles themselves drive more
than 15 degC, the IC change alone will not reach the target and that has to be said
rather than engineered around quietly.

For each test case, three swings:
  actual      the IC the sampler drew
  consistent  theta_TO(0) set to the profile's own periodic steady state
  driving     peak-to-peak of theta_ss over the window, the forcing itself
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, rk45_ground_truth
from cod.data.physics import N_SENSORS, TW, hot_spot_ETC_np, tau_oil
from cod.data.steady_state import formula_A, true_fixed_point_np

N_T = 241
GUARD = 0.9999


def periodic_steady_theta0(K_s, Ta_s, n_cycles: int = 3) -> float:
    """theta_TO(0) for a unit that has been running this profile repeatedly.

    Integrates the thermal ODE over `n_cycles` copies of the window and returns the
    end value. With tau_oil = 150 min against a 720 min window, three cycles is
    ample: the transient decays by exp(-3*720/150) = e^-14.4.
    """
    ns = len(K_s)
    s = np.linspace(0.0, TW, ns)
    ss = true_fixed_point_np(K_s, Ta_s)
    ds = TW / (ns - 1)
    r = ds / tau_oil
    er = np.exp(-r)
    coef = (r - 1.0 + er) / r
    theta = float(ss[0])
    for _ in range(n_cycles):
        for k in range(ns - 1):
            theta = theta * er + ss[k] * (1.0 - er) + (ss[k + 1] - ss[k]) * coef
    return theta


def hot_spot_swing(theta_TO_traj, K_at) -> float:
    hs = np.array([hot_spot_ETC_np(float(theta_TO_traj[i]), float(K_at[i]))
                   for i in range(len(theta_TO_traj))])
    return 0.5 * (hs.max() - hs.min())


def main() -> int:
    cases = build_test_set(n_test=100, seed=999, T=TW, steady_state=formula_A)
    t = np.linspace(0.0, TW, N_T)
    grid = np.linspace(0.0, TW, N_SENSORS)

    actual = np.zeros(100)
    consistent = np.zeros(100)
    driving = np.zeros(100)
    offset = np.zeros(100)
    kinds = np.array([c.kind for c in cases])

    for k, c in enumerate(cases):
        K_at = np.interp(t, grid, c.K_sensors)
        xt = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t, T=TW,
                               t_clip_frac=GUARD)
        actual[k] = hot_spot_swing(xt[:, 0], K_at)

        th0 = periodic_steady_theta0(c.K_sensors, c.Ta_sensors)
        x0c = c.x0.copy()
        x0c[0] = th0
        xc = rk45_ground_truth(x0c, c.K_sensors, c.Ta_sensors, t, T=TW,
                               t_clip_frac=GUARD)
        consistent[k] = hot_spot_swing(xc[:, 0], K_at)
        offset[k] = float(c.x0[0]) - th0

        ss = true_fixed_point_np(np.interp(t, grid, c.K_sensors),
                                 np.interp(t, grid, c.Ta_sensors))
        driving[k] = 0.5 * (ss.max() - ss.min())

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/100")

    def row(lbl, a, m):
        return (f"  {lbl:34s} {a[m].mean():7.2f} {np.median(a[m]):7.2f} "
                f"{np.percentile(a[m], 90):7.2f} {a[m].max():7.2f} "
                f"{100 * (a[m] > 15).mean():7.0f}%")

    for lab, m in [("all 100", np.ones(100, bool)), ("constant K", kinds == "CK"),
                   ("time-varying", kinds == "TV")]:
        print(f"\n=== {lab} ===")
        print(f"  {'hot-spot swing, degC':34s} {'mean':>7s} {'median':>7s} "
              f"{'p90':>7s} {'max':>7s} {'>15C':>8s}")
        print(row("actual IC (current sampler)", actual, m))
        print(row("profile-consistent IC", consistent, m))
        print(row("theta_ss forcing (half p-p)", driving, m))

    print(f"\nIC offset from the consistent value, degC:")
    print(f"  mean |offset| {np.abs(offset).mean():.2f}  median "
          f"{np.median(np.abs(offset)):.2f}  p90 "
          f"{np.percentile(np.abs(offset), 90):.2f}  max {np.abs(offset).max():.2f}")
    print(f"\nShare of the swing removed by fixing the IC alone: "
          f"{100 * (1 - np.median(consistent) / np.median(actual)):.0f}% "
          f"(median {np.median(actual):.2f} -> {np.median(consistent):.2f} degC)")
    np.savez(ROOT / "audit_port" / "swing_decomposition.npz", actual=actual,
             consistent=consistent, driving=driving, offset=offset, kinds=kinds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
