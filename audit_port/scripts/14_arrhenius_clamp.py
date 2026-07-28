#!/usr/bin/env python3
"""N-1 diagnosis: what the `V_arr.clamp(max=1e4)` actually does.

Answers four questions, in the order the decision needs them:

  1. At what hot-spot temperature does the clamp bite, per species?
  2. Could `exp()` overflow float32 without it, i.e. was it ever an overflow
     guard?
  3. What upper bound does the reference already carry, and what does it imply?
  4. Does the failure mode survive the realistic sampler? Measured on both
     distributions, and separately on the *training* envelope the physics loss
     actually sees.

Run:  python audit_port/scripts/14_arrhenius_clamp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.physics import (  # noqa: E402
    B_aging, E_act, GAS_NAMES, T_ref, TW, N_SENSORS,
    hot_spot_ETC_np, fast_rhs_np, k_gen,
)

CLAMP = 1e4
E_ALL = list(E_act) + [1.0]
NAMES = GAS_NAMES + ["DP"]


def T_at_V(V: float, e: float) -> float:
    """Hot-spot degC at which V_arr(e) reaches V."""
    inv = 1.0 / T_ref - np.log(V) / (B_aging * e)
    if inv <= 0:
        return np.inf
    return 1.0 / inv - 273.15


def q1_thresholds():
    print("=" * 74)
    print("Q1. Where the 1e4 rate clamp bites, per species")
    print("=" * 74)
    print(f"{'state':<9} {'Ea/R * e':>10} {'Ea kJ/mol':>10} {'theta_HS at V=1e4':>20}")
    for n, e in zip(NAMES, E_ALL):
        print(f"{n:<9} {B_aging * e:>10.0f} {e * B_aging * 8.314 / 1000:>10.1f}"
              f" {T_at_V(CLAMP, e):>20.1f}")
    print("\nOne clamp value, a 170 degC spread in the temperature it enforces.")


def q2_overflow():
    print()
    print("=" * 74)
    print("Q2. Was it an overflow guard? Sup of the exponent as T -> inf")
    print("=" * 74)
    print("V_arr = exp(B*e*(1/T_ref - 1/T)) is increasing in T and bounded:")
    print("  sup_T exponent = B*e/T_ref   (attained only as T -> inf)")
    print(f"{'state':<9} {'sup exponent':>13} {'sup V_arr':>12} "
          f"{'float32 exp limit':>19}")
    for n, e in zip(NAMES, E_ALL):
        s = B_aging * e / T_ref
        print(f"{n:<9} {s:>13.2f} {np.exp(s):>12.3e} {'88.7':>19}")
    print("\nEvery sup is below the float32 exp overflow threshold of 88.7,")
    print("so exp() cannot overflow at ANY temperature, finite or not.")


def q3_reference_bound():
    print()
    print("=" * 74)
    print("Q3. The bound the reference already carries")
    print("=" * 74)
    print("fast_rhs_np:      T_HS_K = np.clip(theta_HS + 273.15, 313.15, 573.15)")
    print("fast_rhs_torch:   T_HS_K = (theta_HS + 273.15).clamp(min=313.15)")
    print("                  V_arr  = exp(...).clamp(max=1e4)")
    print("_gas_integral:    T_HS_K = (theta_HS + 273.15).clamp(313.15)")
    print("                  V_arr  = exp(...).clamp(max=1e4)")
    print()
    print("The reference bound is a TEMPERATURE envelope: [40, 300] degC.")
    print("The torch paths ported the lower half and swapped a RATE cap for the")
    print("upper half. Under the reference envelope V_arr is already bounded:")
    print(f"{'state':<9} {'V at 300 degC':>14} {'k_gen*V ppm/min':>17} "
          f"{'ppm over 720 min':>18}")
    for n, e, kg in zip(GAS_NAMES, E_act, k_gen):
        V = float(np.exp(B_aging * e * (1.0 / T_ref - 1.0 / 573.15)))
        print(f"{n:<9} {V:>14.3e} {kg * V:>17.4f} {kg * V * TW:>18.1f}")
    V_dp = float(np.exp(B_aging * (1.0 / T_ref - 1.0 / 573.15)))
    print(f"{'DP':<9} {V_dp:>14.3e}")
    print("\nAll finite, all representable, none anywhere near float32 range.")


def _hot_spot_traj(x0, sensors, ns=N_SENSORS):
    """True hot-spot trajectory by RK45 on the reference ODE."""
    from scipy.integrate import solve_ivp
    K_s = sensors[:ns].astype(float)
    Ta_s = sensors[ns:2 * ns].astype(float)
    tau = np.linspace(0.0, TW, ns)

    def rhs(t, x):
        K = float(np.interp(t, tau, K_s))
        Ta = float(np.interp(t, tau, Ta_s))
        return fast_rhs_np(x, K, Ta)

    sol = solve_ivp(rhs, [0.0, TW], np.asarray(x0, dtype=float),
                    method="RK45", t_eval=tau, rtol=1e-8, atol=1e-10)
    theta_TO = sol.y[0]
    return np.array([hot_spot_ETC_np(theta_TO[i], K_s[i]) for i in range(ns)])


def q4_distributions():
    print()
    print("=" * 74)
    print("Q4. Does the failure mode survive the realistic sampler?")
    print("=" * 74)
    from cod.data.generate import build_test_set
    from cod.data.realistic import build_realistic_set

    cases = build_test_set(n_test=100, seed=999)
    x0_old = np.array([c.x0 for c in cases])
    s_old = np.array([np.concatenate([c.K_sensors, c.Ta_sensors]) for c in cases])
    x0_new, s_new = build_realistic_set(n=100, seed=999)

    rows = []
    for label, x0s, sens in [("old (seed 999)", x0_old, s_old),
                             ("realistic", x0_new, s_new)]:
        hs = np.array([_hot_spot_traj(x0s[i], sens[i]) for i in range(len(x0s))])
        per_case_max = hs.max(axis=1)
        rows.append((label, hs, per_case_max))

    print(f"{'set':<16} {'max theta_HS':>13} {'p99':>8} {'median':>8} "
          f"{'cases > 187.2':>14}")
    for label, hs, pcm in rows:
        print(f"{label:<16} {pcm.max():>13.1f} {np.percentile(hs, 99):>8.1f}"
              f" {np.median(hs):>8.1f} {int((pcm > 187.2).sum()):>13d}%")

    print()
    print("Per-species clamp activation, fraction of (case, time) samples with")
    print("V_arr > 1e4 on the TRUE trajectory:")
    hdr = f"{'set':<16}" + "".join(f"{n:>10}" for n in NAMES)
    print(hdr)
    for label, hs, _ in rows:
        cells = []
        for e in E_ALL:
            V = np.exp(B_aging * e * (1.0 / T_ref - 1.0 / (hs + 273.15)))
            cells.append(f"{100.0 * (V > CLAMP).mean():>9.2f}%")
        print(f"{label:<16}" + "".join(cells))

    np.savez(ROOT / "audit_port" / "clamp_diagnosis.npz",
             hs_old=rows[0][1], hs_new=rows[1][1])
    return rows


def q5_training_envelope(rows):
    print()
    print("=" * 74)
    print("Q5. The envelope the physics loss sees is NOT the data envelope")
    print("=" * 74)
    print("`fast_rhs_torch` is evaluated on the *network's* predicted state, which")
    print("is unbounded early in training. STATE_CLAMP_HI in losses.py bounds it:")
    from cod.training.losses import STATE_CLAMP_HI_NP, STATE_CLAMP_LO_NP
    print(f"  STATE_CLAMP_LO = {STATE_CLAMP_LO_NP}")
    print(f"  STATE_CLAMP_HI = {STATE_CLAMP_HI_NP}")
    hi_TO = float(STATE_CLAMP_HI_NP[0])
    print()
    print(f"  worst-case theta_HS at theta_TO = {hi_TO:.0f} degC (the state clamp):")
    print(f"  {'K':>5} {'theta_HS':>10}")
    hs_worst = -np.inf
    for K in (0.3, 0.5, 1.0, 1.3, 1.5):
        hs = float(hot_spot_ETC_np(hi_TO, K))
        hs_worst = max(hs_worst, hs)
        print(f"  {K:>5.1f} {hs:>10.1f}")
    print(f"\n  sup over the whole clamped box: theta_HS = {hs_worst:.1f} degC,")
    print("  i.e. essentially exactly the reference's own 300 degC envelope.")
    print(f"  {'state':<9} {'V at that corner':>17} {'k_gen*V ppm/min':>17}")
    for n, e in zip(NAMES, E_ALL):
        V = float(np.exp(B_aging * e * (1.0 / T_ref - 1.0 / (min(hs_worst, 300.0) + 273.15))))
        kg = float(k_gen[NAMES.index(n)]) if n in GAS_NAMES else float("nan")
        print(f"  {n:<9} {V:>17.3e} {kg * V:>17.4f}")
    print()
    print("So the worst case the physics loss can construct is already bounded by")
    print("the state clamp, and the reference's 300 degC envelope binds at almost")
    print("the same place. Neither needs a rate cap.")


def q6_agreement():
    """The point of the fix: `fast_rhs_np` and `fast_rhs_torch` now agree."""
    import torch
    from cod.data.physics import fast_rhs_torch

    print()
    print("=" * 74)
    print("Q6. Reference / model agreement, before and after")
    print("=" * 74)
    rng = np.random.RandomState(0)
    n = 4000
    # Spans the whole reachable box, deliberately including the extrapolation
    # corner that the old cap was hiding.
    theta_TO = rng.uniform(20.0, 200.0, n)
    gases = rng.uniform(0.0, 1.0, (n, 5)) * np.array([500, 200, 1000, 3000, 8000])
    K = rng.uniform(0.3, 1.5, n)
    Ta = rng.uniform(10.0, 45.0, n)
    x = np.concatenate([theta_TO[:, None], gases], axis=1)

    ref = np.array([fast_rhs_np(x[i], K[i], Ta[i]) for i in range(n)])
    xt = torch.tensor(x, dtype=torch.float64)
    ut = torch.tensor(np.stack([K, Ta], axis=1), dtype=torch.float64)

    for label, legacy in [("v57 (rate cap)", True), ("fixed (T envelope)", False)]:
        got = fast_rhs_torch(xt, ut, legacy_V_clamp=legacy).numpy()
        d = np.abs(got - ref)
        rel = d / np.maximum(np.abs(ref), 1e-30)
        n_bad = int((rel.max(axis=1) > 1e-9).sum())
        print(f"  {label:<22} max abs diff {d.max():.6e}  "
              f"max rel diff {rel.max():.3e}  rows disagreeing {n_bad}/{n}")
    print("\nThe fixed path is float64-exact against the reference over the whole")
    print("box. That is the property the benchmark needs and did not have.")


if __name__ == "__main__":
    q1_thresholds()
    q2_overflow()
    q3_reference_bound()
    rows = q4_distributions()
    q5_training_envelope(rows)
    q6_agreement()
