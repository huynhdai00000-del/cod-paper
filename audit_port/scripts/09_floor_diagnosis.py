#!/usr/bin/env python3
"""Why does COD's C2H2 error survive a perfect thermal input?

Script 08 refuted the dissipation-linearisation hypothesis: solving
dc/dt = gen - k_dis*c exactly leaves the 0.5914 ppm C2H2 floor untouched.

Next candidate, found by reading the two RHS implementations against each other:

    cod/data/physics.py fast_rhs_np      (the ground truth, numpy)
        T_HS_K = np.clip(theta_HS + 273.15, 313.15, 573.15)
        V_arr  = np.exp(B_aging * E_act * (1/T_ref - 1/T_HS_K))      <- NO clamp

    cod/data/physics.py fast_rhs_torch   and   CODOperator._gas_integral
        V_arr = torch.exp(...).clamp(max=1e4)                        <- clamped

So the reference trajectory has an unbounded Arrhenius factor while the model's
cascade caps it at 1e4. For C2H2, E_act = 1.4, so V_arr crosses 1e4 at a hot-spot
temperature near 197 degC — reachable, because training ICs go to theta_TO = 150 degC
and the hot-spot sits above the top-oil temperature.

Test: recompute the floor with the clamp removed. If the floor collapses, the
mechanism is a reference/model mismatch, not a quadrature error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, rk45_ground_truth
from cod.data.physics import (
    B_aging,
    E_act,
    K_PD_onset,
    N_SENSORS,
    PD_gain,
    STATE_NAMES_FAST,
    T_ref,
    TW,
    hot_spot_ETC_np,
    k_dis,
    k_gen,
)
from cod.data.steady_state import formula_A

GAS = STATE_NAMES_FAST[1:]
N_EVAL = 50
GUARD = 0.9999
V_CLAMP = 1e4


def cascade(theta_grid, K_s, x0_gas, t_eval, clamp_V=True, exact_diss=False,
            T=TW):
    ns = len(theta_grid)
    s = np.linspace(0, T, ns)
    th_hs = np.array([hot_spot_ETC_np(float(theta_grid[i]), float(K_s[i]))
                      for i in range(ns)])
    T_HS_K = np.clip(th_hs + 273.15, 313.15, None)
    V = np.exp(B_aging * E_act[None, :] * (1.0 / T_ref - 1.0 / T_HS_K[:, None]))
    hit = (V > V_CLAMP)
    if clamp_V:
        V = np.minimum(V, V_CLAMP)
    pd = 1.0 + PD_gain * np.clip(K_s - K_PD_onset, 0, None) ** 2
    V = V.copy()
    V[:, 1] = V[:, 1] * pd
    gen = k_gen[None, :] * V
    ds = T / (ns - 1)

    if exact_diss:
        w = np.exp(k_dis[None, :] * s[:, None])
        integ = gen * w
        trap = 0.5 * (integ[:-1] + integ[1:]) * ds
        cum = np.vstack([np.zeros((1, 5)), np.cumsum(trap, axis=0)])
        out = np.zeros((len(t_eval), 5))
        for j, tt in enumerate(t_eval):
            F = np.array([np.interp(tt, s, cum[:, g]) for g in range(5)])
            dd = np.exp(-k_dis * tt)
            out[j] = x0_gas * dd + dd * F
    else:
        trap = 0.5 * (gen[:-1] + gen[1:]) * ds
        cum = np.vstack([np.zeros((1, 5)), np.cumsum(trap, axis=0)])
        out = np.zeros((len(t_eval), 5))
        for j, tt in enumerate(t_eval):
            F = np.array([np.interp(tt, s, cum[:, g]) for g in range(5)])
            out[j] = x0_gas + F - k_dis * x0_gas * tt
    return out, hit


def main() -> int:
    cases = build_test_set(n_test=100, seed=999, T=TW, steady_state=formula_A)
    t_eval = np.linspace(0, TW, N_EVAL)
    grid_t = np.linspace(0, TW, N_SENSORS)

    variants = {
        "as shipped (clamp, linearised)": dict(clamp_V=True, exact_diss=False),
        "no V_arr clamp":                 dict(clamp_V=False, exact_diss=False),
        "exact dissipation":              dict(clamp_V=True, exact_diss=True),
        "both fixes":                     dict(clamp_V=False, exact_diss=True),
    }
    mae = {k: np.zeros((100, 5)) for k in variants}
    frac_clamped = np.zeros((100, 5))
    hs_max = np.zeros(100)

    for k, c in enumerate(cases):
        xt = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval,
                               T=TW, t_clip_frac=GUARD)
        xg = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, grid_t,
                               T=TW, t_clip_frac=GUARD)
        for name, kw in variants.items():
            pred, hit = cascade(xg[:, 0], c.K_sensors, c.x0[1:], t_eval, **kw)
            mae[name][k] = np.abs(xt[:, 1:] - pred).mean(axis=0)
            if name == "as shipped (clamp, linearised)":
                frac_clamped[k] = hit.mean(axis=0)
        hs_max[k] = max(hot_spot_ETC_np(float(xg[i, 0]), float(c.K_sensors[i]))
                        for i in range(N_SENSORS))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/100")

    print(f"\nFloor with a PERFECT thermal input, mean MAE in ppm:\n")
    hdr = f"{'gas':10s}" + "".join(f"{n[:22]:>24s}" for n in variants)
    print(hdr)
    for i, g in enumerate(GAS):
        row = f"{g:10s}"
        for n in variants:
            row += f"{mae[n][:, i].mean():24.6f}"
        print(row)

    print(f"\nFraction of grid points where V_arr > 1e4 (the clamp binding):")
    for i, g in enumerate(GAS):
        f = frac_clamped[:, i]
        print(f"  {g:10s} mean {f.mean():7.3%}  cases affected "
              f"{int((f > 0).sum()):3d}/100  worst case {f.max():7.1%}")

    print(f"\nMax hot-spot temperature reached: mean {hs_max.mean():.1f} degC, "
          f"p90 {np.percentile(hs_max, 90):.1f}, max {hs_max.max():.1f}")
    thr = {g: T_ref * 1 for g in GAS}
    for i, g in enumerate(GAS):
        # T at which V_arr = 1e4 for this gas
        inv = 1.0 / T_ref - np.log(V_CLAMP) / (B_aging * E_act[i])
        print(f"  {g:10s} V_arr hits 1e4 at theta_HS = {1.0 / inv - 273.15:7.1f} degC")

    n_aff = int((frac_clamped[:, 1] > 0).sum())
    c2h2_aff = mae["as shipped (clamp, linearised)"][frac_clamped[:, 1] > 0, 1]
    c2h2_un = mae["as shipped (clamp, linearised)"][frac_clamped[:, 1] == 0, 1]
    print(f"\nC2H2 floor split by whether the clamp bound:")
    print(f"  clamp bound     ({n_aff:3d} cases): mean {c2h2_aff.mean():.4f} ppm"
          if n_aff else "  clamp bound     (  0 cases)")
    print(f"  clamp never     ({100 - n_aff:3d} cases): mean {c2h2_un.mean():.4f} ppm")

    np.savez(ROOT / "audit_port" / "floor_diagnosis.npz",
             **{f"mae_{i}": mae[n] for i, n in enumerate(variants)},
             frac_clamped=frac_clamped, hs_max=hs_max,
             variant_names=np.array(list(variants)))
    print("\nWrote audit_port/floor_diagnosis.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
