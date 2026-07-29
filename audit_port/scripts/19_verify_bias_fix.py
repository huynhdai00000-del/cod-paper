#!/usr/bin/env python3
"""Verify the O-9 metric fix, on the two claims its docstring makes.

1. `cyclic_endpoint_theta` integrates only the thermal state, at a looser
   tolerance than the diagnosis script's six-state RK45 burn-in. Check 1 is that
   the two agree — the first attempt at this fix used `DailyMeanArrhenius`'s
   closed-form recurrence instead and was wrong by 0.11 to 0.30 degC, growing
   with load, because that recurrence treats `theta_ss` as a fixed target while
   the real ODE's `fac_n` depends on theta_TO through `Rf`. Asserting agreement
   is what caught it.

2. A model with zero error by construction — `ExactModel` from the diagnosis
   script, RK45 on `fast_rhs_np` wearing `CODOperator`'s call signature — must
   now score ~0 on `theta_bias`, where it scored -2.86 to -3.88 degC before.

Run:  python audit_port/scripts/19_verify_bias_fix.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.physics import N_SENSORS, TW, fast_rhs_np  # noqa: E402
from cod.data.steady_state import true_fixed_point_np  # noqa: E402
from cod.eval.rollout import chi_lifetime_rollout, cyclic_endpoint_theta  # noqa: E402

# `16_bias_diagnosis` is not an importable name, so load it by path to reuse
# ExactModel rather than restating it (one definition per function).
_spec = importlib.util.spec_from_file_location(
    "bias_diag", Path(__file__).with_name("16_bias_diagnosis.py"))
_bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bd)


def rk45_cyclic_endpoint(K_s, Ta_s, theta_seed, n_cycles=12):
    """The diagnosis script's definition, thermal state only."""
    tau = np.linspace(0.0, TW, N_SENSORS)
    x = np.concatenate([[theta_seed], np.array([50.0, 15.0, 80.0, 300.0, 800.0])])
    for _ in range(n_cycles):
        sol = solve_ivp(
            lambda s, y: fast_rhs_np(y, float(np.interp(s, tau, K_s)),
                                     float(np.interp(s, tau, Ta_s))),
            [0.0, TW], x, method="RK45", t_eval=[TW], rtol=1e-9, atol=1e-11)
        x = sol.y[:, -1]
    return float(x[0])


def main() -> int:
    tau = np.linspace(0.0, TW, N_SENSORS)
    print("=== 1. analytic recurrence vs RK45 burn-in ===")
    print(f"{'K_base':>7} {'Ta_w':>7} {'analytic':>10} {'RK45':>10} {'diff':>9}")
    worst = 0.0
    for K_base in (0.85, 1.00, 1.10):
        for Ta_w in (15.0, 27.0, 39.0):
            Ta_s = Ta_w + 2.0 * np.sin(2 * np.pi * tau / TW)
            K_s = np.clip(K_base + 0.05 * np.sin(2 * np.pi * tau / TW), 0.3, 1.5)
            seed = float(true_fixed_point_np(K_base, Ta_w))
            a = cyclic_endpoint_theta(K_s, Ta_s, seed)
            b = rk45_cyclic_endpoint(K_s, Ta_s, seed)
            worst = max(worst, abs(a - b))
            print(f"{K_base:7.2f} {Ta_w:7.1f} {a:10.4f} {b:10.4f} {a - b:+9.4f}")
    print(f"worst |analytic - RK45| = {worst:.4f} degC")

    print("\n=== 2. zero-error model against the fixed metric ===")
    m = _bd.ExactModel()
    print(f"{'K_base':>7} {'theta_bias':>12} {'theta_ss_offset':>17} "
          f"{'windows':>8}")
    all_bias = []
    for K_base in (0.85, 0.95, 1.00, 1.10):
        r = chi_lifetime_rollout(m, K_base, dp_source="model",
                                 steady_state=true_fixed_point_np,
                                 max_windows=30)
        all_bias.append(r.theta_bias)
        print(f"{K_base:7.2f} {r.theta_bias.mean():+12.4f} "
              f"{r.theta_ss_offset.mean():+17.4f} {len(r.theta_bias):8d}")
    pooled = np.concatenate(all_bias)
    print(f"pooled theta_bias: mean {pooled.mean():+.4f} degC, "
          f"max |.| {np.abs(pooled).max():.4f} degC")

    # The rollout starts at steady_state(K_base, 27.0), which is not the cyclic
    # state, so window 0 carries a genuine startup transient: the unit really is
    # off-cycle there. That is the metric working, not failing. Everything after
    # window 0 is what has to be ~0 for a zero-error model.
    print("\nfirst three windows of each scenario:")
    for K_base, b in zip((0.85, 0.95, 1.00, 1.10), all_bias):
        print(f"  K={K_base:.2f}  " + "  ".join(f"{v:+.4f}" for v in b[:3]))
    settled = np.concatenate([b[1:] for b in all_bias])
    print(f"excluding window 0: mean {settled.mean():+.6f} degC, "
          f"max |.| {np.abs(settled).max():.6f} degC")

    ok = worst < 0.01 and np.abs(settled).max() < 0.05
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
