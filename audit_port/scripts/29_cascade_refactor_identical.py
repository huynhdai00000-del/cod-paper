#!/usr/bin/env python3
"""Is the extracted gas cascade bit-identical to the code it replaced?

`CODOperator._gas_integral` was moved into `cod.models.cascade.gas_integral` so
that the C-11 in-cascade configurations of FNO, MIONet and S-DeepONet use one
definition of the Arrhenius quadrature rather than a second copy — the defect
CLAUDE.md names by example, since `ode_physics_loss` once existed three times.

A refactor of a physics routine is exactly the kind of change that is assumed
harmless and occasionally is not. The Phase 1 gates cover it end to end, but they
report to about 0.1%, which would hide a small numerical difference. This asserts
**bit-identity**: same bits out, not same to within float32 rounding.

Method: read `cod/models/cod.py` as it stood before the refactor straight out of
git, load it as a separate module, and run both versions on the same inputs with
the same weights. Reading it from git rather than keeping a copy is the same
approach `26_prefix7_arm_and_ett_gap.py` uses for the pre-fix-7 sampler, and for
the same reason — a hand-maintained snapshot of old code silently stops matching
what it claims to be.

Run:  python audit_port/scripts/29_cascade_refactor_identical.py
Exit: 0 only if every output matches bit for bit.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW  # noqa: E402
from cod.models.cod import CODOperator  # noqa: E402

#: The commit before the delegation. Resolved at run time so this keeps working
#: as history moves: the last commit that still contains the inlined quadrature.
PRE_REFACTOR_REV = "206713a"


def load_pre_refactor():
    src = subprocess.run(
        ["git", "show", f"{PRE_REFACTOR_REV}:cod/models/cod.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        check=True).stdout
    if "def _gas_integral" not in src or "V_arr_s = V_arr_s.clone()" not in src:
        raise SystemExit(f"{PRE_REFACTOR_REV} does not contain the inlined "
                         "quadrature; pick a revision that predates the refactor.")
    tmp = Path(tempfile.mkdtemp()) / "cod_pre.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("cod_pre_refactor", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    pre = load_pre_refactor()
    print(f"[git] pre-refactor cod.py from {PRE_REFACTOR_REV}")

    x_mean = np.array([90., 50., 3., 20., 60., 250.])
    x_std = np.array([15., 20., 2., 10., 30., 100.])

    failures = []
    # Both clamp regimes and both grid sizes: `ns == n_sensors` takes the direct
    # path and `ns != n_sensors` takes the resampling path, and only the second
    # exercises `resample_sensor_grid`.
    for legacy in (False, True):
        for ns in (N_SENSORS, 20):
            new = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS,
                              d_h=64, p=32, n_layers=2, n_exp_feats=12, T=TW,
                              x_mean=x_mean, x_std=x_std,
                              theta_ss_mode="true_fixed_point",
                              legacy_V_clamp=legacy)
            old = pre.CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS,
                                  d_h=64, p=32, n_layers=2, n_exp_feats=12, T=TW,
                                  x_mean=x_mean, x_std=x_std,
                                  theta_ss_mode="true_fixed_point",
                                  legacy_V_clamp=legacy)
            old.load_state_dict(new.state_dict(), strict=True)
            new.eval(); old.eval()

            B = 8
            t = torch.tensor(rng.uniform(0, TW, (B, 1)), dtype=torch.float32)
            u = torch.tensor(np.concatenate(
                [rng.uniform(0.3, 1.5, (B, N_SENSORS)),
                 rng.uniform(10, 45, (B, N_SENSORS))], axis=1),
                dtype=torch.float32)
            x0_gas = torch.tensor(rng.uniform(0.1, 400, (B, 5)),
                                  dtype=torch.float32)
            # Span the clamp envelope: below 313.15 K and above 573.15 K both.
            th_grid = torch.tensor(rng.uniform(20, 260, (B, ns)),
                                   dtype=torch.float32)

            with torch.no_grad():
                a = new._gas_integral(t, u, x0_gas, th_grid)
                b = old._gas_integral(t, u, x0_gas, th_grid)
            same = torch.equal(a, b)
            maxdiff = float((a - b).abs().max())
            tag = f"legacy_V_clamp={legacy}, ns={ns}"
            print(f"  [{'ok ' if same else 'DIFF'}] {tag:32s} "
                  f"max|diff| {maxdiff:.3e}  bit-identical {same}")
            if not same:
                failures.append(f"{tag}: max|diff| {maxdiff:.3e}")

            # And the whole forward, which is what the gates exercise.
            x0 = torch.tensor(rng.normal(x_mean, x_std, (B, 6)),
                              dtype=torch.float32).abs()
            with torch.no_grad():
                fa, fb = new(x0, u, t), old(x0, u, t)
            fsame = torch.equal(fa, fb)
            print(f"  [{'ok ' if fsame else 'DIFF'}] {'full forward, ' + tag:32s} "
                  f"max|diff| {float((fa - fb).abs().max()):.3e}  "
                  f"bit-identical {fsame}")
            if not fsame:
                failures.append(f"full forward {tag}")

    if failures:
        print("\nFAIL — the refactor changed the numbers:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS — bit-identical on every clamp regime and both grid paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
