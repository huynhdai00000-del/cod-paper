#!/usr/bin/env python3
"""Is the extracted analytic baseline bit-identical to the code it replaced?

`CODOperator._ode_baseline` was moved into
`cod.models.analytic_baseline.ode_baseline` so the factorial's with-baseline
cells for FNO, MIONet and S-DeepONet use **one** definition of `H` rather than a
second copy — the defect CLAUDE.md names by example, since `ode_physics_loss` once
existed in three versions.

A refactor of a physics routine is exactly the kind of change assumed harmless
and occasionally not. The Phase 1 gates cover it end to end but report to about
0.1%, which would hide a small numerical difference. This asserts **bit-identity**.

Method, the same as `29_cascade_refactor_identical.py` uses for the cascade: read
`cod/models/cod.py` as it stood before the refactor straight out of git, load it
as a separate module, and run both versions on identical inputs with identical
weights. Reading from git rather than keeping a copy is the point — a
hand-maintained snapshot of old code silently stops matching what it claims to be.

Two paths are exercised, and only the second reaches `AnalyticBaseline.on_grid`:

  * `_ode_baseline` at scattered query times, which is what `forward` uses;
  * `_ode_baseline` with a supplied `theta_ss_grid`, the cached path the dataset
    provides, since a caller holding the cache must get the same answer as one
    that recomputes.

Run:  python audit_port/scripts/35_baseline_refactor_identical.py
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
from cod.models.analytic_baseline import AnalyticBaseline  # noqa: E402
from cod.models.cod import CODOperator  # noqa: E402

#: The last commit that still contains the inlined baseline, i.e. the one before
#: the delegation.
PRE_REFACTOR_REV = "db54c26"


def load_pre_refactor():
    src = subprocess.run(
        ["git", "show", f"{PRE_REFACTOR_REV}:cod/models/cod.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        check=True).stdout
    if "def _ode_baseline" not in src or "exp_s = torch.exp(s_grid / tau)" not in src:
        raise SystemExit(f"{PRE_REFACTOR_REV} does not contain the inlined "
                         "baseline; pick a revision that predates the refactor.")
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

    kw = dict(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=64, p=32,
              n_layers=2, n_exp_feats=12, T=TW, x_mean=x_mean, x_std=x_std)
    failures = []

    for mode in ("true_fixed_point", "formula_C"):
        new = CODOperator(theta_ss_mode=mode, **kw)
        old = pre.CODOperator(theta_ss_mode=mode, **kw)
        old.load_state_dict(new.state_dict(), strict=True)
        new.eval()
        old.eval()

        B = 8
        t = torch.tensor(rng.uniform(0, TW, (B, 1)), dtype=torch.float32)
        u = torch.tensor(np.concatenate(
            [rng.uniform(0.3, 1.5, (B, N_SENSORS)),
             rng.uniform(10, 45, (B, N_SENSORS))], axis=1), dtype=torch.float32)
        x0_TO = torch.tensor(rng.uniform(30, 130, (B, 1)), dtype=torch.float32)
        x0 = torch.tensor(rng.normal(x_mean, x_std, (B, 6)),
                          dtype=torch.float32).abs()

        for tag, ss_grid in (("theta_ss recomputed", None),
                             ("theta_ss cached", "cache")):
            if ss_grid == "cache":
                with torch.no_grad():
                    ss_grid = new._theta_ss(u[:, :N_SENSORS], u[:, N_SENSORS:])
            with torch.no_grad():
                a = new._ode_baseline(x0_TO, u, t, theta_ss_grid=ss_grid)
                b = old._ode_baseline(x0_TO, u, t, theta_ss_grid=ss_grid)
            same = torch.equal(a, b)
            label = f"{mode}, {tag}"
            print(f"  [{'ok ' if same else 'DIFF'}] {label:36s} "
                  f"max|diff| {float((a - b).abs().max()):.3e}  "
                  f"bit-identical {same}")
            if not same:
                failures.append(label)

        # The whole forward, which is what the gates exercise.
        with torch.no_grad():
            fa, fb = new(x0, u, t), old(x0, u, t)
        fsame = torch.equal(fa, fb)
        print(f"  [{'ok ' if fsame else 'DIFF'}] {'full forward, ' + mode:36s} "
              f"max|diff| {float((fa - fb).abs().max()):.3e}  "
              f"bit-identical {fsame}")
        if not fsame:
            failures.append(f"full forward {mode}")

    # The standalone module the factorial cells will use must agree with the
    # method it was extracted from, or the with-baseline cells are running a
    # different H from COD and the factorial's baseline factor is not one
    # variable.
    print("\n=== standalone AnalyticBaseline against CODOperator ===")
    ab = AnalyticBaseline(T=TW, n_sensors=N_SENSORS)
    ref = CODOperator(theta_ss_mode="true_fixed_point", **kw)
    ref.eval()
    B = 8
    t = torch.tensor(rng.uniform(0, TW, (B, 1)), dtype=torch.float32)
    u = torch.tensor(np.concatenate(
        [rng.uniform(0.3, 1.5, (B, N_SENSORS)),
         rng.uniform(10, 45, (B, N_SENSORS))], axis=1), dtype=torch.float32)
    x0_TO = torch.tensor(rng.uniform(30, 130, (B, 1)), dtype=torch.float32)
    with torch.no_grad():
        a = ref._ode_baseline(x0_TO, u, t)
        b = ab(x0_TO, u, t)
    same = torch.equal(a, b)
    print(f"  [{'ok ' if same else 'DIFF'}] {'query-time H':36s} "
          f"max|diff| {float((a - b).abs().max()):.3e}  bit-identical {same}")
    if not same:
        failures.append("AnalyticBaseline vs CODOperator at query time")

    # `on_grid` is what the in-cascade with-baseline cells need, and it must
    # agree with the query-time path evaluated at those same times.
    t_grid = torch.linspace(0.0, TW, N_SENSORS).view(-1, 1)
    with torch.no_grad():
        grid = ab.on_grid(x0_TO, u, t_grid)
        one_by_one = torch.cat([
            ab(x0_TO, u, t_grid[j].view(1, 1).expand(B, 1)) for j in range(N_SENSORS)
        ], dim=1)
    same = torch.equal(grid, one_by_one)
    print(f"  [{'ok ' if same else 'DIFF'}] {'on_grid vs pointwise':36s} "
          f"max|diff| {float((grid - one_by_one).abs().max()):.3e}  "
          f"bit-identical {same}")
    if not same:
        failures.append("AnalyticBaseline.on_grid vs pointwise")

    # Parameter-free, which is what keeps the factorial's baseline factor from
    # also being a capacity change.
    n_par = sum(p.numel() for p in ab.parameters() if p.requires_grad)
    print(f"  [{'ok ' if n_par == 0 else 'FAIL'}] "
          f"{'AnalyticBaseline is parameter-free':36s} n_parameters = {n_par}")
    if n_par != 0:
        failures.append("AnalyticBaseline carries parameters")

    if failures:
        print("\nFAIL — the refactor changed the numbers:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS — the analytic baseline is bit-identical on every path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
