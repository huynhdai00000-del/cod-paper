#!/usr/bin/env python3
"""Phase 2 fix 1: the iterative fixed point must equal the root-found one.

`true_fixed_point` (brentq) is the definition. `true_fixed_point_np` and
`true_fixed_point_torch` are the fast differentiable versions that go into IC
generation and the model's forward pass. If they disagree, fix 1 has replaced one
inconsistency with another.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.steady_state import (
    formula_A, formula_B, formula_C,
    true_fixed_point, true_fixed_point_np, true_fixed_point_torch,
)

fail = 0


def check(label, ok, detail=""):
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fail += 1


# The (K, theta_a) box actually used: IC sampler draws K in [0.4, 1.3],
# theta_a in [15, 40]; the CK test set reaches K = 1.4 and theta_a = 45.
Ks = np.linspace(0.30, 1.50, 49)
Tas = np.linspace(10.0, 50.0, 41)
KK, TT = np.meshgrid(Ks, Tas, indexing="ij")

print("=== Iterative vs root-found fixed point ===")
ref = np.array([[true_fixed_point(float(k), float(t)) for t in Tas] for k in Ks])
it_np = true_fixed_point_np(KK, TT)
err_np = np.abs(it_np - ref)
print(f"  grid {KK.size} points, K in [{Ks.min()}, {Ks.max()}], "
      f"theta_a in [{Tas.min()}, {Tas.max()}]")
print(f"  numpy iteration  max|err| = {err_np.max():.3e} degC  "
      f"mean = {err_np.mean():.3e}")
check("numpy iteration matches brentq to 1e-6", err_np.max() < 1e-6)

it_t = true_fixed_point_torch(torch.tensor(KK, dtype=torch.float64),
                              torch.tensor(TT, dtype=torch.float64)).numpy()
err_t = np.abs(it_t - ref)
print(f"  torch iteration  max|err| = {err_t.max():.3e} degC  "
      f"mean = {err_t.mean():.3e}")
check("torch iteration matches brentq to 1e-6", err_t.max() < 1e-6)

it_t32 = true_fixed_point_torch(torch.tensor(KK, dtype=torch.float32),
                                torch.tensor(TT, dtype=torch.float32)).numpy()
err_t32 = np.abs(it_t32 - ref)
print(f"  torch float32    max|err| = {err_t32.max():.3e} degC")
check("float32 iteration matches brentq to 1e-3 (float32 floor)",
      err_t32.max() < 1e-3)

print("\n=== Convergence rate (how many iterations are really needed) ===")
for n in (1, 2, 3, 5, 8, 12, 20, 24):
    e = np.abs(true_fixed_point_np(KK, TT, n_iter=n) - ref).max()
    print(f"  n_iter={n:2d}  max|err| = {e:.3e} degC")

print("\n=== Differentiability ===")
k = torch.tensor([1.0, 1.3], dtype=torch.float64, requires_grad=True)
ta = torch.tensor([30.0, 30.0], dtype=torch.float64, requires_grad=True)
out = true_fixed_point_torch(k, ta)
g_k, g_ta = torch.autograd.grad(out.sum(), [k, ta])
print(f"  d(theta_ss)/dK      = {g_k.tolist()}")
print(f"  d(theta_ss)/dtheta_a = {g_ta.tolist()}")
check("gradients finite and nonzero",
      bool(torch.isfinite(g_k).all() and torch.isfinite(g_ta).all()
           and (g_k.abs() > 0).all()))

print("\n=== What fix 1 changes, against the true fixed point ===")
print(f"  {'K':>5} {'Ta':>5} {'TRUE':>8} {'A':>8} {'A-TRUE':>8} {'C':>8} {'C-TRUE':>8}")
for k_, t_ in [(1.0, 30.0), (1.2, 30.0), (1.3, 30.0), (1.3, 45.0)]:
    tr = true_fixed_point(k_, t_)
    print(f"  {k_:5.2f} {t_:5.1f} {tr:8.2f} {formula_A(k_, t_):8.2f} "
          f"{formula_A(k_, t_) - tr:+8.2f} {formula_C(k_, t_):8.2f} "
          f"{formula_C(k_, t_) - tr:+8.2f}")

print(f"\n{'FIX 1 SOLVER OK' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
