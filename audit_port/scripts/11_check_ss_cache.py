#!/usr/bin/env python3
"""The theta_ss cache must be bit-exact and must actually be faster.

Three assertions:
  1. The dataset's cached theta_ss equals the model's own `_theta_ss` bit for bit.
  2. A forward pass with the cache equals one without it, bit for bit, in both
     training mode (n_grid=20 sub-grid path) and eval mode (full grid).
  3. Per-epoch cost comes back to roughly the pre-fix-1 (formula_C) level.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import load_training_set, steady_state_on_grid
from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW
from cod.models.cod import CODOperator, steady_state_grid
from cod.training.train import CODTrainer

ART = ROOT / "reference" / "artifacts"
fail = 0


def check(label, ok, detail=""):
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fail += 1


ts = load_training_set(ART / "transformer_training_v57.npz")
cache = steady_state_on_grid(ts.sensors)

print("=== 1. dataset cache vs the model's own _theta_ss ===")
m = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
                n_layers=4, n_exp_feats=12, T=TW, x_mean=ts.x_mean,
                x_std=ts.x_std, theta_ss_mode="true_fixed_point")
u = torch.tensor(ts.sensors[:512], dtype=torch.float32)
own = steady_state_grid(m, u).numpy()
check("bit-identical to the model", np.array_equal(cache[:512], own),
      f"max|diff| = {np.abs(cache[:512] - own).max():.3e}")

print("\n=== 2. forward with cache == forward without, bit for bit ===")
torch.manual_seed(0)
x0 = torch.tensor(ts.x0s[:16], dtype=torch.float32)
uu = torch.tensor(ts.sensors[:16], dtype=torch.float32)
ss = torch.tensor(cache[:16], dtype=torch.float32)
tq = torch.rand(16, 1) * TW
for mode, flag in [("eval (full grid)", False), ("train (n_grid=20)", True)]:
    m.train(flag)
    with torch.no_grad():
        a = m(x0, uu, tq)
        b = m(x0, uu, tq, theta_ss_grid=ss)
    check(f"{mode}", torch.equal(a, b),
          f"max|diff| = {(a - b).abs().max().item():.3e}")

print("\n=== 3. gradient through t is preserved ===")
m.train(True)
tg = (torch.rand(16, 1) * TW).requires_grad_(True)
out = m(x0, uu, tg, theta_ss_grid=ss)
g = torch.autograd.grad(out[:, 0].sum(), tg, retain_graph=True)[0]
check("d(theta_TO)/dt finite and nonzero",
      bool(torch.isfinite(g).all() and (g.abs() > 0).any()),
      f"|g| mean {g.abs().mean().item():.4e}")

print("\n=== 4. per-epoch cost ===")
res = {}
for label, mode, use_cache in [("formula_C (pre-fix-1)", "formula_C", False),
                               ("true_fixed_point, no cache", "true_fixed_point", False),
                               ("true_fixed_point, cached", "true_fixed_point", True)]:
    torch.manual_seed(0)
    mm = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
                     n_layers=4, n_exp_feats=12, T=TW, x_mean=ts.x_mean,
                     x_std=ts.x_std, theta_ss_mode=mode)
    tr = CODTrainer(mm, ts.x0s[:2000], ts.sensors[:2000], n_fb=64, n_col=80,
                    max_epochs=25000,
                    theta_ss=(cache[:2000] if use_cache else None))
    tr.train_step()
    t0 = time.time()
    for _ in range(4):
        tr.train_step()
    res[label] = (time.time() - t0) / 4
    print(f"  {label:32s} {res[label]:6.2f} s/epoch"
          f"  -> 25k epochs = {res[label] * 25000 / 3600:5.1f} h")

base = res["formula_C (pre-fix-1)"]
slow = res["true_fixed_point, no cache"] / base
fast = res["true_fixed_point, cached"] / base
print(f"\n  fix 1 without cache: {slow:.2f}x the pre-fix-1 cost")
print(f"  fix 1 with cache:    {fast:.2f}x the pre-fix-1 cost")
print(f"  cache removed {(slow - fast) / max(slow - 1.0, 1e-9) * 100:.0f}% of "
      f"fix 1's overhead")
# 1.6x, not 1.5x: this machine's timings drift and an interleaved median-of-5
# measurement put the same quantity at 1.48x while a sequential one gave 1.47x.
# The threshold is a regression guard, not a precise claim.
check("cached cost within 1.6x of pre-fix-1", fast < 1.6, f"{fast:.2f}x")
print("\n  Why it is not 1.0x: the residual is the ONE theta_ss use that must stay")
print("  differentiable -- the query-time value feeding the `driving` trunk feature,")
print("  whose 20-iteration graph ode_physics_loss back-propagates six times per")
print("  step (once per state, retain_graph=True). Making that cheap would mean")
print("  interpolating theta_ss instead of evaluating theta_ss at interpolated")
print("  (K, Ta) -- a change to the model's semantics, not a cache. Not done.")

print(f"\n{'SS CACHE OK' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
