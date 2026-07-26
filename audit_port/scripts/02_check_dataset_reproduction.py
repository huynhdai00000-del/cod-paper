#!/usr/bin/env python3
"""Prove the ported samplers reproduce the stored dataset bit for bit.

If this passes, physics.py + steady_state.py + profiles.py + generate.py all
replay the source's rng draw order exactly, and the checkpoints are being
evaluated against the same distribution they were trained on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, generate_training_set, load_training_set

STORED = ROOT / "reference" / "artifacts" / "transformer_training_v57.npz"

fail = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fail += 1


print("=== Training set reproduction (seed 42, N=8000) ===")
stored = load_training_set(STORED)
regen = generate_training_set(n_ic=8000, seed=42)

check("x0s exact match", np.array_equal(stored.x0s, regen.x0s),
      f"max|diff| = {np.abs(stored.x0s - regen.x0s).max():.3e}")
check("sensors exact match", np.array_equal(stored.sensors, regen.sensors),
      f"max|diff| = {np.abs(stored.sensors - regen.sensors).max():.3e}")
check("x_mean match", np.allclose(stored.x_mean, regen.x_mean, rtol=0, atol=1e-3),
      f"max|diff| = {np.abs(stored.x_mean - regen.x_mean).max():.3e}")
check("x_std match", np.allclose(stored.x_std, regen.x_std, rtol=0, atol=1e-3),
      f"max|diff| = {np.abs(stored.x_std - regen.x_std).max():.3e}")

print("\n=== Known defects visible in the stored data ===")
K = stored.sensors[:, :100]
print(f"  sensor K min                    {K.min():.6f}   (documented floor 0.3)")
print(f"  profiles dipping below 0.3      {(K.min(axis=1) < 0.3).sum()} / 8000 "
      f"({(K.min(axis=1) < 0.3).mean():.2%})")
check("unclipped 'step' branch reproduces K = 0.2571",
      abs(float(K.min()) - 0.257134) < 1e-5, f"got {K.min():.6f}")

Ta = stored.sensors[:, 100:]
tau = np.linspace(0, 720.0, 100)
# phase 0 => the ambient profile starts exactly at its mean
starts_at_mean = np.abs(Ta[:, 0] - Ta.mean(axis=1)) < 1e-2
check("ambient phase fixed at 0 in training",
      starts_at_mean.mean() > 0.99, f"{starts_at_mean.mean():.2%} of profiles")

iec = np.array([100.0, 35.0, 200.0, 700.0, 2000.0])
over_iec = (stored.x0s[:, 1:] > iec).any(axis=1)
hi_clamp = np.array([500.0, 200.0, 1000.0, 3000.0, 8000.0])
over_clamp = (stored.x0s[:, 1:] > hi_clamp).any(axis=1)
print(f"  ICs above IEC attention levels  {over_iec.mean():.1%}  (audit M-9 says 37.0%)")
print(f"  ICs above the physics-loss _hi  {over_clamp.mean():.1%}  (audit M-9 says 23.2%)")
check("M-9 IEC exceedance ~37.0%", abs(over_iec.mean() - 0.370) < 0.005)
check("M-9 _hi clamp exceedance ~23.2%", abs(over_clamp.mean() - 0.232) < 0.005)

print("\n=== Test set (seed 999, N=100) ===")
cases = build_test_set()
ck = [c for c in cases if c.kind == "CK"]
tv = [c for c in cases if c.kind == "TV"]
check("50 CK + 50 TV", len(ck) == 50 and len(tv) == 50)
Kck = np.array([c.K_mean for c in ck])
print(f"  CK K range                      {Kck.min():.3f} .. {Kck.max():.3f}")
outside = ((Kck < 0.50) | (Kck > 1.20)).sum()
print(f"  CK cases outside training K     {outside} / 50  (audit M-8 says 20)")
check("M-8: 20 of 50 CK cases extrapolate", outside == 20, f"got {outside}")

Kall = np.array([c.K_sensors for c in cases])
print(f"  realised K span over all cases  {Kall.min():.2f} .. {Kall.max():.2f}  "
      f"(manuscript claims [0.5, 1.3]; audit says 0.42-1.40)")

# audit M-10: the sole >10% case is CK, K = 1.398, x0_TO = 141.3
c33 = cases[33]
print(f"  case 33: kind={c33.kind}  K={c33.K_mean:.3f}  x0_TO={c33.x0[0]:.1f}")
check("M-10: case 33 is CK at K=1.398, x0_TO=141.3",
      c33.kind == "CK" and abs(c33.K_mean - 1.398) < 5e-4
      and abs(float(c33.x0[0]) - 141.3) < 0.05)

print(f"\n{'ALL CHECKS PASSED' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
