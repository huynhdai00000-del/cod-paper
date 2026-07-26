#!/usr/bin/env python3
"""Reconcile audit M-8's "20 of 50 CK cases lie outside the training load support".

Our reproduction of the seed-999 test set is bit-exact on every other
fingerprint (the training .npz matches byte for byte; case 33 is CK at K=1.398,
x0_TO=141.3 exactly as both the audit and the notebook's own stored output say).
So the test set is right and the disagreement is in the counting rule. This
script tries every plausible rule to find which one yields 20, and cross-checks
against the audit's own stored per-state ranges in 06_test_ranges.npy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, solve_test_set
from cod.data.steady_state import formula_A

cases = build_test_set(steady_state=formula_A)
ck = [c for c in cases if c.kind == "CK"]
K = np.array([c.K_mean for c in ck])
x0_TO = np.array([float(c.x0[0]) for c in ck])

print(f"CK K: n={len(K)} min={K.min():.4f} max={K.max():.4f} mean={K.mean():.4f}")
print(f"  expected count outside [0.5,1.2] for K~U(0.4,1.4): 50*0.3 = 15.0\n")

rules = {
    "K < 0.50 or K > 1.20  (audit's stated rule)": ((K < 0.50) | (K > 1.20)).sum(),
    "K < 0.50":                                     (K < 0.50).sum(),
    "K > 1.20":                                     (K > 1.20).sum(),
    "K < 0.50 or K > 1.30  (IC sampler's K range)": ((K < 0.50) | (K > 1.30)).sum(),
    "K < 0.40 or K > 1.20":                         ((K < 0.40) | (K > 1.20)).sum(),
    "K outside [0.5,1.2] OR x0_TO > 130":           (((K < 0.50) | (K > 1.20)) | (x0_TO > 130)).sum(),
    "all 100 cases outside [0.5,1.2]":              int(sum(
        (c.K_sensors.min() < 0.50) or (c.K_sensors.max() > 1.20) for c in cases)),
}
for name, n in rules.items():
    flag = "  <-- gives 20" if n == 20 else ""
    print(f"  {name:46s} {n:3d}{flag}")

print("\nCross-check against the audit's stored test ranges (06_test_ranges.npy):")
p = ROOT / "reference" / "audit" / "results" / "06_test_ranges.npy"
stored = np.load(p, allow_pickle=True)
print(f"  stored array shape={stored.shape} dtype={stored.dtype}")
gt = solve_test_set(cases, n_eval=50, t_clip_frac=0.9999)
ours = gt.max(axis=1) - gt.min(axis=1)          # (100, 6) per-case variation
print(f"  ours shape={ours.shape}")
if stored.shape == ours.shape:
    rel = np.abs(stored - ours) / np.maximum(np.abs(stored), 1e-30)
    print(f"  max relative difference: {rel.max():.3e}")
    print(f"  medians (ours):   {np.median(ours, axis=0)}")
    print(f"  medians (stored): {np.median(stored, axis=0)}")
    print("  -> test set identical" if rel.max() < 1e-6
          else "  -> test set DIFFERS")
else:
    print("  shape mismatch; dumping stored head:")
    print(stored[:3])
