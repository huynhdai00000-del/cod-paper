#!/usr/bin/env python3
"""Inspect every checkpoint we must load in Phase 1.

Prints, for each .pt in reference/artifacts:
  - top-level container type / keys
  - state_dict parameter and buffer names with shapes
  - the scalar values of registered physics buffers, so that shadowing bugs
    (Mono Fair registers 'ne' from the *parameter* n_exp=12, not the thermal
    exponent 0.8) are visible rather than inferred.

Read-only. Writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parents[2] / "reference" / "artifacts"

TARGETS = [
    "transformer_pideepOnet_v57.pt",
    "mono_fair_v2_perstate.pt",
    "mono_multihead.pt",
    "sweep_cod_p4.pt",
    "sweep_cod_p8.pt",
    "sweep_cod_p16.pt",
    "sweep_cod_p32.pt",
    "sweep_cod_p64.pt",
    "sweep_mono_fair_p4.pt",
    "sweep_mono_fair_p8.pt",
    "sweep_mono_fair_p16.pt",
    "sweep_mono_fair_p32.pt",
    "sweep_mono_fair_p64.pt",
    "battery_cod_v7.pt",
]

SCALAR_BUFS = {
    "tau_oil_buf", "R_load_buf", "n_exp_buf", "m_exp_buf", "DTheta_oil_R_buf",
    "DTheta_HS_R_buf", "alpha_Cu_buf", "T_HS_ref_C_buf",
    "tau", "R", "ne", "me", "Do", "Dhs", "ac", "Tr",
    "tau_buf", "ds_buf", "e_r_buf", "coef_buf", "T_mean", "T_std",
}


def show(name: str) -> None:
    path = ART / name
    if not path.exists():
        print(f"\n=== {name}: MISSING ===")
        return
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {name} ===")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        print(f"  container dict, keys: {sorted(obj.keys())}")
        sd = obj["model_state_dict"]
        hist = obj.get("hist")
        if hist:
            print(f"  hist: {len(hist)} entries, last = {hist[-1][:4]}")
    else:
        sd = obj
        print("  bare state_dict")
    total = 0
    for k, v in sd.items():
        n = v.numel() if hasattr(v, "numel") else 0
        total += n
        extra = ""
        if k in SCALAR_BUFS or (hasattr(v, "numel") and v.numel() <= 6):
            extra = f"  value={np.asarray(v.detach().cpu()).ravel().tolist()}"
        shape = str(tuple(v.shape)) if hasattr(v, "shape") else "-"
        print(f"    {k:32s} {shape:>20}{extra}")
    print(f"  total tensor elements: {total:,}")


def show_npz() -> None:
    path = ART / "transformer_training_v57.npz"
    d = np.load(path)
    print(f"\n=== transformer_training_v57.npz ===")
    for k in d.files:
        a = d[k]
        print(f"    {k:12s} shape={a.shape} dtype={a.dtype} "
              f"min={a.min():.6g} max={a.max():.6g}")
    print(f"  x_mean = {d['x_mean'].tolist()}")
    print(f"  x_std  = {d['x_std'].tolist()}")
    ks = d["sensors"][:, :100]
    print(f"  sensor K: min={ks.min():.6f} max={ks.max():.6f}  "
          f"(documented floor 0.3 -> below floor on "
          f"{(ks.min(axis=1) < 0.3).mean():.2%} of profiles)")


if __name__ == "__main__":
    show_npz()
    for t in TARGETS:
        show(t)
    sys.exit(0)
