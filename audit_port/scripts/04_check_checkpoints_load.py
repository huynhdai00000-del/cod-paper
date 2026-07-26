#!/usr/bin/env python3
"""Every checkpoint must load into the ported classes with strict=True.

A strict load proves the ported architecture has exactly the parameters and
buffers the trained model had - no silently missing head, no renamed block, no
extra buffer that would take its value from the constructor instead of the
checkpoint. It also confirms the shadowed `ne = 12.0` arrives from the file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import load_training_set
from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW
from cod.models.cod import CODOperator
from cod.models.monolithic import MonolithicFair, MonolithicMultiHead

ART = ROOT / "reference" / "artifacts"
ts = load_training_set(ART / "transformer_training_v57.npz")

fail = 0


def load(label: str, model, filename: str, key: str | None = None) -> None:
    global fail
    path = ART / filename
    if not path.exists():
        print(f"  [SKIP] {label:26s} {filename} not supplied")
        return
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj[key] if key else obj
    try:
        model.load_state_dict(sd, strict=True)
        ne = float(model.ne) if hasattr(model, "ne") else float(model.n_exp_buf)
        print(f"  [PASS] {label:26s} {model.n_parameters():>9,} params  "
              f"thermal exponent = {ne}")
    except RuntimeError as exc:
        fail += 1
        print(f"  [FAIL] {label:26s} {str(exc)[:300]}")


print("=== COD ===")
load("COD v57 (p=64)",
     CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
                 n_layers=4, n_exp_feats=12, T=TW,
                 x_mean=ts.x_mean, x_std=ts.x_std,
                 theta_ss_mode='formula_C'),
     "transformer_pideepOnet_v57.pt", key="model_state_dict")

for p in (4, 8, 16, 32, 64):
    load(f"COD sweep p={p}",
         CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS,
                     d_h=max(64, p * 2), p=p, n_layers=4, n_exp_feats=12, T=TW,
                     x_mean=ts.x_mean, x_std=ts.x_std,
                     theta_ss_mode='formula_C'),
         f"sweep_cod_p{p}.pt")

print("\n=== Monolithic ===")
load("Mono Fair (v2 per-state)",
     MonolithicFair(d_h=128, p=64, n_layers=4, n_exp=12,
                    x_mean=ts.x_mean, x_std=ts.x_std),
     "mono_fair_v2_perstate.pt")
load("Mono Multi-head",
     MonolithicMultiHead(d_h=128, p=64, n_layers=4, n_exp=12,
                         x_mean=ts.x_mean, x_std=ts.x_std),
     "mono_multihead.pt")
for p in (4, 8, 16, 32, 64):
    load(f"Mono sweep p={p}",
         MonolithicFair(d_h=max(64, p * 2), p=p, n_layers=4, n_exp=12,
                        x_mean=ts.x_mean, x_std=ts.x_std),
         f"sweep_mono_fair_p{p}.pt")
load("Mono SoftIC (v1)", MonolithicFair(x_mean=ts.x_mean, x_std=ts.x_std),
     "mono_fair_v1.pt")

print(f"\n{'ALL CHECKPOINTS LOAD STRICTLY' if fail == 0 else f'{fail} FAILED'}")
sys.exit(1 if fail else 0)
