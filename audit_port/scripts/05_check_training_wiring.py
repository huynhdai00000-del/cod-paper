#!/usr/bin/env python3
"""Both trainers must satisfy the Trainable protocol and drive harness.train.

Checks, on a deliberately tiny budget so it runs in seconds on CPU:
  - train_step() returns loss, causal_weight_min and at least one clamp_frac_*
  - the harness's underflow check actually fires when weights underflow
  - validation_loss() and grad_norm() work and do not leave stray gradients
  - harness.train() completes and reports a stop_reason
  - the loss actually decreases over a few dozen steps (wiring, not convergence)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import load_training_set
from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW
from cod.models.cod import CODOperator
from cod.models.monolithic import MonolithicFair, mono_predict
from cod.training.harness import ConvergenceCriterion, train
from cod.training.train import CODTrainer, SharedPhysicsTrainer

ART = ROOT / "reference" / "artifacts"
ts = load_training_set(ART / "transformer_training_v57.npz")
SUB = 256
x0s, sensors = ts.x0s[:SUB], ts.sensors[:SUB]

fail = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fail += 1


def protocol_checks(name: str, trainer) -> None:
    print(f"\n=== {name} ===")
    m = trainer.train_step()
    check("train_step returns loss", "loss" in m and np.isfinite(m["loss"]),
          f"loss = {m.get('loss')!r}")
    check("train_step returns causal_weight_min", "causal_weight_min" in m,
          f"wm = {m.get('causal_weight_min'):.3e}")
    clamps = {k: v for k, v in m.items() if k.startswith("clamp_frac_")}
    check("train_step returns clamp_frac_*", len(clamps) > 0,
          ", ".join(f"{k[len('clamp_frac_'):]}={v:.1%}" for k, v in clamps.items()))
    v = trainer.validation_loss()
    check("validation_loss finite", np.isfinite(v), f"val = {v:.4e}")
    g = trainer.grad_norm()
    check("grad_norm finite", np.isfinite(g), f"|g| = {g:.4e}")
    stray = [n for n, p in trainer.model.named_parameters()
             if p.grad is not None and torch.any(p.grad != 0)]
    check("validation_loss leaves no stray gradients", len(stray) == 0,
          f"{len(stray)} params with nonzero grad")

    crit = ConvergenceCriterion(max_epochs=12, max_wall_seconds=300,
                                patience=2, check_every=4)
    out = train(trainer, crit, log_every=1000)
    check("harness.train completes", out.epochs_reached > 0,
          f"stop_reason={out.stop_reason}, epochs={out.epochs_reached}")
    check("pathology report captured wm", out.pathology.causal_weight_min is not None,
          f"wm = {out.pathology.causal_weight_min:.3e}")
    check("fix 3: causal weight never reaches exactly zero",
          out.pathology.causal_weight_underflowed is False
          and out.pathology.causal_weight_min > 0.0,
          f"min wm = {out.pathology.causal_weight_min:.3e}, "
          f"underflowed={out.pathology.causal_weight_underflowed}")
    check("pathology report captured clamps",
          len(out.pathology.clamp_hit_fraction) > 0,
          str({k: round(v, 4) for k, v in out.pathology.clamp_hit_fraction.items()}))
    check("non-converged run is reported as such",
          out.converged is False and out.stop_reason == "epoch_budget",
          f"converged={out.converged}")
    check("is_fair_comparison_candidate() False for a 12-epoch run",
          out.is_fair_comparison_candidate() is False)


torch.manual_seed(0)
cod = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=32, p=8,
                  n_layers=2, n_exp_feats=12, T=TW,
                  x_mean=ts.x_mean, x_std=ts.x_std)
protocol_checks("CODTrainer (train_v34)",
                CODTrainer(cod, x0s, sensors, n_fb=8, n_col=20, max_epochs=12))

torch.manual_seed(0)
mono = MonolithicFair(d_h=32, p=8, n_layers=2, n_exp=12,
                      x_mean=ts.x_mean, x_std=ts.x_std)
protocol_checks("SharedPhysicsTrainer (train_physics)",
                SharedPhysicsTrainer(mono, mono_predict, x0s, sensors,
                                     n_fb=8, n_col=20, max_epochs=12))

print("\n=== The harness underflow check fires on a real underflow ===")
from cod.training.harness import PathologyReport
rep = PathologyReport()
rep.causal_weight_min = 0.0
rep.causal_weight_underflowed = True
warns = rep.warnings()
check("underflow produces a warning", len(warns) == 1 and "underflow" in warns[0].lower())
print(f"        {warns[0][:120]}...")

print("\n=== Loss decreases over 40 steps (wiring only, not convergence) ===")
torch.manual_seed(0)
cod2 = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=32, p=8,
                   n_layers=2, n_exp_feats=12, T=TW,
                   x_mean=ts.x_mean, x_std=ts.x_std)
tr2 = CODTrainer(cod2, x0s, sensors, n_fb=8, n_col=20, max_epochs=40)
losses = [tr2.train_step()["loss"] for _ in range(40)]
print(f"  first 5 mean {np.mean(losses[:5]):.4e}   last 5 mean {np.mean(losses[-5:]):.4e}")
check("loss decreases", np.mean(losses[-5:]) < np.mean(losses[:5]))

print(f"\n{'TRAINING WIRING OK' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
