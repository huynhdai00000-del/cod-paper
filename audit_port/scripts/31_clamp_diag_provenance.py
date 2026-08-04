#!/usr/bin/env python3
"""Does `clamp_frac_*` measure the model's prediction, or the input batch?

THE ANOMALY. FNO, MIONet and S-DeepONet — three architectures with different
parameter counts, converging at different epochs — reported **identical** clamp
statistics: `state_hi` peak 7.8% at epoch 1466, `would_hit_scalar_500` peak 14.1%
at epoch 1016. Identical values at identical epochs across three different models
is not a coincidence and cannot be a property of any of them.

THE HYPOTHESIS. `clamp_frac_state_hi` is computed as
`(xp > hi).any(dim=-1).mean()` where `xp` is the model output. For an
**in-cascade** model the six outputs are not all predicted: `theta_TO` is, and the
five gases come from the analytic quadrature, which ends in

    return x0_gas + F_t - k_dis * x0_gas * t

so the gas channels are dominated by `x0_gas` — a value taken straight from the
training batch. `c_CO2` initial conditions reach 6116 ppm on this distribution
(`30_scalar500_clamp.py`), far above any ceiling, so `(xp > hi).any(dim=-1)` is
mostly answering "did this batch contain a high initial gas concentration",
which is a property of the **data**, not of the model.

And the batch sequence is identical across cells: `TrainingData.train_batch`
draws from a `torch.Generator` seeded from `training.seed`, which every matrix
config sets to the same value on the same dataset. So three different models see
the same batches in the same order and would report the same "clamp" fractions at
the same epochs.

WHAT IS AT STAKE. If the hypothesis holds:

  * every "past the midpoint" verdict from `clamp_onset_table` is meaningless for
    a cascade model, because the series describes the data order, not training;
  * the earlier `state_scalar_500` diagnosis attributed the 18% to the model
    over-predicting, and that attribution is wrong;
  * it is a fourth instance of PORT_LOG J-89's "silent sentinel" — a diagnostic
    that returns a confident number about something other than what its name says.

It would NOT follow that the clamp fix was wrong. The clamp is applied to `xp`
before `fast_rhs_torch`, so whatever its cause, truncating a 1456 ppm CO2 to 500
does distort the residual, and the two loops really did apply different ceilings.
This script separates "the fix was needed" from "the reason given for it".

Run:  python audit_port/scripts/31_clamp_diag_provenance.py
Exit: 0 if the diagnostic tracks the model, 1 if it tracks the input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import generate_realistic_training_set  # noqa: E402
from cod.data.physics import STATE_NAMES_FAST  # noqa: E402
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.models.fno import FNOInCascade, FNOMonolithic, fno_predict  # noqa: E402
from cod.models.mionet import MIONetInCascade, mionet_predict  # noqa: E402
from cod.models.sdeeponet import SDeepONetInCascade, sdeeponet_predict  # noqa: E402
from cod.training.losses import STATE_CLAMP_HI_NP, ode_physics_loss_shared  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "CLAMP_DIAG_PROVENANCE.md"


def main() -> int:
    params = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])
    ts = generate_realistic_training_set(n_ic=128, seed=42, params=params)
    x_mean, x_std = ts.x_mean, ts.x_std

    B = 32
    x0 = torch.tensor(ts.x0s[:B], dtype=torch.float32)
    u = torch.tensor(ts.sensors[:B], dtype=torch.float32)

    cells = [
        ("FNOInCascade", FNOInCascade, fno_predict, True),
        ("MIONetInCascade", MIONetInCascade, mionet_predict, True),
        ("SDeepONetInCascade", SDeepONetInCascade, sdeeponet_predict, True),
        ("FNOMonolithic", FNOMonolithic, fno_predict, False),
    ]

    print("=== same batch, different architectures and different weights ===")
    print(f"{'cell':22s} {'params':>9s} {'state_hi':>10s} {'>500':>8s} "
          f"{'state_lo':>9s}")
    got = {}
    for name, cls, pf, in_cascade in cells:
        # Two different random initialisations, to separate "the model" from
        # "the batch": if the diagnostic tracked the model, these would differ.
        vals = []
        for seed in (1, 7):
            torch.manual_seed(seed)
            m = cls(x_mean=x_mean, x_std=x_std)
            d = {}
            ode_physics_loss_shared(m, pf, x0, u, n_col=10, n_chunks=5,
                                    diagnostics=d)
            vals.append((d["clamp_frac_state_hi"],
                         d["clamp_frac_would_hit_scalar_500"],
                         d["clamp_frac_state_lo"]))
        got[name] = vals
        a, b = vals
        same = all(abs(x - y) < 1e-12 for x, y in zip(a, b))
        print(f"{name:22s} {m.n_parameters():>9,} {a[0]:10.4f} {a[1]:8.4f} "
              f"{a[2]:9.4f}   two inits identical: {same}")

    # What the initial conditions alone would give, with no model at all.
    hi = torch.tensor(STATE_CLAMP_HI_NP)
    ic_hi = float((x0 > hi).any(dim=-1).float().mean())
    ic_500 = float((x0 > 500).any(dim=-1).float().mean())
    print(f"\n{'INITIAL CONDITIONS ALONE':22s} {'-':>9s} {ic_hi:10.4f} "
          f"{ic_500:8.4f}   (no model involved)")

    # Which channel triggers it.
    print("\n=== which state channel exceeds the ceiling, in the ICs ===")
    for i, nm in enumerate(STATE_NAMES_FAST[:6]):
        f_hi = float((x0[:, i] > hi[i]).float().mean())
        f_500 = float((x0[:, i] > 500).float().mean())
        print(f"  {nm:>10} > per-state hi {100 * f_hi:6.2f}%   > 500 "
              f"{100 * f_500:6.2f}%")

    # The test: do the three in-cascade cells agree with each other and with the
    # ICs, while the monolithic one does not?
    casc = [got[n][0] for n in ("FNOInCascade", "MIONetInCascade",
                                "SDeepONetInCascade")]
    cascade_agree = all(abs(casc[0][j] - c[j]) < 1e-12
                        for c in casc[1:] for j in (0, 1))
    matches_ic = (abs(casc[0][0] - ic_hi) < 1e-12
                  and abs(casc[0][1] - ic_500) < 1e-12)
    mono = got["FNOMonolithic"][0]
    mono_differs = abs(mono[0] - casc[0][0]) > 1e-12 or abs(mono[1] - casc[0][1]) > 1e-12

    print(f"\nthree in-cascade cells agree exactly     : {cascade_agree}")
    print(f"and equal the initial conditions alone   : {matches_ic}")
    print(f"monolithic differs from them             : {mono_differs}")

    broken = cascade_agree and matches_ic
    md = ["# Does `clamp_frac_*` measure the model or the batch?\n",
          "Generated by `audit_port/scripts/31_clamp_diag_provenance.py`.\n",
          "## The anomaly\n",
          "FNO, MIONet and S-DeepONet — different parameter counts, different "
          "convergence epochs — reported identical clamp statistics at identical "
          "epochs. That cannot be a property of any of them.\n",
          "## Measurement\n",
          "| cell | state_hi | would_hit_500 | identical across two random inits |",
          "|---|---|---|---|"]
    for name, _, _, _ in cells:
        a, b = got[name]
        same = all(abs(x - y) < 1e-12 for x, y in zip(a, b))
        md.append(f"| `{name}` | {a[0]:.4f} | {a[1]:.4f} | {same} |")
    md.append(f"| **initial conditions alone, no model** | **{ic_hi:.4f}** | "
              f"**{ic_500:.4f}** | — |")
    md.append("")
    if broken:
        md.append("## Verdict: it measures the INPUT\n")
        md.append("The three in-cascade cells agree with each other **and with "
                  "the initial conditions alone**, to the last bit, and do not "
                  "change when the weights are reinitialised. For an in-cascade "
                  "model the five gas channels come from the analytic quadrature, "
                  "which ends in `x0_gas + F_t - k_dis * x0_gas * t`, so they are "
                  "dominated by `x0_gas` — a value taken straight from the batch. "
                  "`(xp > hi).any(dim=-1)` therefore answers *did this batch "
                  "contain a high initial gas concentration*, not *did the model "
                  "predict one*.\n")
        md.append("The identical epochs follow: `train_batch` draws from a "
                  "`torch.Generator` seeded from `training.seed`, which every "
                  "matrix config sets identically on the same dataset, so all "
                  "three cells see the same batches in the same order.\n")
        md.append("## Consequences\n")
        md.append("1. **Every 'past the midpoint' verdict from "
                  "`clamp_onset_table` is meaningless for a cascade model.** The "
                  "series describes the order of the data, not the progress of "
                  "training. That includes the `state_hi` readings reported for "
                  "COD and Ablation A, which are cascade models too.\n")
        md.append("2. **The earlier `state_scalar_500` diagnosis was wrong about "
                  "the cause.** It attributed 18% to the model over-predicting "
                  "gas concentrations. The number is mostly the initial "
                  "conditions arriving through the cascade.\n")
        md.append("3. **The clamp fix itself stands.** The clamp is applied to "
                  "`xp` before `fast_rhs_torch`, so truncating a 1456 ppm CO2 to "
                  "500 distorts the residual whatever put it there, and the two "
                  "training loops really did apply different ceilings to the same "
                  "quantity. What was wrong was the reason given, not the "
                  "change.\n")
        md.append("4. **Fourth instance of the 'silent sentinel' failure mode** "
                  "(PORT_LOG J-89): a diagnostic returning a confident number "
                  "about something other than what its name says. The previous "
                  "three were the NMAE floor, `theta_bias`, and `eol_months`.\n")
        md.append("## The fix\n")
        md.append("A clamp diagnostic must be computed on the part of the output "
                  "the model is responsible for. For a cascade model that is "
                  "`theta_TO` plus whatever the quadrature *adds*, not the "
                  "initial condition it carries. Reporting `state_hi` per channel "
                  "rather than `.any(dim=-1)` would also have made this visible "
                  "immediately, since the gas channels would have been obviously "
                  "constant across architectures.\n")
    else:
        md.append("## Verdict: it tracks the model\n")
        md.append("The cells do not agree and do not equal the ICs, so the "
                  "diagnostic is about the prediction and the anomaly needs "
                  "another explanation.\n")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    if broken:
        print("\nFAIL — clamp_frac_* on a cascade model measures the input batch, "
              "not the prediction.")
        return 1
    print("\nPASS — the diagnostic tracks the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
