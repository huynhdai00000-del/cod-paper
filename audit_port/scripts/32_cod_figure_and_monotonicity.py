#!/usr/bin/env python3
"""COD's six-state figure, the Theorem 2(iii) assumption, and the bounded
correction the IEC baseline provides.

Three things O-5 could not report, because it ran before figures were wired in
and before anyone asked what the analytic baseline lets the paper claim.

1. **The figure.** Regenerated from `artifacts/o5/predictions.npz` so COD can be
   put beside the three baseline figures on the same axes and the same case.

2. **Theorem 2(iii).** The manuscript's monotonicity result assumes
   `c_i,0 <= c_i,eq`, i.e. every gas starts at or below its equilibrium and can
   therefore only rise. The realistic sampler draws
   `gases = c_eq * U(0.45, 1.35)`, with a `fault_prob` minority multiplied by
   2-7x, so **the assumption is violated by construction** for any draw with a
   service factor above 1. This measures how often, and how often gases actually
   decrease in ground truth — the observable consequence, and the reason the
   figure looks different from the manuscript's Fig 4.

3. **Sigma and the realised correction.** The output is
   `H(t) + phi(t) * sigma * net(t)`, so `sigma` bounds how far the network is
   permitted to move the prediction away from the IEC standard, and
   `phi * sigma * net` is how far it actually moved. Both are reportable numbers
   rather than an argument, which is what the auditability claim needs.

Run:  python audit_port/scripts/32_cod_figure_and_monotonicity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cod.data.generate import build_realistic_test_set, rk45_ground_truth  # noqa: E402
from cod.data.physics import (  # noqa: E402
    B_aging, E_act, GAS_NAMES, IEC_ATTENTION, N_SENSORS, STATE_NAMES_FAST,
    T_ref, TW, hot_spot_ETC_np, k_dis, k_gen,
)
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.models.cod import CODOperator  # noqa: E402
from figures import fig_state_predictions  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
O5 = ROOT / "artifacts" / "o5"
OUT = ROOT / "audit_port" / "COD_FIGURE_AND_MONOTONICITY.md"
FIGDIR = ROOT / "audit_port" / "figures"
NQ = 60


def c_eq_at(theta_HS_C: np.ndarray) -> np.ndarray:
    """Equilibrium gas concentrations at a hot-spot temperature, (n, 5).

    `c_eq = k_gen V_arr / k_dis`, the same expression `sample_realistic_ic` uses
    to draw the initial condition and the same one `fast_rhs_np` relaxes toward.
    """
    T_HS_K = np.asarray(theta_HS_C, float) + 273.15
    V = np.exp(B_aging * np.asarray(E_act)[None, :]
               * (1.0 / T_ref - 1.0 / T_HS_K[:, None]))
    return np.asarray(k_gen)[None, :] * V / np.asarray(k_dis)[None, :]


def main() -> int:
    params = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])

    # ── 1. The figure ──────────────────────────────────────────────────────
    npz = np.load(O5 / "predictions.npz", allow_pickle=False)
    written = fig_state_predictions(
        npz["pred"], npz["gt"], npz["t_eval"],
        [str(s) for s in npz["state_names"]], FIGDIR,
        case=None, kinds=npz["kind"] if "kind" in npz.files else None)
    print(f"[fig] COD six-state figure -> {[p.name for p in written]}")

    # ── 2. Theorem 2(iii): is c_i,0 <= c_i,eq? ─────────────────────────────
    print("\n=== Theorem 2(iii): c_i,0 <= c_i,eq on this distribution ===")
    n_cases = 300
    cases = build_realistic_test_set(n_test=n_cases, seed=999, params=params)
    tq = np.linspace(0.0, TW, NQ)

    above_ic = np.zeros((n_cases, 5), bool)      # IC above equilibrium
    decreased = np.zeros((n_cases, 5), bool)     # gas actually fell over window
    net_change = np.zeros((n_cases, 5))
    for i, c in enumerate(cases):
        gt = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, tq, T=TW)
        K_q = np.interp(tq, np.linspace(0, TW, N_SENSORS), c.K_sensors)
        hs = np.array([hot_spot_ETC_np(float(gt[j, 0]), float(K_q[j]))
                       for j in range(NQ)])
        # The equilibrium the trajectory is relaxing toward, averaged over the
        # window: c_eq moves with temperature, so a single instant would be a
        # weaker test than the level the window actually pulls toward.
        ceq = c_eq_at(hs).mean(axis=0)
        above_ic[i] = np.asarray(c.x0[1:], float) > ceq
        net_change[i] = gt[-1, 1:] - gt[0, 1:]
        decreased[i] = net_change[i] < 0

    print(f"{'gas':>10} {'IC above c_eq':>15} {'actually fell':>15} "
          f"{'median change ppm':>19}")
    rows = []
    for g in range(5):
        f_above = float(above_ic[:, g].mean())
        f_dec = float(decreased[:, g].mean())
        med = float(np.median(net_change[:, g]))
        rows.append((GAS_NAMES[g], f_above, f_dec, med))
        print(f"{GAS_NAMES[g]:>10} {100 * f_above:14.1f}% {100 * f_dec:14.1f}% "
              f"{med:19.3e}")
    any_above = float(above_ic.any(axis=1).mean())
    any_dec = float(decreased.any(axis=1).mean())
    print(f"\nany gas starting above equilibrium : {100 * any_above:.1f}% of cases")
    print(f"any gas decreasing over the window : {100 * any_dec:.1f}% of cases")

    # ── 3. Sigma and the realised correction ───────────────────────────────
    print("\n=== the bounded correction: H(t) + phi(t) * sigma * net(t) ===")
    ck = torch.load(O5 / "model.pt", map_location="cpu", weights_only=False)
    model = CODOperator(state_dim=6, n_sensors=N_SENSORS, d_h=128, p=64,
                        n_layers=4, n_exp_feats=12, T=TW,
                        x_mean=np.zeros(6), x_std=np.ones(6),
                        theta_ss_mode=ck.get("theta_ss_mode",
                                             "true_fixed_point"))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    sigma = float(model.output_scale.detach()[0]) if hasattr(
        model, "output_scale") else float("nan")

    # Realised correction = prediction minus the analytic baseline alone.
    corr, sig_amp = [], []
    with torch.no_grad():
        for c in cases[:100]:
            s = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                             dtype=torch.float32).unsqueeze(0).expand(NQ, -1)
            x0 = torch.tensor(c.x0, dtype=torch.float32).unsqueeze(0).expand(NQ, -1)
            t = torch.tensor(tq, dtype=torch.float32).unsqueeze(-1)
            pred = model(x0.contiguous(), s.contiguous(), t)[:, 0].numpy()
            base = model._ode_baseline(x0[:, 0:1].contiguous(),
                                       s.contiguous(), t).squeeze(-1).numpy()
            corr.append(np.abs(pred - base))
            sig_amp.append(pred.max() - pred.min())
    corr = np.concatenate(corr)
    sig_amp = np.array(sig_amp)
    print(f"  sigma (theta_TO output scale)        {sigma:.4f} degC")
    print(f"  |correction| median / p95 / max      {np.median(corr):.4f} / "
          f"{np.percentile(corr, 95):.4f} / {corr.max():.4f} degC")
    print(f"  signal peak-to-peak, median          {np.median(sig_amp):.2f} degC")
    print(f"  correction as % of signal (median)   "
          f"{100 * np.median(corr) / np.median(sig_amp):.2f}%")

    md = ["# COD's figure, Theorem 2(iii), and the bounded correction\n",
          "Generated by `audit_port/scripts/32_cod_figure_and_monotonicity.py`.\n",
          "## 1. The figure\n",
          "`audit_port/figures/state_predictions.{pdf,svg}`, regenerated from "
          "`artifacts/o5/predictions.npz`. O-5 ran before figures were wired into "
          "`run.py`, so this is the first one for COD and it is on the same axes, "
          "the same median-case selection rule and the same conventions as the "
          "baseline figures.\n",
          "**Caption note, and it is not cosmetic.** The ground truth here differs "
          "from the manuscript's Fig 4 because the sampler changed. Under the "
          "realistic sampler (fix 7) the initial condition sits near the unit's "
          "own gas equilibrium rather than 30 degC away from it, so a window can "
          "start *above* equilibrium and the gases **decrease**. In the "
          "manuscript's sampler they essentially always rose.\n",
          "## 2. Theorem 2(iii): `c_i,0 <= c_i,eq` does not hold here\n",
          "The monotonicity result assumes every gas starts at or below its "
          "equilibrium. `sample_realistic_ic` draws "
          "`gases = c_eq * U(0.45, 1.35)`, with a `fault_prob = 0.08` minority "
          "multiplied by 2-7x, so **the assumption is violated by construction** "
          "whenever the service factor exceeds 1. Measured against the "
          "window-mean equilibrium the trajectory actually relaxes toward, "
          f"n = {n_cases}:\n",
          "| gas | IC above `c_eq` | actually decreased | median net change (ppm) |",
          "|---|---|---|---|"]
    for nm, fa, fd, med in rows:
        md.append(f"| `{nm}` | {100 * fa:.1f}% | {100 * fd:.1f}% | {med:.3e} |")
    md.append("")
    md.append(f"**{100 * any_above:.1f}% of cases have at least one gas starting "
              f"above equilibrium, and {100 * any_dec:.1f}% have at least one gas "
              "decreasing over the window.**\n")
    md.append("This is a property of the distribution, not of any model, so it "
              "holds for every cell in the matrix. Two consequences:\n")
    md.append("1. **Theorem 2(iii) cannot be stated unconditionally against this "
              "benchmark.** Either its hypothesis is restated to cover "
              "`c_i,0 > c_i,eq`, or the paper says explicitly that the guarantee "
              "applies to the sub-population that satisfies it and reports what "
              "fraction that is.\n")
    md.append("2. **Non-negativity is unaffected.** The gases relax *toward* "
              "equilibrium from above rather than growing without bound, so they "
              "stay positive; it is monotonicity that fails, not positivity. "
              "Worth separating, because the two are often stated together.\n")
    md.append("## 3. Sigma and the realised correction\n")
    md.append("The output is `H(t) + phi(t) * sigma * net(t)`, so `sigma` bounds "
              "how far the network may move the prediction away from the IEC "
              "standard and the realised `|prediction - H|` is how far it did.\n")
    md.append("| quantity | value |")
    md.append("|---|---|")
    md.append(f"| `sigma`, theta_TO output scale | {sigma:.4f} degC |")
    md.append(f"| realised correction, median | {np.median(corr):.4f} degC |")
    md.append(f"| realised correction, p95 | {np.percentile(corr, 95):.4f} degC |")
    md.append(f"| realised correction, max | {corr.max():.4f} degC |")
    md.append(f"| signal peak-to-peak, median | {np.median(sig_amp):.2f} degC |")
    md.append(f"| correction as % of signal | "
              f"{100 * np.median(corr) / np.median(sig_amp):.2f}% |")
    md.append("")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
