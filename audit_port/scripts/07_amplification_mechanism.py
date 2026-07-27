#!/usr/bin/env python3
"""O-1: does the monolithic baseline's gas output pass through Arrhenius quadrature?

Static answer is in the forward passes (see AMPLIFICATION_MECHANISM.md). This
script is the decisive empirical test.

Construct a HYBRID: take the monolithic baseline's predicted theta_TO, and push it
through COD's analytical cascade (`CODOperator._gas_integral`) with the true gas
initial conditions. That isolates one question:

    what does Arrhenius amplification of a 13.41 degC thermal error actually
    produce, as opposed to what the monolithic architecture produces?

Four arms per gas, all on the seed-999 test set, all in ppm:

  COD              theta_TO from COD          -> COD cascade
  Mono actual      the baseline's own six outputs (whatever mechanism that is)
  Mono -> cascade  theta_TO from Mono         -> COD cascade      (the hybrid)
  GT -> cascade    theta_TO from RK45         -> COD cascade      (error floor)

The last arm matters: it separates quadrature error from thermal error, so the
hybrid's excess over it is attributable to the thermal input alone.

Repeated for Mono Fair and Mono Multi-head.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, load_training_set, rk45_ground_truth
from cod.data.physics import (
    E_act,
    IEC_ATTENTION,
    N_SENSORS,
    STATE_DIM_FAST,
    STATE_NAMES_FAST,
    TW,
    B_aging,
)
from cod.data.steady_state import formula_A
from cod.models.cod import CODOperator, cod_predict
from cod.models.monolithic import MonolithicFair, MonolithicMultiHead, mono_predict

ART = ROOT / "reference" / "artifacts"
OUT = ROOT / "audit_port" / "amplification_data.json"

# Everything is scored with n12's `evaluate_v44` right-edge guard so all arms are
# directly comparable. Gate 3's stored Mono figure used n15's 0.999; the recomputed
# 0.9999 value is reported alongside so the difference is visible, not hidden.
GUARD = 0.9999
N_EVAL = 50
GAS_NAMES = STATE_NAMES_FAST[1:]


def load_models(device):
    ts = load_training_set(ART / "transformer_training_v57.npz")
    cod = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
                      n_layers=4, n_exp_feats=12, T=TW,
                      x_mean=ts.x_mean, x_std=ts.x_std,
                      theta_ss_mode="formula_C").to(device)
    cod.load_state_dict(torch.load(ART / "transformer_pideepOnet_v57.pt",
                                   map_location=device,
                                   weights_only=False)["model_state_dict"],
                        strict=True)
    cod.eval()

    mf = MonolithicFair(d_h=128, p=64, n_layers=4, n_exp=12,
                        x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mf.load_state_dict(torch.load(ART / "mono_fair_v2_perstate.pt",
                                  map_location=device, weights_only=False),
                       strict=True)
    mf.eval()

    mh = MonolithicMultiHead(d_h=128, p=64, n_layers=4, n_exp=12,
                             x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mh.load_state_dict(torch.load(ART / "mono_multihead.pt", map_location=device,
                                  weights_only=False), strict=True)
    mh.eval()
    return cod, mf, mh


@torch.no_grad()
def cascade_from_theta(cod, theta_grid_np, x0_gas, u_sensors, t_q, device):
    """Push an arbitrary theta_TO grid through COD's analytical cascade.

    `theta_grid_np` is (N_SENSORS,) on the uniform sensor grid. It is broadcast to
    one row per query time, exactly as `CODOperator.forward` does with its own
    `_thermal_predict_grid` output.
    """
    n_q = t_q.shape[0]
    grid = torch.tensor(theta_grid_np, dtype=torch.float32, device=device)
    grid = grid.unsqueeze(0).expand(n_q, -1).contiguous()
    x0g = torch.tensor(x0_gas, dtype=torch.float32, device=device)
    x0g = x0g.unsqueeze(0).expand(n_q, -1).contiguous()
    u = u_sensors.expand(n_q, -1).contiguous()
    return cod._gas_integral(t_q, u, x0g, grid).cpu().numpy()


def main() -> int:
    device = torch.device("cpu")
    cod, mf, mh = load_models(device)
    cases = build_test_set(n_test=100, seed=999, T=TW, steady_state=formula_A)

    t_eval = np.linspace(0, TW, N_EVAL)
    grid_t = np.linspace(0, TW, N_SENSORS)
    t_q = torch.tensor(t_eval, dtype=torch.float32, device=device).unsqueeze(-1)
    t_grid_torch = torch.tensor(grid_t, dtype=torch.float32,
                                device=device).unsqueeze(-1)

    arms = ["cod", "mono_fair", "mono_mh",
            "hyb_mono_fair", "hyb_mono_mh", "hyb_gt"]
    mae = {a: np.zeros((100, STATE_DIM_FAST)) for a in arms}
    swing = np.zeros(100)
    kinds = []

    for k, c in enumerate(cases):
        kinds.append(c.kind)
        x_true = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval,
                                   T=TW, t_clip_frac=GUARD)
        # Ground-truth theta_TO on the sensor grid, for the floor arm.
        x_true_grid = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, grid_t,
                                        T=TW, t_clip_frac=GUARD)

        s_k = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                           dtype=torch.float32, device=device).unsqueeze(0)
        x0_t = torch.tensor(c.x0, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            p_cod = cod_predict(cod, x0_t.expand(N_EVAL, -1).contiguous(),
                                s_k.expand(N_EVAL, -1).contiguous(), t_q).numpy()
            p_mf = mono_predict(mf, x0_t.expand(N_EVAL, -1).contiguous(),
                                s_k.expand(N_EVAL, -1).contiguous(), t_q).numpy()
            p_mh = mono_predict(mh, x0_t.expand(N_EVAL, -1).contiguous(),
                                s_k.expand(N_EVAL, -1).contiguous(), t_q).numpy()
            # Each baseline's theta_TO evaluated on the 100-point sensor grid.
            g_mf = mono_predict(mf, x0_t.expand(N_SENSORS, -1).contiguous(),
                                s_k.expand(N_SENSORS, -1).contiguous(),
                                t_grid_torch).numpy()[:, 0]
            g_mh = mono_predict(mh, x0_t.expand(N_SENSORS, -1).contiguous(),
                                s_k.expand(N_SENSORS, -1).contiguous(),
                                t_grid_torch).numpy()[:, 0]

            gas_hyb_mf = cascade_from_theta(cod, g_mf, c.x0[1:], s_k, t_q, device)
            gas_hyb_mh = cascade_from_theta(cod, g_mh, c.x0[1:], s_k, t_q, device)
            gas_hyb_gt = cascade_from_theta(cod, x_true_grid[:, 0], c.x0[1:],
                                            s_k, t_q, device)

        mae["cod"][k] = np.abs(x_true - p_cod).mean(axis=0)
        mae["mono_fair"][k] = np.abs(x_true - p_mf).mean(axis=0)
        mae["mono_mh"][k] = np.abs(x_true - p_mh).mean(axis=0)
        for name, gas in [("hyb_mono_fair", gas_hyb_mf),
                          ("hyb_mono_mh", gas_hyb_mh),
                          ("hyb_gt", gas_hyb_gt)]:
            mae[name][k, 0] = np.nan          # these arms have no thermal output
            mae[name][k, 1:] = np.abs(x_true[:, 1:] - gas).mean(axis=0)

        # Realised hot-spot swing, from the true trajectory on the grid.
        from cod.data.physics import hot_spot_ETC_np
        th_hs = np.array([hot_spot_ETC_np(float(x_true_grid[i, 0]),
                                          float(c.K_sensors[i]))
                          for i in range(N_SENSORS)])
        swing[k] = 0.5 * (th_hs.max() - th_hs.min())

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/100 cases")

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'':12s} {'COD':>12s} {'Mono Fair':>12s} {'Mono MH':>12s}")
    print(f"{'theta_TO':12s} {mae['cod'][:, 0].mean():12.4f} "
          f"{mae['mono_fair'][:, 0].mean():12.4f} "
          f"{mae['mono_mh'][:, 0].mean():12.4f}   degC")

    print(f"\n{'gas':10s} {'COD':>11s} {'Mono act':>11s} {'Mono->casc':>11s} "
          f"{'MH act':>11s} {'MH->casc':>11s} {'GT->casc':>11s}   unit")
    for i, g in enumerate(GAS_NAMES, start=1):
        print(f"{g:10s} {mae['cod'][:, i].mean():11.4f} "
              f"{mae['mono_fair'][:, i].mean():11.4f} "
              f"{mae['hyb_mono_fair'][:, i].mean():11.4f} "
              f"{mae['mono_mh'][:, i].mean():11.4f} "
              f"{mae['hyb_mono_mh'][:, i].mean():11.4f} "
              f"{mae['hyb_gt'][:, i].mean():11.4f}   ppm")

    payload = {
        "guard": GUARD,
        "n_cases": 100,
        "kinds": kinds,
        "swing_amplitude_C": swing.tolist(),
        "iec_attention": dict(zip(GAS_NAMES, [float(v) for v in IEC_ATTENTION])),
        "E_act_kJ_per_mol": dict(zip(GAS_NAMES,
                                     [float(e * B_aging * 8.314 / 1000) for e in E_act])),
        "mae": {a: mae[a].tolist() for a in arms},
        "state_names": STATE_NAMES_FAST,
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
