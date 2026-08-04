#!/usr/bin/env python3
"""Before building FNO-with-baseline: where does the residual's energy sit?

THE QUESTION, AND WHY IT COMES FIRST. Adding the IEC baseline to FNO means the
spectral branch no longer represents `theta_TO(t)` but the residual
`theta_TO(t) - H(t)`. FNO truncates at `k_max = 16` of the 50 modes a 100-point
grid admits (Li et al. 2021 Eq. 5). `H` carries the first-order relaxation, which
is the dominant *low-frequency* content, so subtracting it removes exactly the
part FNO represents best and leaves a residual whose energy sits relatively
higher in the spectrum.

    If the residual's energy above mode 16 is negligible, the FNO-with-baseline
    cell tests delta-learning, which is what the factorial is for.

    If it is not, that cell tests MODE TRUNCATION wearing delta-learning's name,
    and a bad result would be misattributed.

Knowing which **before the number exists** is the point. This costs minutes on
ground truth we already have and it decides how one cell of the factorial may be
read.

A SECOND PREDICTION, TESTED HERE TOO. PORT_LOG J-90 predicted that FNO pays a
Gibbs penalty at the window edge because the FFT treats the window as periodic
while `theta_TO(0) != theta_TO(T)`. The residual should be *closer* to periodic:
it is zero at `t = 0` by the IC mask, and `residual(T) = theta_TO(T) - H(T)` is
small when `H` tracks the endpoint. So adding the baseline may *reduce* the
endpoint mismatch FNO pays for. Both the mismatch and its reduction are measured
below.

Run:  python audit_port/scripts/33_fno_spectral_precheck.py
Exit: 0 if the residual is representable within k_max, 1 if adding the baseline
      would push energy past the truncation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_realistic_test_set, rk45_ground_truth  # noqa: E402
from cod.data.physics import N_SENSORS, TW  # noqa: E402
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.models.cod import CODOperator  # noqa: E402
from cod.models.fno import FNO_MODES  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "FNO_SPECTRAL_PRECHECK.md"
N_CASES = 200
#: Fraction of energy above k_max above which the truncation, not delta-learning,
#: would dominate what the with-baseline cell measures.
TAIL_LIMIT = 0.05


def energy_above(sig: np.ndarray, k: int) -> float:
    """Fraction of spectral energy in modes strictly above `k`.

    The mean is removed first: mode 0 is the offset, which `H` also carries and
    which the truncation never discards, so leaving it in would make every
    signal look overwhelmingly low-frequency and hide the comparison.
    """
    s = sig - sig.mean()
    P = np.abs(np.fft.rfft(s)) ** 2
    tot = P.sum()
    return float(P[k + 1:].sum() / tot) if tot > 0 else 0.0


def main() -> int:
    params = RealisticParams.from_config(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        ["distribution"]["sampler"]["params"])
    cases = build_realistic_test_set(n_test=N_CASES, seed=999, params=params)

    # H(t) from COD's own analytic baseline, so this measures the residual the
    # FNO-with-baseline cell would actually see, not an approximation of it.
    m = CODOperator(state_dim=6, n_sensors=N_SENSORS, d_h=128, p=64, n_layers=4,
                    n_exp_feats=12, T=TW, x_mean=np.zeros(6), x_std=np.ones(6))
    m.eval()

    tq = np.linspace(0.0, TW, N_SENSORS)
    full_tail, res_tail = [], []
    mism_full, mism_res = [], []
    amp_full, amp_res = [], []
    with torch.no_grad():
        for c in cases:
            gt = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, tq, T=TW)
            theta = gt[:, 0]
            s = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                             dtype=torch.float32).unsqueeze(0).expand(N_SENSORS, -1)
            x0 = torch.tensor(c.x0, dtype=torch.float32
                              ).unsqueeze(0).expand(N_SENSORS, -1)
            t = torch.tensor(tq, dtype=torch.float32).unsqueeze(-1)
            H = m._ode_baseline(x0[:, 0:1].contiguous(), s.contiguous(),
                                t).squeeze(-1).numpy()
            resid = theta - H

            full_tail.append(energy_above(theta, FNO_MODES))
            res_tail.append(energy_above(resid, FNO_MODES))
            # Endpoint mismatch: what the periodic extension has to jump across.
            mism_full.append(abs(theta[-1] - theta[0]))
            mism_res.append(abs(resid[-1] - resid[0]))
            amp_full.append(theta.max() - theta.min())
            amp_res.append(np.abs(resid).max())

    full_tail = np.array(full_tail); res_tail = np.array(res_tail)
    mism_full = np.array(mism_full); mism_res = np.array(mism_res)
    amp_full = np.array(amp_full); amp_res = np.array(amp_res)

    print(f"=== energy above mode k_max = {FNO_MODES} (of 50 available) ===")
    print(f"{'signal':>28} {'median':>10} {'p90':>10} {'max':>10}")
    print(f"{'theta_TO (no baseline)':>28} {np.median(full_tail):10.2e} "
          f"{np.percentile(full_tail, 90):10.2e} {full_tail.max():10.2e}")
    print(f"{'theta_TO - H (residual)':>28} {np.median(res_tail):10.2e} "
          f"{np.percentile(res_tail, 90):10.2e} {res_tail.max():10.2e}")

    print(f"\n=== endpoint mismatch, what the FFT must jump across ===")
    print(f"{'signal':>28} {'median':>10} {'p90':>10}")
    print(f"{'|theta(T) - theta(0)|':>28} {np.median(mism_full):10.3f} "
          f"{np.percentile(mism_full, 90):10.3f}  degC")
    print(f"{'|resid(T) - resid(0)|':>28} {np.median(mism_res):10.3f} "
          f"{np.percentile(mism_res, 90):10.3f}  degC")

    print(f"\n=== amplitude the branch must represent ===")
    print(f"  theta_TO peak-to-peak, median   {np.median(amp_full):8.2f} degC")
    print(f"  |residual| max, median          {np.median(amp_res):8.2f} degC")
    print(f"  amplitude reduction             "
          f"{np.median(amp_full) / max(np.median(amp_res), 1e-9):8.1f}x")

    tail_med = float(np.median(res_tail))
    verdict_ok = tail_med <= TAIL_LIMIT
    md = ["# FNO spectral pre-check: is the residual representable within k_max?\n",
          "Generated by `audit_port/scripts/33_fno_spectral_precheck.py`, "
          f"n = {N_CASES} ground-truth trajectories from the frozen sampler. "
          "**Run before the FNO-with-baseline cells were written**, so its answer "
          "cannot have been chosen to suit a result.\n",
          "## The question\n",
          f"Adding the IEC baseline makes FNO's spectral branch represent "
          f"`theta_TO - H` instead of `theta_TO`. FNO truncates at "
          f"`k_max = {FNO_MODES}` of 50 modes. `H` carries the first-order "
          "relaxation — the dominant low-frequency content — so removing it takes "
          "away exactly what the truncation keeps. If the residual's energy sits "
          "above mode 16, that cell would measure mode truncation and not "
          "delta-learning.\n",
          f"| signal | energy above mode {FNO_MODES}, median | p90 | max |",
          "|---|---|---|---|",
          f"| `theta_TO` | {np.median(full_tail):.2e} | "
          f"{np.percentile(full_tail, 90):.2e} | {full_tail.max():.2e} |",
          f"| `theta_TO - H` | {np.median(res_tail):.2e} | "
          f"{np.percentile(res_tail, 90):.2e} | {res_tail.max():.2e} |", "",
          "## Endpoint mismatch — J-90's other prediction\n",
          "J-90 predicted FNO pays Gibbs ringing at the window edge because the "
          "FFT treats the window as periodic while `theta_TO(0) != theta_TO(T)`. "
          "The residual is zero at `t = 0` by the IC mask, so if `H` tracks the "
          "endpoint the residual is closer to periodic and the penalty should "
          "fall.\n",
          "| quantity | median | p90 |", "|---|---|---|",
          f"| `\\|theta(T) - theta(0)\\|` | {np.median(mism_full):.3f} degC | "
          f"{np.percentile(mism_full, 90):.3f} degC |",
          f"| `\\|resid(T) - resid(0)\\|` | {np.median(mism_res):.3f} degC | "
          f"{np.percentile(mism_res, 90):.3f} degC |", "",
          f"Amplitude the branch must represent falls from "
          f"{np.median(amp_full):.2f} degC peak-to-peak to "
          f"{np.median(amp_res):.2f} degC, a factor of "
          f"{np.median(amp_full) / max(np.median(amp_res), 1e-9):.1f}.\n",
          "## Verdict\n"]
    if verdict_ok:
        md.append(f"**The residual is representable within `k_max`.** Median "
                  f"energy above mode {FNO_MODES} is {tail_med:.2e}, under the "
                  f"{TAIL_LIMIT:.0%} limit set before measuring. The "
                  "FNO-with-baseline cell therefore tests delta-learning, and a "
                  "poor result there may be read as evidence about delta-learning "
                  "rather than about the truncation.\n")
    else:
        md.append(f"**The residual is NOT representable within `k_max`.** Median "
                  f"energy above mode {FNO_MODES} is {tail_med:.2e}, above the "
                  f"{TAIL_LIMIT:.0%} limit. The FNO-with-baseline cell would "
                  "measure mode truncation wearing delta-learning's name. Either "
                  "raise `k_max` for that cell and record it as a departure from "
                  "the paper's 1-d configuration, or report the cell with this "
                  "confound stated. **Do not read a poor result there as evidence "
                  "about delta-learning.**\n")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    print("\nPASS — residual representable within k_max" if verdict_ok else
          "\nFAIL — adding the baseline pushes energy past the truncation")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
