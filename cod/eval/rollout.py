"""Multi-window CHI / DP lifetime rollout.

Ported from n12 cell 3: `plot_chi_trajectory`'s inner `predict_chi_lifetime`
(L1705) and `plot_chi_model_dp` (L1811). Data only — plotting stays out of the
package, and the notebook is not allowed to hold science.

The two source functions differ in one respect, and it is the interesting one:

    dp_source="reference"  DP is advanced from `theta_TO_ss_ETC(K_w, Ta_w)`, i.e.
                           reference physics. EOL then does not depend on the
                           model at all, so it cannot reflect model quality.
    dp_source="model"      DP is advanced from the model's predicted theta_TO, so
                           a thermal bias shows up as an EOL shift: a negative
                           bias gives a conservatively long life, a positive bias
                           an optimistically short one.

PHASE 2 FIX 1 lands here as `steady_state`, which now defaults to the true fixed
point. Under v57 this rollout used formula B while the model's own attractor was
formula C and every IC came from formula A — three different steady states in one
experiment.

That matters for audit M-5. The manuscript explains a -3 degC rollout bias by a
staircase in the ETC term at K = 1, but B and C are *identical* at K = 1 (the load
factor is 1, so every exponent gives 1) and the only non-smooth element, the Rf
clamp, is inactive there. Over the rollout's K_base range of 0.85-1.10 the C-B
disagreement runs +0.67 to -1.03 degC and flips sign, so it can account for about
1 degC at most, not 3. With all three sites unified there is no formula mismatch
left to appeal to, and the bias must be re-diagnosed against the retrained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cod.data.physics import (
    B_aging,
    DP0,
    DP_EOL,
    N_SENSORS,
    T_ref,
    TW,
    hot_spot_ETC_np,
    k0_aging,
)
from cod.data.steady_state import gas_ic_from_ss, true_fixed_point_np
from cod.training.losses import compute_chi


@dataclass
class RolloutResult:
    """One K_base scenario."""

    K_base: float
    years: np.ndarray
    chi: np.ndarray
    dp: np.ndarray
    theta_TO_end: np.ndarray      # model prediction at the end of each window
    theta_ss_ref: np.ndarray      # the steady state the bias is measured against
    reached_eol: bool

    @property
    def eol_years(self) -> float:
        return float(self.years[-1])

    @property
    def chi_violations(self) -> int:
        """Windows where CHI rose. Should be zero if the model is monotone."""
        return int(np.sum(np.diff(self.chi) > 1e-4))

    @property
    def theta_bias(self) -> np.ndarray:
        return self.theta_TO_end - self.theta_ss_ref


@torch.no_grad()
def chi_lifetime_rollout(model, K_base: float, max_years: int = 50,
                         n_eval: int = 20, T: float = TW,
                         dp_source: str = "model", steady_state=None,
                         device=None, max_windows: int | None = None
                         ) -> RolloutResult:
    """Roll the model forward window by window until DP reaches end of life.

    Each window: K_w = K_base + 0.05 sin(2 pi day / 365),
    Ta_w = 27 - 12 cos(2 pi day / 365), with intra-window K and Ta ripples of
    +/-0.05 and +/-2 degC. The next window's IC is the model's own prediction at
    the end of the previous one — theta_TO is continuous across windows and is
    deliberately not reset (the v55-B change).

    `steady_state` defaults to `true_fixed_point_np` (Phase 2 fix 1). Pass
    `formula_B` to reproduce v57.

    `max_windows` caps the horizon directly, for diagnostics that need many
    scenarios rather than long ones. `None` leaves the `max_years` behaviour
    untouched, so nothing that does not pass it can change.
    """
    if steady_state is None:
        steady_state = true_fixed_point_np
    if dp_source not in ("model", "reference"):
        raise ValueError("dp_source must be 'model' or 'reference'")
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    windows_per_year = int(365 * 1440 / T)
    if max_windows is None:
        max_windows = max_years * windows_per_year
    t_q = torch.linspace(T * 0.01, T, n_eval, device=device).unsqueeze(-1)
    t_eval_np = t_q.squeeze(-1).cpu().numpy()
    tau_np = np.linspace(0, T, N_SENSORS)
    sensor_grid = np.linspace(0, T, N_SENSORS)

    Ta_init = 27.0
    theta_ss0 = float(steady_state(K_base, Ta_init))
    gas_ic = gas_ic_from_ss(K_base, Ta_init, steady_state=steady_state)
    x_cur = torch.tensor(np.concatenate([[theta_ss0], gas_ic]).astype(np.float32),
                         dtype=torch.float32, device=device).unsqueeze(0)
    DP_cur = DP0

    chi_l, dp_l, yr_l, th_l, ss_l = [], [], [], [], []
    reached_eol = False

    for w in range(max_windows):
        t_day = w * T / 1440
        K_w = K_base + 0.05 * np.sin(2 * np.pi * t_day / 365)
        Ta_w = 27.0 - 12.0 * np.cos(2 * np.pi * t_day / 365)
        Ta_s = (Ta_w + 2.0 * np.sin(2 * np.pi * tau_np / T)).astype(np.float32)
        K_s = np.clip(K_w + 0.05 * np.sin(2 * np.pi * tau_np / T), 0.3, 1.5).astype(np.float32)

        s_t = torch.tensor(np.concatenate([K_s, Ta_s]), dtype=torch.float32,
                           device=device).unsqueeze(0)
        xp = model(x_cur.expand(n_eval, -1).contiguous(),
                   s_t.expand(n_eval, -1).contiguous(), t_q)

        K_interp = np.interp(t_eval_np, sensor_grid, K_s)
        if dp_source == "reference":
            theta_for_dp = np.full(n_eval, float(steady_state(K_w, Ta_w)))
        else:
            theta_for_dp = xp[:, 0].cpu().numpy().astype(float)

        V_arr_pts = np.array([
            np.exp(B_aging * (1.0 / T_ref
                              - 1.0 / (hot_spot_ETC_np(float(theta_for_dp[i]),
                                                       float(K_interp[i])) + 273.15)))
            for i in range(n_eval)
        ])
        delta_inv = k0_aging * np.trapz(V_arr_pts, t_eval_np)
        DP_cur = max(1.0 / (1.0 / DP_cur + delta_inv), DP_EOL)

        # CHI is computed AFTER the DP update, so CHI = 0 exactly when DP = EOL.
        chi_l.append(float(compute_chi(xp[-1:], DP_cur).item()))
        dp_l.append(DP_cur)
        yr_l.append(w * T / 1440 / 365)
        th_l.append(float(xp[-1, 0].item()))
        ss_l.append(float(steady_state(K_w, Ta_w)))

        x_cur = xp[-1:].clone()

        if DP_cur <= DP_EOL + 1.0:
            reached_eol = True
            break

    return RolloutResult(
        K_base=K_base, years=np.array(yr_l), chi=np.array(chi_l),
        dp=np.array(dp_l), theta_TO_end=np.array(th_l),
        theta_ss_ref=np.array(ss_l), reached_eol=reached_eol,
    )


def chi_lifetime_sweep(model, K_scenarios=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
                       **kwargs) -> dict[float, RolloutResult]:
    """The six-scenario sweep the source plots. Returns data, not figures."""
    return {K: chi_lifetime_rollout(model, K, **kwargs) for K in K_scenarios}


def bias_table(results: dict[float, RolloutResult]) -> str:
    """theta_TO bias per scenario — the quantity audit M-5 disputes."""
    lines = ["| K_base | EOL (yr) | reached EOL | mean bias degC | "
             "CHI violations |", "|---|---|---|---|---|"]
    for K, r in sorted(results.items()):
        lines.append(f"| {K:.2f} | {r.eol_years:.1f} | {r.reached_eol} | "
                     f"{r.theta_bias.mean():+.2f} | {r.chi_violations} |")
    return "\n".join(lines)


__all__ = ["RolloutResult", "chi_lifetime_rollout", "chi_lifetime_sweep",
           "bias_table"]
