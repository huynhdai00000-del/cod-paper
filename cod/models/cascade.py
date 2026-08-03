"""The one-way gas cascade, as a component any architecture can sit upstream of.

WHY THIS MODULE EXISTS. DECISIONS C-11 asks for every tier-1 architecture in two
configurations: **monolithic**, predicting all six states directly, and
**in-cascade**, predicting only `theta_TO` with the five gases and DP obtained by
Arrhenius quadrature. The second configuration is the one that separates "the
cascade is the problem" from "this architecture is the problem" — if
FNO-monolithic fails and FNO-in-cascade does not, the cascade is implicated and
the architecture is not.

That requires the quadrature to be usable by FNO and MIONet, not only by
`CODOperator`. Copying it would put a third variant of a physics routine in the
repo, which is the defect CLAUDE.md names explicitly (`ode_physics_loss` existed
in three versions). So it is extracted here **once** and `CODOperator._gas_integral`
delegates to it. `audit_port/scripts/29_cascade_refactor_identical.py` asserts the
delegation is bit-identical, and the Phase 1 gates are the end-to-end check.

No `nn.Parameter` appears here and none may be added. The cascade is analytic: it
is the part of the model that is known physics rather than learned, which is what
makes "predict theta_TO and derive the rest" a meaningful ablation rather than a
smaller network.
"""

from __future__ import annotations

import torch

from cod.data.physics import (
    B_aging,
    DTheta_HS_R,
    E_act,
    K_PD_onset,
    PD_gain,
    R_load,
    T_HS_ref_C,
    T_ref,
    alpha_Cu,
    k_dis,
    k_gen,
    m_exp,
)

#: Reference envelope on the hot-spot temperature, matching `fast_rhs_np`.
#: Phase 2 fix 6 (DECISIONS N-1): the model and the reference must integrate the
#: same kinetics, so the bound is on temperature and not on the rate.
T_HS_K_MIN = 313.15
T_HS_K_MAX = 573.15
#: The v57 rate cap, reachable only through `legacy_V_clamp`.
LEGACY_V_ARR_CAP = 1e4


def resample_sensor_grid(u_sensors: torch.Tensor, ns: int,
                         full_ns: int) -> torch.Tensor:
    """`K` on an `ns`-point grid, linearly resampled from the `full_ns` sensors.

    Split out because both the quadrature and any caller that needs `K` on the
    same grid as a predicted trajectory must resample it the same way; two
    resamplings that differ by an interpolation convention would put the load and
    the temperature on subtly different grids inside a convex integral.
    """
    if ns == full_ns:
        return u_sensors[:, :ns]
    idx_f = torch.linspace(0, full_ns - 1, ns, device=u_sensors.device)
    idx_lo = idx_f.long().clamp(0, full_ns - 2)
    frac = (idx_f - idx_lo.float()).unsqueeze(0)
    K_full = u_sensors[:, :full_ns]
    return K_full[:, idx_lo] * (1 - frac) + K_full[:, idx_lo + 1] * frac


def hot_spot_from_theta_grid(theta_TO_grid: torch.Tensor,
                             K_s: torch.Tensor) -> torch.Tensor:
    """Hot-spot temperature on a grid, with the two clamps `fast_rhs_np` has.

    One Newton step on the copper-resistance correction, exactly as
    `hot_spot_ETC_np` does it: `Rf` is formed from a first-pass hot spot, clamped
    to [0.8, 1.5], and fed back.
    """
    fac_m0 = ((1.0 + K_s ** 2 * R_load) / (1.0 + R_load)) ** m_exp
    th_HS0 = theta_TO_grid + DTheta_HS_R * fac_m0
    Rf = (1.0 + alpha_Cu * (th_HS0 - T_HS_ref_C)).clamp(0.8, 1.5)
    fac_m1 = ((1.0 + K_s ** 2 * R_load * Rf) / (1.0 + R_load)) ** m_exp
    return theta_TO_grid + DTheta_HS_R * fac_m1


def gas_integral(t: torch.Tensor, u_sensors: torch.Tensor,
                 x0_gas: torch.Tensor, theta_TO_grid: torch.Tensor,
                 T: float, n_sensors: int,
                 k_gen_t: torch.Tensor, k_dis_t: torch.Tensor,
                 E_act_t: torch.Tensor,
                 interp: "callable",
                 legacy_V_clamp: bool = False) -> torch.Tensor:
    """Analytic Arrhenius quadrature for the five gases.

        c_i(t) = c_i(0) + trapz_0^t k_gen_i V_arr_i(theta_HS(s)) ds - k_dis_i c_i(0) t

    `k_gen_t`, `k_dis_t` and `E_act_t` are passed as tensors rather than read from
    `cod.data.physics` so the caller's registered buffers are used verbatim; that
    is what keeps the delegation from `CODOperator` bit-identical rather than
    merely equal to within float32 rounding.

    `interp` is the caller's grid interpolator, `(F_grid, t) -> F_t`, for the same
    reason: the interpolation convention belongs to the model that owns the grid.

    PHASE 2 FIX 6 (DECISIONS N-1). This quadrature is the model's entire gas
    prediction, so a cap here is a cap on the answer. It carried
    `V_arr.clamp(max=1e4)` while the reference ODE that produced every label caps
    the temperature at 573.15 K and never caps the rate. It now uses the
    reference's envelope. `legacy_V_clamp=True` restores v57.
    """
    ns = theta_TO_grid.shape[1]
    B = t.shape[0]
    ds = T / (ns - 1)
    K_s = resample_sensor_grid(u_sensors, ns, n_sensors)

    theta_HS_s = hot_spot_from_theta_grid(theta_TO_grid, K_s)
    if legacy_V_clamp:
        T_HS_K_s = (theta_HS_s + 273.15).clamp(T_HS_K_MIN)
    else:
        T_HS_K_s = (theta_HS_s + 273.15).clamp(T_HS_K_MIN, T_HS_K_MAX)
    V_arr_s = torch.exp(
        B_aging * E_act_t * (1.0 / T_ref - 1.0 / T_HS_K_s.unsqueeze(-1))
    )
    if legacy_V_clamp:
        V_arr_s = V_arr_s.clamp(max=LEGACY_V_ARR_CAP)
    # Partial-discharge enhancement on C2H2 only, above the onset load.
    pd_s = 1.0 + PD_gain * (K_s - K_PD_onset).clamp(min=0.0) ** 2
    V_arr_s = V_arr_s.clone()
    V_arr_s[:, :, 1] = V_arr_s[:, :, 1] * pd_s
    integrand = k_gen_t * V_arr_s
    trap = 0.5 * (integrand[:, :-1, :] + integrand[:, 1:, :]) * ds
    F_grid = torch.cat(
        [torch.zeros(B, 1, 5, device=t.device), torch.cumsum(trap, dim=1)], dim=1)
    F_t = interp(F_grid, t)
    return x0_gas + F_t - k_dis_t * x0_gas * t


class GasCascade(torch.nn.Module):
    """The quadrature with its constants, for architectures that are not COD.

    `CODOperator` keeps its own buffers and calls `gas_integral` directly; this
    wrapper exists so `FNOInCascade` and `MIONetInCascade` get the same physics
    without either inheriting from `CODOperator` (which would drag in a branch and
    trunk they do not use) or restating the constants.

    Deliberately parameter-free. `n_parameters()` on a model containing one of
    these must not count anything here, because the cascade is known physics and
    counting it as model capacity would make the in-cascade configuration look
    larger than the monolithic one for no learned reason.
    """

    def __init__(self, T: float, n_sensors: int, legacy_V_clamp: bool = False):
        super().__init__()
        self.T = float(T)
        self.n_sensors = int(n_sensors)
        self.legacy_V_clamp = bool(legacy_V_clamp)
        self.register_buffer("k_gen_buf", torch.tensor(k_gen, dtype=torch.float32))
        self.register_buffer("k_dis_buf", torch.tensor(k_dis, dtype=torch.float32))
        self.register_buffer("E_act_buf", torch.tensor(E_act, dtype=torch.float32))

    def interp_grid_5d(self, F_grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of a (B, ns, 5) grid at per-sample times t (B, 1).

        Matches `CODOperator._interp_grid_5d`; `29_cascade_refactor_identical.py`
        asserts the two agree exactly.
        """
        ns = F_grid.shape[1]
        pos = (t.squeeze(-1) / self.T).clamp(0.0, 1.0) * (ns - 1)
        lo = pos.floor().long().clamp(0, ns - 2)
        frac = (pos - lo.float()).unsqueeze(-1)
        idx = lo.view(-1, 1, 1).expand(-1, 1, 5)
        f_lo = torch.gather(F_grid, 1, idx).squeeze(1)
        f_hi = torch.gather(F_grid, 1, idx + 1).squeeze(1)
        return f_lo * (1 - frac) + f_hi * frac

    def forward(self, t: torch.Tensor, u_sensors: torch.Tensor,
                x0_gas: torch.Tensor, theta_TO_grid: torch.Tensor
                ) -> torch.Tensor:
        return gas_integral(
            t, u_sensors, x0_gas, theta_TO_grid, self.T, self.n_sensors,
            self.k_gen_buf, self.k_dis_buf, self.E_act_buf,
            self.interp_grid_5d, self.legacy_V_clamp)


__all__ = ["GasCascade", "gas_integral", "resample_sensor_grid",
           "hot_spot_from_theta_grid", "T_HS_K_MIN", "T_HS_K_MAX",
           "LEGACY_V_ARR_CAP"]
