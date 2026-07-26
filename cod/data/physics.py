"""Reference physics of the transformer benchmark.

PHASE 1 — FAITHFUL PORT. Behaviour here matches `PI_DeepONet_v57_done.ipynb`
(reference/audit/extracted/n12.txt) as it actually runs, including the known
defects the audit recorded. Nothing is corrected in this module during Phase 1;
each defect carries a `KNOWN DEFECT` note naming the audit finding.

Source map (n12.txt = Paper/Code/PI_DeepONet_v57_done.ipynb, the authoritative
copy from the same directory as the checkpoint and the training data):

    constants               cell 0, L49-L110
    hot_spot_np             cell 0, L112
    hot_spot_ETC_np         cell 0, L117
    pd_factor_np            cell 0, L133
    fast_rhs_np             cell 0, L137
    compute_rhs_scale_physics  cell 0, L152
    fast_rhs_torch          cell 0, L174
    compute_deriv_scale     cell 0, L214
    update_DP               cell 0, L279
    compute_theta_HS_torch  cell 0, L651

Each function is defined exactly once here. Where the notebook defined a name
more than once with differing bodies, the surviving (last) definition is the
one ported, and the discarded ones are described in a comment.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

# ═══════════════════════════════════════════════════════════════════════════
# Thermal parameters (IEC 60076-7) — n12 cell 0 L52-L58
# ═══════════════════════════════════════════════════════════════════════════
tau_oil = 150.0        # Top-oil thermal time constant [min]
DTheta_oil_R = 55.0    # Rated top-oil temperature rise [degC]
DTheta_HS_R = 23.0     # Rated hot-spot gradient [degC]
R_load = 8.0           # Load loss / no-load loss ratio
n_exp = 0.8            # ONAN thermal exponent
m_exp = 1.3            # Hot-spot exponent
theta_a0 = 30.0        # Baseline ambient [degC]

# ETC parameters — n12 cell 0 L61-L64
alpha_Cu = 3.93e-3     # Copper resistance temperature coefficient [/K]
T_HS_ref_C = 98.0      # IEC 60076-7 hot-spot reference; a v44 fix from 75 degC.
                       # CDOL_BaseClass_LevelB_v4 still uses 75 (audit MINOR).
K_PD_onset = 0.85      # Partial discharge onset load factor
PD_gain = 6.0          # PD acceleration gain

# Aging — n12 cell 0 L74-L79. k0 is the exact IEEE C57.91 value; the rounded
# 3.70e-10 used by v2 yields DP = 171 rather than 200 at end of life.
B_aging = 15000.0            # Arrhenius constant [K] = Ea/R
T_ref = 383.0                # Reference temperature [K] = 110 degC + 273
T_life_min = 180_000 * 60    # [min] normal insulation life at T_ref
k0_aging = (1 / 200 - 1 / 1000) / T_life_min   # 3.7037e-10
DP0 = 1000.0
DP_EOL = 200.0

# ═══════════════════════════════════════════════════════════════════════════
# DGA parameters — n12 cell 0 L91-L96
#
# KNOWN GAP (audit M-7): these constants have no stated provenance anywhere in
# the source repository. They appear under a bare `# DGA parameters` header with
# no citation and no units. The manuscript's attribution to IEC 60599 is a
# category error: that standard interprets diagnoses, it does not specify
# generation-rate constants, dissipation constants or activation energies.
# Ported as-is because the checkpoints were trained with these values.
# ═══════════════════════════════════════════════════════════════════════════
GAS_NAMES = ["c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2"]
N_GAS = 5
k_gen = np.array([7.6e-6, 9.5e-8, 1.9e-6, 1.4e-5, 2.8e-5])
k_dis = np.array([1.0e-7, 5.0e-8, 8.0e-8, 1.0e-7, 1.0e-7])
E_act = np.array([0.9, 1.4, 1.1, 0.7, 0.6])
IEC_ATTENTION = [100, 35, 200, 700, 2000]

STATE_DIM_FAST = 6
STATE_NAMES_FAST = ["theta_TO", "c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2"]
TW = 720.0        # [min] = 12 h evaluation window
N_SENSORS = 100   # raised from 13 in v38 to reduce the baseline trapz error


# ═══════════════════════════════════════════════════════════════════════════
# Hot-spot maps
# ═══════════════════════════════════════════════════════════════════════════
def hot_spot_np(theta_TO, K):
    """Standard IEC hot-spot.

    Kept because the notebook keeps it, but it is not on any live path: the DP
    update and the ODE both use `hot_spot_ETC_np`.
    """
    fac = ((1 + K ** 2 * R_load) / (1 + R_load)) ** m_exp
    return theta_TO + DTheta_HS_R * fac


def hot_spot_ETC_np(theta_TO, K):
    """ETC-consistent hot-spot: one Newton step on R_load_eff(T).

    R_load_eff(T) = R_load * (1 + alpha_Cu * (theta_HS - T_HS_ref_C)), clamped
    to [0.8, 1.5]. The clamp is a low-load / low-ambient effect and never
    activates near K = 1 (audit step3 §3.5).
    """
    fac0 = ((1 + K ** 2 * R_load) / (1 + R_load)) ** m_exp
    theta_HS0 = theta_TO + DTheta_HS_R * fac0
    Rf = 1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C)
    R_eff = R_load * np.clip(Rf, 0.8, 1.5)
    fac1 = ((1 + K ** 2 * R_eff) / (1 + R_load)) ** m_exp
    return theta_TO + DTheta_HS_R * fac1


def pd_factor_np(K):
    """Partial-discharge acceleration of the C2H2 generation rate."""
    excess = max(K - K_PD_onset, 0.0)
    return 1.0 + PD_gain * excess ** 2


# ═══════════════════════════════════════════════════════════════════════════
# The data-generating ODE
# ═══════════════════════════════════════════════════════════════════════════
def fast_rhs_np(x, K, theta_a):
    """Reference ODE for the six fast states. Ground truth for every metric.

    State: [theta_TO, c_H2, c_C2H2, c_C2H4, c_CO, c_CO2].
    `pd_factor_np(K)` is applied to gen[1] (C2H2), which lands at index 2 of the
    returned derivative array.
    """
    theta_TO = x[0]
    dgas = x[1:6]
    theta_HS = hot_spot_ETC_np(theta_TO, K)
    T_HS_K = np.clip(theta_HS + 273.15, 313.15, 573.15)
    Rf = 1.0 + alpha_Cu * (theta_HS - T_HS_ref_C)
    R_load_eff = R_load * np.clip(Rf, 0.8, 1.5)
    fac_n = ((1 + K ** 2 * R_load_eff) / (1 + R_load)) ** n_exp
    d0 = (1.0 / tau_oil) * (DTheta_oil_R * fac_n - (theta_TO - theta_a))
    V_arr = np.exp(B_aging * E_act * (1.0 / T_ref - 1.0 / T_HS_K))
    gen = k_gen * V_arr
    gen[1] *= pd_factor_np(K)
    d_gas = gen - k_dis * dgas
    return np.concatenate([[d0], d_gas])


def compute_rhs_scale_physics(double_pd_factor: bool = False):
    """Physics-derived RHS_SCALE.

    PHASE 2 FIX 2 (audit MINOR, §8.3). `fast_rhs_np` already multiplies gen[1] by
    `pd_factor_np(K)`, and that element arrives at index 2 of the returned array.
    The source then applied `dx[2] *= pd_factor_np(K)` to the same element,
    squaring the factor — 2.215 becomes 4.906 at K = 1.3. The second application
    is now gone.

    `double_pd_factor=True` restores the v57 arithmetic, for the before/after
    comparison in PHASE2_EFFECTS.md.

    Blast radius in v57 was nil, and that is worth stating rather than glossing:
    the surviving `ode_physics_loss` uses a raw residual and never consumes
    RHS_SCALE (PORT_LOG J-4), so no published number changes. This is a hygiene
    fix — it stops the next person who reaches for RHS_SCALE from silently getting
    a squared partial-discharge factor on the acetylene channel.
    """
    rhs_max = np.zeros(STATE_DIM_FAST)
    for K in np.linspace(0.5, 1.3, 10):
        for T in np.linspace(40, 120, 10):
            x = np.array([T, 50.0, 15.0, 80.0, 300.0, 800.0])
            dx = np.abs(fast_rhs_np(x, K, 30.0))
            if double_pd_factor:
                dx[2] *= pd_factor_np(K)      # v57: squares the factor
            rhs_max = np.maximum(rhs_max, dx)
    return np.maximum(rhs_max, rhs_max.max() * 1e-4).astype(np.float32)


# Torch constants are cached per (device, dtype) instead of being bound to one
# module-level `device` global as the notebook does. Behaviour is identical; the
# package must work on CPU and GPU in the same process.
_TORCH_CONST_CACHE: dict[tuple, tuple] = {}


def _torch_consts(device, dtype=torch.float32):
    key = (str(device), dtype)
    if key not in _TORCH_CONST_CACHE:
        _TORCH_CONST_CACHE[key] = (
            torch.tensor(k_gen, dtype=dtype, device=device),
            torch.tensor(k_dis, dtype=dtype, device=device),
            torch.tensor(E_act, dtype=dtype, device=device),
        )
    return _TORCH_CONST_CACHE[key]


def fast_rhs_torch(x, u):
    """Differentiable counterpart of `fast_rhs_np`, used by the physics loss.

    `x` is (B, 6); `u` is (B, 2) holding [K, theta_a] interpolated at the
    collocation time. Two clamps here can hide behaviour and are reported by
    the training loop as `clamp_frac_*` (audit §8.4):
      * `Rf.clamp(0.8, 1.5)` on the ETC resistance correction,
      * `V_arr.clamp(max=1e4)` on the Arrhenius factor,
      * `T_HS_K.clamp(min=313.15)`.
    """
    k_gen_t, k_dis_t, E_act_t = _torch_consts(x.device, x.dtype)

    theta_TO = x[..., 0:1]
    dgas = x[..., 1:6]
    K = u[..., 0:1]
    theta_a = u[..., 1:2]

    # Step 0: non-ETC estimate
    fac_m0 = ((1.0 + K ** 2 * R_load) / (1.0 + R_load)) ** m_exp
    theta_HS0 = theta_TO + DTheta_HS_R * fac_m0
    # Step 1: ETC correction
    Rf1 = (1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C)).clamp(0.8, 1.5)
    R_eff1 = R_load * Rf1
    fac_m1 = ((1.0 + K ** 2 * R_eff1) / (1.0 + R_load)) ** m_exp
    theta_HS = theta_TO + DTheta_HS_R * fac_m1
    T_HS_K = (theta_HS + 273.15).clamp(min=313.15)

    # Thermal ODE
    Rf_n = (1.0 + alpha_Cu * (theta_HS - T_HS_ref_C)).clamp(0.8, 1.5)
    R_eff_n = R_load * Rf_n
    fac_n = ((1.0 + K ** 2 * R_eff_n) / (1.0 + R_load)) ** n_exp
    d0 = (1.0 / tau_oil) * (DTheta_oil_R * fac_n - (theta_TO - theta_a))

    # Gas ODEs
    V_arr = torch.exp(B_aging * E_act_t * (1.0 / T_ref - 1.0 / T_HS_K)).clamp(max=1e4)
    gen = k_gen_t * V_arr
    excess = torch.clamp(K - K_PD_onset, min=0.0)
    pd_fac = 1.0 + PD_gain * excess ** 2
    gen_c2h2 = gen[:, 1:2] * pd_fac
    gen_full = torch.cat([gen[:, :1], gen_c2h2, gen[:, 2:]], dim=-1)
    d_gas = gen_full - k_dis_t * dgas
    return torch.cat([d0, d_gas], dim=-1)


def compute_theta_HS_torch(x_fast, K_mean):
    """ETC-consistent theta_HS from the fast states (torch).

    Used by the CHI losses. `K_mean` may be a scalar or a (B,) tensor.
    """
    theta_TO = x_fast[:, 0]
    if isinstance(K_mean, (int, float)):
        K_t = torch.full_like(theta_TO, float(K_mean))
    else:
        K_t = K_mean.to(theta_TO.device)
    fac_m0 = ((1.0 + K_t ** 2 * R_load) / (1.0 + R_load)) ** m_exp
    theta_HS0 = theta_TO + DTheta_HS_R * fac_m0
    Rf_1 = (1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C)).clamp(0.8, 1.5)
    fac_m1 = ((1.0 + K_t ** 2 * R_load * Rf_1) / (1.0 + R_load)) ** m_exp
    return (theta_TO + DTheta_HS_R * fac_m1).clamp(40.0, 200.0)


# ═══════════════════════════════════════════════════════════════════════════
# Loss scales
# ═══════════════════════════════════════════════════════════════════════════
def compute_deriv_scale(T_window: float = 720.0, n_samples: int = 300,
                        percentile: int = 90) -> np.ndarray:
    """DERIV_SCALE: typical state variation per window, not peak rate.

    deriv_scale_i = P{percentile}(|x_i(T) - x_i(0)|) / T_window.

    Costs ~300 `solve_ivp` calls, so it is a function rather than an
    import-time constant as in the notebook. The live v57 loss does not use it
    (the surviving `ode_physics_loss` divides by nothing); it is retained
    because the discarded cell-0 loss did, and Phase 2 comparisons may need it.
    """
    rng = np.random.RandomState(0)
    deltas = []

    for _ in range(n_samples):
        K = rng.uniform(0.4, 1.3)
        Ta = rng.uniform(15, 40)
        theta_TO = Ta + rng.uniform(10, 70)
        theta_HS = hot_spot_ETC_np(theta_TO, K)
        T_HS_K = theta_HS + 273.15
        V_arr = np.exp(B_aging * E_act * (1.0 / T_ref - 1.0 / T_HS_K))
        c_eq = k_gen * V_arr / k_dis
        x0 = np.concatenate([[theta_TO], c_eq * rng.uniform(0.5, 1.2, N_GAS)])

        K_arr = K + 0.15 * np.sin(2 * np.pi * np.linspace(0, TW, 20) / TW)
        K_arr = K_arr.clip(0.3, 1.4)
        K_mean = K_arr.mean()

        sol = solve_ivp(
            lambda t, x: fast_rhs_np(x, K_mean, Ta),
            [0, T_window], x0,
            method="RK45", t_eval=[0, T_window],
            rtol=1e-6, atol=1e-8,
        )
        if sol.success and sol.y.shape[1] == 2:
            deltas.append(np.abs(sol.y[:, -1] - sol.y[:, 0]))

    deltas = np.array(deltas)
    scale = np.percentile(deltas, percentile, axis=0) / T_window
    scale = np.maximum(scale, scale.max() * 1e-3)
    return scale.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Slow state: DP by Arrhenius quadrature, never by a network
# ═══════════════════════════════════════════════════════════════════════════
def update_DP(DP_cur: float, theta_HS_ETC_mean_C: float, dt_min: float) -> float:
    """DP(t + dt) = 1 / (1/DP(t) + k0 * V(theta_HS_ETC) * dt).

    Monotone by construction, so no loss penalty is needed. `theta_HS` must be
    the ETC-corrected hot-spot: passing `hot_spot_np` instead underestimates
    aging by ~3.6x at K > 1.0.
    """
    T_HS_K = float(np.clip(theta_HS_ETC_mean_C, 40.0, 200.0)) + 273.15
    V = np.exp(B_aging * (1.0 / T_ref - 1.0 / T_HS_K))
    inv_DP_new = 1.0 / DP_cur + k0_aging * V * dt_min
    return max(1.0 / inv_DP_new, 1.0)


def dp_at_reference_life() -> float:
    """Self-check used by the notebook: should return 200.0 (IEEE C57.91)."""
    return 1.0 / (1 / DP0 + k0_aging * T_life_min)


def arrhenius_V(theta_HS_C, E_a: float = 1.0):
    """Arrhenius acceleration factor at a hot-spot temperature in degC."""
    return np.exp(B_aging * E_a * (1.0 / T_ref - 1.0 / (np.asarray(theta_HS_C) + 273.15)))


__all__ = [
    "tau_oil", "DTheta_oil_R", "DTheta_HS_R", "R_load", "n_exp", "m_exp",
    "theta_a0", "alpha_Cu", "T_HS_ref_C", "K_PD_onset", "PD_gain",
    "B_aging", "T_ref", "T_life_min", "k0_aging", "DP0", "DP_EOL",
    "GAS_NAMES", "N_GAS", "k_gen", "k_dis", "E_act", "IEC_ATTENTION",
    "STATE_DIM_FAST", "STATE_NAMES_FAST", "TW", "N_SENSORS",
    "hot_spot_np", "hot_spot_ETC_np", "pd_factor_np",
    "fast_rhs_np", "fast_rhs_torch", "compute_rhs_scale_physics",
    "compute_deriv_scale", "compute_theta_HS_torch",
    "update_DP", "dp_at_reference_life", "arrhenius_V",
]
