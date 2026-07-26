"""The three steady-state top-oil formulas, plus the true fixed point.

PHASE 1 — FAITHFUL PORT. All three formulas are ported side by side under the
names the audit gave them (results/step3_steady_state.md §3.2), so that the
inconsistency is visible in code rather than hidden across three notebook cells:

    formula_A  = `theta_TO_ss`      builds EVERY initial condition
                                    (n12 cell 0 L904 / cell 2 L1510)
    formula_B  = `theta_TO_ss_ETC`  the reference the rollout bias is scored
                                    against (n12 cell 2 L1515)
    formula_C  = `model._theta_ss`  the analytic attractor the network is built
                                    on (n12 cell 0 L473, torch version lives in
                                    cod/models/cod.py; numpy mirror here)

`theta_TO_ss` is defined twice in n12 (cell 0 L904, cell 2 L1510). The bodies
are byte-identical — `theta_a + DTheta_oil_R * K**(2*n_exp)` — and only the
docstrings differ ("Steady-state top-oil theo IEC 60076-7" vs the same plus
"giu nguyen cho training consistency" / "kept unchanged for training
consistency"). The audit's §8.1 table marks it as differing because it compares
full source text including docstrings. Either definition gives the same numbers,
so there is no last-definition-wins hazard here; ported once as `formula_A`.

KNOWN DEFECT (audit M-6): formula A is the least accurate of the three and it is
the one generating every IC. Against the true fixed point of `fast_rhs_np` it is
off by -18.25 degC at K = 1.3, theta_a = 30 degC, and -23.80 degC at K = 1.3,
theta_a = 45 degC. `sample_consistent_ic` feeds that theta_TO into
`hot_spot_ETC_np` to build c_eq = k_gen*V_arr/k_dis, and V_arr is exponential in
temperature, so the error propagates multiplicatively into the gas ICs. The
docstring in the source calls these ICs "ETC-consistent"; they are not.
Phase 2 fix 1 unifies everything on `true_fixed_point`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from cod.data.physics import (
    B_aging,
    DTheta_HS_R,
    DTheta_oil_R,
    E_act,
    R_load,
    T_HS_ref_C,
    T_ref,
    alpha_Cu,
    fast_rhs_np,
    hot_spot_ETC_np,
    k_dis,
    k_gen,
    m_exp,
    n_exp,
)


def formula_A(K, theta_a):
    """IEC 60076-7 steady-state top-oil, no ETC correction.

    theta_TO_ss = theta_a + DTheta_oil_R * K**(2*n_exp)

    Source: n12 cell 0 L904 (`theta_TO_ss`). Used by `sample_consistent_ic`,
    therefore by every training and test initial condition.
    """
    return theta_a + DTheta_oil_R * (K ** (2 * n_exp))


def formula_B(K, theta_a):
    """ETC-corrected steady state, hot-spot estimated with BOTH rise terms.

    Source: n12 cell 2 L1515 (`theta_TO_ss_ETC`). Used by `gas_ic_from_ss`, the
    CHI lifetime rollouts and the bias diagnostic. Note the theta_HS0 estimate
    adds DTheta_oil_R*fac0 and DTheta_HS_R*fac0 using the SAME exponent m_exp
    for both, which is what makes B differ from C away from K = 1.
    """
    fac0 = ((1 + K ** 2 * R_load) / (1 + R_load)) ** m_exp
    theta_HS0 = theta_a + DTheta_oil_R * fac0 + DTheta_HS_R * fac0
    Rf = np.clip(1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C), 0.8, 1.5)
    R_eff = R_load * Rf
    fac_n = ((1 + K ** 2 * R_eff) / (1 + R_load)) ** n_exp
    return theta_a + DTheta_oil_R * fac_n


def formula_C(K, theta_a):
    """The model's own analytic attractor, in numpy.

    Source: n12 cell 0 L473 (`PIDeepONet_v24._theta_ss`). Differs from B only in
    the theta_HS0 estimate, which uses n_exp for the oil-rise term and m_exp for
    the hot-spot gradient. At K = 1 the load factor ((1+K^2 R)/(1+R)) equals 1,
    so every exponent gives 1 and B and C coincide exactly; they diverge as K
    moves away from 1 in either direction (audit step3 §3.4).

    Kept in numpy here for comparison and for Phase 2; the live torch version
    used by the network is `CODOperator._theta_ss`, which must stay bit-identical
    to the checkpoint's arithmetic.
    """
    fac_m = ((1.0 + K ** 2 * R_load) / (1.0 + R_load)) ** m_exp
    fac_n = ((1.0 + K ** 2 * R_load) / (1.0 + R_load)) ** n_exp
    theta_HS0 = theta_a + DTheta_oil_R * fac_n + DTheta_HS_R * fac_m
    Rf = np.clip(1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C), 0.8, 1.5)
    return theta_a + DTheta_oil_R * ((1.0 + K ** 2 * R_load * Rf) / (1.0 + R_load)) ** n_exp


def true_fixed_point(K: float, theta_a: float,
                     bracket: tuple[float, float] = (-50.0, 400.0),
                     xtol: float = 1e-10) -> float:
    """The actual fixed point of the data-generating ODE, by root finding.

    Solves d(theta_TO)/dt = 0 for the thermal component of `fast_rhs_np`. The
    gas states do not enter the thermal equation, so a scalar root is enough and
    any gas values may be passed in the dummy state vector.

    This is the quantity formulas A, B and C all approximate, and the one Phase 2
    unifies on. Values here reproduce the TRUE column of audit step3 §3.3.
    """
    dummy_gas = np.array([50.0, 15.0, 80.0, 300.0, 800.0])

    def residual(theta_TO: float) -> float:
        x = np.concatenate([[theta_TO], dummy_gas])
        return float(fast_rhs_np(x, K, theta_a)[0])

    lo, hi = bracket
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"true_fixed_point: no sign change on {bracket} for K={K}, "
            f"theta_a={theta_a} (f={f_lo:.4g}, {f_hi:.4g}). Widen the bracket."
        )
    return float(brentq(residual, lo, hi, xtol=xtol))


def true_fixed_point_torch(K_t, Ta_t, n_iter: int = 20,
                           R_load_v=None, n_exp_v=None, m_exp_v=None,
                           DTheta_oil_R_v=None, DTheta_HS_R_v=None,
                           alpha_Cu_v=None, T_HS_ref_C_v=None):
    """Differentiable, vectorised true fixed point — Phase 2 fix 1.

    `true_fixed_point` uses brentq: scalar, slow and not differentiable, so it
    cannot go inside a torch forward pass evaluated on (B, n_sensors) tensors.

    The fixed-point equation is

        theta = theta_a + DTheta_oil_R * ((1 + K^2 R_eff(theta)) / (1 + R))^n_exp

    and the right-hand side is a contraction in theta, because the only dependence
    is through `Rf = 1 + alpha_Cu * (theta_HS - T_HS_ref_C)` with
    alpha_Cu = 3.93e-3. The measured contraction factor is about 0.34 per
    iteration, so 20 steps from the formula-C estimate reach ~1e-9 degC in float64
    and ~7e-5 degC in float32 (the float32 floor at these magnitudes), against
    brentq over the whole (K, theta_a) box — audit_port/scripts/06_check_fix1.py.

    7e-5 degC is five orders of magnitude below formula A's -18.25 degC error at
    K = 1.3, theta_a = 30, which is the error this fix exists to remove.

    The buffer overrides let `CODOperator` pass its own registered buffers, so the
    model's attractor is computed from the same constants as the rest of it.
    """
    R = R_load if R_load_v is None else R_load_v
    ne = n_exp if n_exp_v is None else n_exp_v
    me = m_exp if m_exp_v is None else m_exp_v
    Do = DTheta_oil_R if DTheta_oil_R_v is None else DTheta_oil_R_v
    Dhs = DTheta_HS_R if DTheta_HS_R_v is None else DTheta_HS_R_v
    ac = alpha_Cu if alpha_Cu_v is None else alpha_Cu_v
    Tr = T_HS_ref_C if T_HS_ref_C_v is None else T_HS_ref_C_v

    load = (1.0 + K_t ** 2 * R) / (1.0 + R)
    theta = Ta_t + Do * load ** ne     # uncorrected IEC start; converges regardless

    for _ in range(n_iter):
        theta_HS0 = theta + Dhs * ((1.0 + K_t ** 2 * R) / (1.0 + R)) ** me
        Rf_hs = (1.0 + ac * (theta_HS0 - Tr)).clamp(0.8, 1.5)
        theta_HS = theta + Dhs * ((1.0 + K_t ** 2 * R * Rf_hs) / (1.0 + R)) ** me
        Rf_n = (1.0 + ac * (theta_HS - Tr)).clamp(0.8, 1.5)
        theta = Ta_t + Do * ((1.0 + K_t ** 2 * R * Rf_n) / (1.0 + R)) ** ne

    return theta


def true_fixed_point_np(K, theta_a, n_iter: int = 20) -> np.ndarray:
    """Numpy mirror of `true_fixed_point_torch`, for IC generation.

    Same iteration, so IC generation and the model's attractor agree exactly
    rather than to within a solver tolerance. Agrees with brentq to <1e-6 degC.
    """
    K = np.asarray(K, dtype=np.float64)
    theta_a = np.asarray(theta_a, dtype=np.float64)
    load = (1.0 + K ** 2 * R_load) / (1.0 + R_load)
    theta = theta_a + DTheta_oil_R * load ** n_exp
    for _ in range(n_iter):
        theta_HS0 = theta + DTheta_HS_R * load ** m_exp
        Rf_hs = np.clip(1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C), 0.8, 1.5)
        theta_HS = theta + DTheta_HS_R * ((1.0 + K ** 2 * R_load * Rf_hs)
                                          / (1.0 + R_load)) ** m_exp
        Rf_n = np.clip(1.0 + alpha_Cu * (theta_HS - T_HS_ref_C), 0.8, 1.5)
        theta = theta_a + DTheta_oil_R * ((1.0 + K ** 2 * R_load * Rf_n)
                                          / (1.0 + R_load)) ** n_exp
    return theta if theta.ndim else float(theta)


def true_fixed_point_grid(K_arr, theta_a_arr) -> np.ndarray:
    """Vectorised `true_fixed_point` over matching arrays (elementwise)."""
    K_arr = np.atleast_1d(np.asarray(K_arr, dtype=float))
    theta_a_arr = np.atleast_1d(np.asarray(theta_a_arr, dtype=float))
    K_b, Ta_b = np.broadcast_arrays(K_arr, theta_a_arr)
    out = np.empty(K_b.shape, dtype=float)
    for idx in np.ndindex(K_b.shape):
        out[idx] = true_fixed_point(float(K_b[idx]), float(Ta_b[idx]))
    return out


def gas_ic_from_ss(K, theta_a, steady_state=None) -> np.ndarray:
    """Gas equilibrium at (K, theta_a): c_eq = k_gen * V_arr / k_dis.

    Source: n12 cell 2 L1525 (`gas_ic_from_ss`), which calls `theta_TO_ss_ETC`
    (= formula B).

    PHASE 2 FIX 1: the default is now `true_fixed_point_np`. Pass
    `steady_state=formula_B` to reproduce v57.
    """
    if steady_state is None:
        steady_state = true_fixed_point_np
    theta_TO_s = steady_state(K, theta_a)
    theta_HS_s = hot_spot_ETC_np(theta_TO_s, K)
    T_HS_K = np.clip(theta_HS_s + 273.15, 313.15, 573.15)
    V_arr = np.exp(B_aging * E_act * (1.0 / T_ref - 1.0 / T_HS_K))
    c_eq = k_gen * V_arr / k_dis
    return c_eq.astype(np.float32)


FORMULAS = {"A": formula_A, "B": formula_B, "C": formula_C,
            "TRUE": true_fixed_point, "TRUE_ITER": true_fixed_point_np}


def compare(K_values=(0.5, 0.8, 1.0, 1.2, 1.3),
            theta_a_values=(15.0, 30.0, 45.0)) -> str:
    """Reproduce audit step3 §3.3 as text. Used by the Phase 2 fix-1 report."""
    lines = [
        "| K | theta_a | TRUE | A | B | C | A-TRUE | B-TRUE | C-TRUE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for theta_a in theta_a_values:
        for K in K_values:
            t = true_fixed_point(K, theta_a)
            a, b, c = formula_A(K, theta_a), formula_B(K, theta_a), formula_C(K, theta_a)
            lines.append(
                f"| {K} | {theta_a:g} | {t:.2f} | {a:.2f} | {b:.2f} | {c:.2f} | "
                f"{a - t:+.2f} | {b - t:+.2f} | {c - t:+.2f} |"
            )
    return "\n".join(lines)


__all__ = [
    "formula_A", "formula_B", "formula_C",
    "true_fixed_point", "true_fixed_point_np", "true_fixed_point_torch",
    "true_fixed_point_grid", "gas_ic_from_ss", "FORMULAS", "compare",
]
