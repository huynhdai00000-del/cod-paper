"""Physics losses.

PHASE 1 — FAITHFUL PORT. Each function is defined once. Where the notebook
defined a name three times, the surviving (last) definition is ported and the
discarded ones are described here.

    make_log_collocation   n12 cell 0 L719, cell 2 L1119, cell 2 L1329
    ode_physics_loss       n12 cell 0 L738, cell 2 L1134, cell 2 L1344
    compute_chi            n12 cell 0 L662, cell 2 L1090, cell 2 L1300
    chi_monotonicity_loss  n12 cell 0 L773, cell 2 L1169, cell 2 L1379
    chi_rate_loss_v10      n12 cell 0 L825, cell 2 L1219, cell 2 L1429

`chi_weight_net` / `AdaptiveCHIWeights` are deliberately NOT ported: instantiated
at n12 cell 0 L993, referenced only from cell-0 bodies that cell 2 overwrites,
never trained, never executed, and absent from every checkpoint's state_dict.
The cell-0 `chi_rate_loss_v10` would have called it; the surviving cell-2 body
uses fixed weights instead (`[V34] Fixed weights`).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cod.data.physics import (
    B_aging,
    DP0,
    DP_EOL,
    T_ref,
    compute_theta_HS_torch,
    fast_rhs_torch,
    k0_aging,
)
from cod.models.blocks import interp_sensors

# ═══════════════════════════════════════════════════════════════════════════
# CHI limits and weights — n12 cell 0 L611-L615, cell 2 L1086-L1088
# ═══════════════════════════════════════════════════════════════════════════
# 60% of the IEC attention levels, i.e. the action threshold.
FAST_LIM_LO_NP = np.array([60.0, 5.0, 0.0, 2.0, 50.0, 200.0], dtype=np.float32)
FAST_LIM_HI_NP = np.array([140.0, 60.0, 21.0, 120.0, 420.0, 1200.0], dtype=np.float32)
X_NORM_LO_NP = np.array([50.0, 5.0, 0.0, 2.0, 50.0, 200.0], dtype=np.float32)
X_NORM_HI_NP = np.array([120.0, 100.0, 35.0, 200.0, 700.0, 2000.0], dtype=np.float32)

# CHI_W[0] = 0: theta_HS is excluded from the weight target because it
# oscillates with the seasonal load and would make CHI non-monotone.
CHI_W_NP = np.array([0.00, 0.10, 0.175, 0.10, 0.075, 0.05], dtype=np.float32)
W_DP = 0.50

# CHI falls from 1 to 0 over 25 years at T_ref.
k_chi_max = 1.0 / (25 * 365 * 1440)

# The physics-loss state clamp — n12 cell 2 L1148-L1149. 23.2% of the training
# ICs already exceed `_hi`, so nearly a quarter of the training set is truncated
# before the residual is formed (audit M-9, §8.4).
STATE_CLAMP_LO_NP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
STATE_CLAMP_HI_NP = np.array([200.0, 500.0, 200.0, 1000.0, 3000.0, 8000.0],
                             dtype=np.float32)

_CONST_CACHE: dict[str, torch.Tensor] = {}


def _c(name: str, arr: np.ndarray, device, dtype=torch.float32) -> torch.Tensor:
    key = f"{name}:{device}:{dtype}"
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = torch.tensor(arr, dtype=dtype, device=device)
    return _CONST_CACHE[key]


def FAST_LIM_LO(device):
    return _c("fll", FAST_LIM_LO_NP, device)


def FAST_LIM_HI(device):
    return _c("flh", FAST_LIM_HI_NP, device)


def X_NORM_LO(device):
    return _c("xnl", X_NORM_LO_NP, device)


def X_NORM_HI(device):
    return _c("xnh", X_NORM_HI_NP, device)


def CHI_W(device):
    return _c("chiw", CHI_W_NP, device)


# ═══════════════════════════════════════════════════════════════════════════
# Collocation
# ═══════════════════════════════════════════════════════════════════════════
def make_log_collocation(T: float, n_col: int, device) -> torch.Tensor:
    """Asymmetric two-sided log spacing: 75% forward, 25% backward.

    Dense near t = 0 AND near t = T. Earlier versions were dense only at t ~ 0,
    so the model rarely saw t -> T and `delta_TO` spiked at that boundary; the
    `peak_then_drop` profile family alone was not enough to fix it.

    All three source definitions have identical bodies (cell 0's docstring says
    `[V55-A]`, cell 2's says `[V56]`), so there is no last-definition-wins hazard
    here — PORT_LOG J-6.
    """
    t_min = T * 0.005
    n_fwd = (n_col * 3) // 4
    n_bwd = n_col - n_fwd
    t_fwd = torch.exp(torch.linspace(math.log(t_min), math.log(T), n_fwd, device=device))
    t_bwd = T - torch.exp(
        torch.linspace(math.log(t_min), math.log(T * 0.5), n_bwd, device=device))
    t_bwd = torch.flip(t_bwd, [0])
    t_all = torch.cat([t_fwd, t_bwd])
    t_all, _ = torch.sort(torch.unique(t_all.clamp(min=t_min, max=T)))
    if t_all.shape[0] < n_col:
        t_extra = torch.linspace(t_min, T, n_col - t_all.shape[0] + 2, device=device)[1:-1]
        t_all = torch.sort(torch.unique(torch.cat([t_all, t_extra])))[0]
    return t_all[:n_col]


# ═══════════════════════════════════════════════════════════════════════════
# The ODE residual loss
# ═══════════════════════════════════════════════════════════════════════════
CAUSAL_WEIGHT_FLOOR = 1e-8


def causal_weights(r2m: torch.Tensor, eps_causal: float,
                   log_space: bool = True,
                   floor: float = CAUSAL_WEIGHT_FLOOR
                   ) -> tuple[torch.Tensor, float]:
    """Causal weighting over collocation chunks: w_j = exp(-eps * sum_{i<j} r2m_i).

    PHASE 2 FIX 3 (audit B-1), the defect that invalidated the Mono Fair
    comparison. In linear float32, `exp(-x)` reaches **exactly 0.0** at x around
    88. Mono Fair finished training with `wm = 0.000` while COD had `wm = 0.988`,
    so the monolithic baseline was effectively trained on the early part of the
    window only, and the two arms of the comparison were not optimising the same
    objective.

    `log_space=True` (default) computes the weights via `log w = -eps * cum`,
    clamps that to `log(floor)`, and exponentiates. The result is mathematically
    the same function wherever it does not underflow, and bottoms out at `floor`
    instead of at zero, so later chunks always retain some gradient. `floor=1e-8`
    is small enough not to distort the weighting and large enough to survive
    float32.

    Clamping in log space rather than clamping the exponentiated value matters:
    `exp(-100)` has already lost all information by the time you could clamp it,
    so `max(exp(-eps*cum), floor)` would give every deep chunk the identical
    weight `floor`. Clamping `-eps*cum` first preserves the ordering right up to
    the floor.

    `log_space=False, floor=0.0` restores the v57 arithmetic for the before/after
    comparison in PHASE2_EFFECTS.md.

    Returning `w.min()` is what lets the harness detect an underflow instead of
    leaving it to be found in a log file.
    """
    cum = torch.cumsum(r2m, dim=1) - r2m
    if log_space:
        log_w = (-eps_causal * cum).clamp(min=math.log(floor))
        w = torch.exp(log_w).detach()
    else:
        w = torch.exp(-eps_causal * cum).detach()
        if floor > 0.0:
            w = w.clamp(min=floor)
    return w, float(w.min().item())


def ode_physics_loss(model, x0, sensors, n_col: int = 40, n_chunks: int = 5,
                     eps_causal: float = 0.01, return_per_state: bool = False,
                     diagnostics: dict | None = None,
                     causal_log_space: bool = True,
                     causal_floor: float = CAUSAL_WEIGHT_FLOOR):
    """Causally weighted mean squared ODE residual.

    LAST DEFINITION WINS. Three definitions exist in n12; the one that trained
    `transformer_pideepOnet_v57.pt` is cell 2's, and the difference that matters
    is a single line:

        cell 0 L758   residual = (dxdt - f_rhs) / deriv_scale_t   [V18 DERIV_SCALE]
        cell 2 L1154  residual = (dxdt - f_rhs)                   [V34] raw residual
        cell 2 L1364  residual = (dxdt - f_rhs)                   (identical to L1154)

    The two cell-2 copies are byte-identical to each other. Cell 2 executes after
    cell 0, so the raw residual is live and `DERIV_SCALE` is dead. Ported the raw
    residual (PORT_LOG J-4).

    Consequence: because nothing on this path consumes `RHS_SCALE`, the
    double-`pd_factor` defect in `compute_rhs_scale_physics` has zero effect on
    v57's results — which is why Phase 2 fix 2 is a hygiene fix, not a
    correctness fix, for this checkpoint.

    Two more things happen here that the paper should state:

      * `x_pred_c = x_pred_raw.detach().clamp(_lo, _hi)` — the RHS is evaluated at
        a **clamped, detached** state, so the residual compares an unclamped
        derivative against an RHS taken at a different point whenever the clamp
        binds. 23.2% of the training ICs already sit above `_hi`.
      * the five gas residual terms are computed and summed but contribute exactly
        zero gradient, because `CODOperator.forward` detaches the thermal grid
        (audit M-1). That is intended cascade behaviour, not a defect.

    `diagnostics`, if given a dict, is filled with clamp-hit fractions. It is
    computed under `no_grad` from the same tensors and cannot alter the loss.
    """
    device = x0.device
    B = x0.shape[0]
    n = model.state_dim
    T = model.T
    ns = model.n_sensors

    tc_base = make_log_collocation(T, n_col, device)
    tc = tc_base.unsqueeze(0).unsqueeze(-1).expand(B, n_col, 1)
    x0_exp = x0.unsqueeze(1).expand(B, n_col, n).reshape(B * n_col, n)
    s_exp = sensors.unsqueeze(1).expand(B, n_col, 2 * ns).reshape(B * n_col, 2 * ns)
    tf = tc.reshape(B * n_col, 1).detach().clone().requires_grad_(True)

    x_pred_raw = model(x0_exp, s_exp, tf)
    dxdt = torch.cat([
        torch.autograd.grad(x_pred_raw[:, i].sum(), tf,
                            create_graph=True, retain_graph=True)[0]
        for i in range(n)
    ], dim=1)

    _lo = _c("clamp_lo", STATE_CLAMP_LO_NP, device)
    _hi = _c("clamp_hi", STATE_CLAMP_HI_NP, device)
    x_pred_c = x_pred_raw.detach().clamp(_lo, _hi)
    u_t = interp_sensors(sensors, tf, T, ns)
    f_rhs = fast_rhs_torch(x_pred_c, u_t)

    residual = (dxdt - f_rhs)          # [V34] raw residual — DERIV_SCALE dropped
    r2 = residual ** 2

    n_col_use = (n_col // n_chunks) * n_chunks
    r2 = r2.reshape(B, n_col, n)[:, :n_col_use, :]
    pch = n_col_use // n_chunks
    r2c = r2.reshape(B, n_chunks, pch, n).mean(dim=2)
    r2m = r2c.mean(dim=-1)

    w, wm = causal_weights(r2m, eps_causal, log_space=causal_log_space,
                           floor=causal_floor)

    if diagnostics is not None:
        diagnostics.update(_clamp_diagnostics(x_pred_raw, x_pred_c, u_t, _lo, _hi))
        diagnostics["causal_weight_min"] = wm

    if return_per_state:
        L_states = [(w * r2c[:, :, s]).sum() / (w.sum() + 1e-20) for s in range(n)]
        return L_states, wm
    return (w * r2m).sum() / (w.sum() + 1e-20)


@torch.no_grad()
def _clamp_diagnostics(x_pred_raw, x_pred_c, u_t, lo, hi) -> dict:
    """Fraction of samples hitting each clamp on the physics-loss path.

    Recomputes the clamped quantities from the same inputs rather than
    instrumenting `fast_rhs_torch`, so the loss path stays byte-identical to the
    source. Audit §8.4 lists these as the clamps that can hide behaviour.
    """
    from cod.data.physics import (
        DTheta_HS_R, R_load, T_HS_ref_C, alpha_Cu, m_exp,
    )

    n_tot = x_pred_raw.shape[0]
    out = {
        "clamp_frac_state_hi": float((x_pred_raw.detach() > hi).any(dim=-1).float().mean()),
        "clamp_frac_state_lo": float((x_pred_raw.detach() < lo).any(dim=-1).float().mean()),
    }

    theta_TO = x_pred_c[..., 0:1]
    K = u_t[..., 0:1]
    fac_m0 = ((1.0 + K ** 2 * R_load) / (1.0 + R_load)) ** m_exp
    theta_HS0 = theta_TO + DTheta_HS_R * fac_m0
    Rf1 = 1.0 + alpha_Cu * (theta_HS0 - T_HS_ref_C)
    out["clamp_frac_Rf_etc"] = float(((Rf1 < 0.8) | (Rf1 > 1.5)).float().mean())

    fac_m1 = ((1.0 + K ** 2 * R_load * Rf1.clamp(0.8, 1.5)) / (1.0 + R_load)) ** m_exp
    theta_HS = theta_TO + DTheta_HS_R * fac_m1
    T_HS_K = theta_HS + 273.15
    out["clamp_frac_T_HS_min"] = float((T_HS_K < 313.15).float().mean())

    from cod.data.physics import E_act
    E_act_t = _c("eact", E_act.astype(np.float32), x_pred_c.device)
    V_arr = torch.exp(B_aging * E_act_t * (1.0 / T_ref - 1.0 / T_HS_K.clamp(min=313.15)))
    out["clamp_frac_V_arr_max"] = float((V_arr > 1e4).any(dim=-1).float().mean())
    out["n_collocation_samples"] = n_tot
    return out


def ode_physics_loss_shared(model, predict_fn, x0, sensors, n_col: int = 60,
                            n_chunks: int = 5, eps_causal: float = 0.01,
                            diagnostics: dict | None = None,
                            causal_log_space: bool = True,
                            causal_floor: float = CAUSAL_WEIGHT_FLOOR):
    """The residual loss used by the SHARED trainer, which is not the same loss.

    This is the inline body of `train_physics` (n15 cell 2 L334, n00 cell 4 L167),
    which trained every monolithic baseline and every capacity-sweep checkpoint —
    including `sweep_cod_p*.pt`. `transformer_pideepOnet_v57.pt` was trained by
    `train_v34` (n12 cell 1) with `ode_physics_loss` above. Three differences,
    none of them cosmetic:

    1. **Collocation.** Here it is one-sided forward log spacing,
       `exp(linspace(log(0.005 T), log T, n_col))`. `ode_physics_loss` uses
       two-sided 75/25 spacing that is also dense near t = T. So the sweep models
       saw far fewer collocation points near the end of the window than v57 did.

    2. **The RHS state clamp.** Here it is a scalar `.clamp(0, 500)` applied to
       every state alike. `ode_physics_loss` uses the per-state vector
       `_hi = [200, 500, 200, 1000, 3000, 8000]`. A scalar ceiling of 500 truncates
       CO (typical ~1e2-1e3 ppm) and CO2 (~1e3 ppm) hard, so the shared trainer's
       gas residuals are formed at a clamped state far more often.

    3. **The adaptive weight target.** Here `Lv[s] = rc[:,:,s].mean()`, the plain
       mean over batch and chunks. In `train_v34` it is the causally *weighted*
       per-state loss. And `lam[1:]` is never floored here.

    Recorded because it bears on the fairness of gate 2: the capacity sweep
    compares COD against Mono Fair under this loop, which is internally
    consistent, but neither arm of that sweep was trained the way the headline
    COD model was.
    """
    device = x0.device
    B = x0.shape[0]
    n = model.state_dim
    T = model.T
    ns = model.n_sensors

    tc = torch.exp(torch.linspace(math.log(T * 0.005), math.log(T), n_col,
                                  device=device))
    tc = tc.unsqueeze(0).unsqueeze(-1).expand(B, n_col, 1)
    x0e = x0.unsqueeze(1).expand(B, n_col, n).reshape(B * n_col, n)
    se = sensors.unsqueeze(1).expand(B, n_col, 2 * ns).reshape(B * n_col, 2 * ns)
    tf = tc.reshape(B * n_col, 1).detach().requires_grad_(True)

    xp = predict_fn(model, x0e, se, tf)
    dxdt = torch.cat([
        torch.autograd.grad(xp[:, i].sum(), tf, create_graph=True, retain_graph=True)[0]
        for i in range(n)
    ], dim=1)

    u_t = interp_sensors(sensors, tf, T, ns)
    xp_clamped = xp.detach().clamp(0, 500)          # scalar clamp, see (2) above
    f_rhs = fast_rhs_torch(xp_clamped, u_t)

    res = (dxdt - f_rhs) ** 2
    n_col_use = (n_col // n_chunks) * n_chunks
    res = res.reshape(B, n_col, n)[:, :n_col_use, :]
    pch = n_col_use // n_chunks
    r2c = res.reshape(B, n_chunks, pch, n).mean(dim=2)
    r2m = r2c.mean(dim=-1)

    w, wm = causal_weights(r2m, eps_causal, log_space=causal_log_space,
                           floor=causal_floor)

    if diagnostics is not None:
        diagnostics["causal_weight_min"] = wm
        diagnostics["clamp_frac_state_scalar_500"] = float(
            (xp.detach() > 500).any(dim=-1).float().mean())
        diagnostics["clamp_frac_state_lo"] = float(
            (xp.detach() < 0).any(dim=-1).float().mean())
    return r2c, w, wm


# ═══════════════════════════════════════════════════════════════════════════
# CHI losses
# ═══════════════════════════════════════════════════════════════════════════
def compute_chi(x_fast, DP_val):
    """Composite health index with a dynamic DP weight.

    LAST DEFINITION WINS. Cell 0's version (L662) reads module globals `CHI_W` and
    `W_DP` that cell 0 never defines, so it is unreachable as written; the two
    cell-2 copies are identical and define both immediately above. Ported the
    cell-2 body (PORT_LOG J-5).

    `w_dp` rises from 0.50 to 1.00 as DP approaches end of life, which forces
    CHI -> 0 at EOL regardless of the gas sub-indices.
    """
    device = x_fast.device
    lo, hi, cw = FAST_LIM_LO(device), FAST_LIM_HI(device), CHI_W(device)
    chi_s = 1.0 - torch.clamp((x_fast - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    chi_g = (chi_s * cw).sum(dim=-1) / cw.sum()
    if isinstance(DP_val, (int, float)):
        chi_DP = torch.clamp(
            torch.tensor((DP_val - DP_EOL) / (DP0 - DP_EOL),
                         dtype=torch.float32, device=device), 0.0, 1.0)
    else:
        chi_DP = torch.clamp((DP_val - DP_EOL) / (DP0 - DP_EOL), 0.0, 1.0)
    w_dp = W_DP + (1.0 - chi_DP) * (1.0 - W_DP)
    return w_dp * chi_DP + (1.0 - w_dp) * chi_g


def chi_monotonicity_loss(model, x0, sensors, n_t: int = 8, DP_cur=None,
                          rng: np.random.RandomState | None = None):
    """Penalise CHI increasing within a window, plus a cross-window DP penalty.

    Physics: d(1/DP)/dt = k0 * V(T_HS_ETC) >= 0, so DP only decreases, so the DP
    sub-index only decreases, so CHI trends down.

    LAST DEFINITION WINS. Cell 0 L773 calls
    `compute_chi(..., sensors=sensors, weight_net=chi_weight_net)`, which would
    raise `TypeError` against the surviving two-argument `compute_chi`; the
    cell-2 bodies call `compute_chi(xp[:, q, :], DP_cur)` and are what runs
    (PORT_LOG J-7).

    `rng` is threaded through instead of calling the global `np.random`, so a run
    is reproducible from its seed. Defaults to the global stream, matching the
    source.
    """
    device = x0.device
    B = x0.shape[0]
    Q = n_t
    ns = model.n_sensors
    draw = rng if rng is not None else np.random

    if DP_cur is None:
        DP_cur = float(draw.uniform(DP_EOL + 10, DP0))

    t_q = torch.linspace(model.T * 0.05, model.T, Q, device=device)
    x0e = x0.unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
    se = sensors.unsqueeze(1).expand(B, Q, 2 * ns).reshape(B * Q, 2 * ns)
    tq = t_q.unsqueeze(0).unsqueeze(-1).expand(B, Q, 1).reshape(B * Q, 1)
    xp = model(x0e, se, tq).reshape(B, Q, -1)

    chi = torch.stack([compute_chi(xp[:, q, :], DP_cur) for q in range(Q)], dim=1)
    viol = torch.clamp(chi[:, 1:] - chi[:, :-1], min=0.0)

    # Cross-window Arrhenius penalty: DP must not increase over the window.
    x_mid = xp[:, Q // 2, :]
    K_mean = sensors[:, :ns].mean(dim=-1).clamp(0.3, 1.5)
    theta_HS_mid = compute_theta_HS_torch(x_mid, K_mean)
    T_HS_K_mid = (theta_HS_mid + 273.15).clamp(min=313.15)
    V_mid = torch.exp(B_aging * (1.0 / T_ref - 1.0 / T_HS_K_mid)).clamp(max=1e4)

    inv_DP_end = 1.0 / float(DP_cur) + k0_aging * V_mid * model.T
    DP_end_pred = (1.0 / inv_DP_end).clamp(1.0, float(DP0))
    cross_viol = torch.clamp(DP_end_pred - float(DP_cur), min=0.0)

    return viol.mean() + 0.5 * cross_viol.mean()


def chi_rate_loss_v10(model, x0, sensors, n_t: int = 6, DP_cur=None,
                      rng: np.random.RandomState | None = None):
    """Penalise CHI falling faster than Arrhenius allows.

    Constraint: -dCHI/dt <= k_chi_max * V_arr(T_HS_ETC).
    violation = clamp(-dCHI/dt - k_chi_max * V_arr, min=0).

    In normal operation the constraint is slack by a factor of 6-8, so it only
    fires when the model predicts an unphysically fast CHI drop, typically early
    in training.

    LAST DEFINITION WINS. Cell 0's version calls `chi_weight_net(...)` for the
    fusion weights; the surviving cell-2 body replaces that with the fixed
    `CHI_W` / `W_DP` (`[V34] Fixed weights`). This is why `AdaptiveCHIWeights` is
    dead code and is not ported.

    The weights are taken under `no_grad`: gradient is only needed through
    `x_pred -> chi_states`, and `create_graph=True` through a weight network would
    build an expensive and unstable second-order graph.
    """
    device = x0.device
    B = x0.shape[0]
    ns = model.n_sensors
    T = model.T
    draw = rng if rng is not None else np.random

    if DP_cur is None:
        DP_cur = float(draw.uniform(DP_EOL + 10, DP0))

    t_q = torch.linspace(T * 0.05, T * 0.95, n_t, device=device)
    tq = t_q.unsqueeze(0).unsqueeze(-1).expand(B, n_t, 1).reshape(B * n_t, 1)
    tq = tq.detach().requires_grad_(True)

    x0e = x0.unsqueeze(1).expand(B, n_t, -1).reshape(B * n_t, -1)
    se = sensors.unsqueeze(1).expand(B, n_t, 2 * ns).reshape(B * n_t, 2 * ns)
    x_pred = model(x0e, se, tq)

    chi_DP_d = torch.tensor((DP_cur - DP_EOL) / (DP0 - DP_EOL), device=device,
                            dtype=torch.float32).expand(B * n_t, 1)
    with torch.no_grad():
        W_d = torch.cat([CHI_W(device).unsqueeze(0).expand(B * n_t, -1),
                         torch.full((B * n_t, 1), W_DP, device=device)], dim=-1)

    K_mean = se[:, :ns].mean(dim=-1).clamp(0.3, 1.5)
    theta_HS_pred = compute_theta_HS_torch(x_pred, K_mean)
    x_for_chi = torch.cat([theta_HS_pred.unsqueeze(-1), x_pred[:, 1:]], dim=-1)
    lo, hi = FAST_LIM_LO(device), FAST_LIM_HI(device)
    chi_states = 1.0 - torch.clamp((x_for_chi - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    chi_all = torch.cat([chi_states, chi_DP_d], dim=-1)
    chi_t = (chi_all * W_d).sum(dim=-1)

    dchi_dt = torch.autograd.grad(chi_t.sum(), tq, create_graph=True)[0].squeeze(-1)

    T_HS_K = (theta_HS_pred + 273.15).clamp(min=313.15).detach()
    V_arr = torch.exp(B_aging * (1.0 / T_ref - 1.0 / T_HS_K)).clamp(max=1e4)
    violation = torch.clamp(-dchi_dt - k_chi_max * V_arr, min=0.0)
    return violation.mean()


__all__ = [
    "FAST_LIM_LO_NP", "FAST_LIM_HI_NP", "X_NORM_LO_NP", "X_NORM_HI_NP",
    "CHI_W_NP", "W_DP", "k_chi_max",
    "STATE_CLAMP_LO_NP", "STATE_CLAMP_HI_NP",
    "FAST_LIM_LO", "FAST_LIM_HI", "X_NORM_LO", "X_NORM_HI", "CHI_W",
    "make_log_collocation", "causal_weights", "CAUSAL_WEIGHT_FLOOR",
    "ode_physics_loss",
    "ode_physics_loss_shared",
    "compute_chi", "chi_monotonicity_loss", "chi_rate_loss_v10",
]
