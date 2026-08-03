"""Cascaded Operator Decomposition architecture.

PHASE 1 — FAITHFUL PORT of `PIDeepONet_v24` (n12 cell 0 L409; the same class
appears as `PIDeepONet_COD` in n15 cell 2 L202 and n00 cell 6 L272).

The cascade, in one sentence: a neural correction is added to an analytic IEC
60076-7 baseline to predict top-oil temperature, and the five gas concentrations
are then obtained by analytic Arrhenius quadrature driven by that temperature —
so the gases are never predicted by a network.

    theta_TO(t) = H(t; x0, u) + phi(t) * sigma * (b_th . tr_th)(t)
    c_i(t)      = c_i(0) + integral_0^t k_gen_i * V_arr_i(theta_HS(s)) ds
                         - k_dis_i * c_i(0) * t

where H is the exact first-order ODE solution on the sensor grid,
phi(t) = (1 - exp(-t/tau)) / (1 - exp(-T/tau)) enforces theta_TO(0) = x0_TO, and
sigma = softplus(output_scale_raw) + 1e-3.

The `.detach()` on the thermal grid in `forward` is INTENDED cascade behaviour,
not a bug. See the class docstring and `reference/audit/results/step4_gradient_flow.md`.

state_dict keys must not change: they are the keys in
`transformer_pideepOnet_v57.pt` and `sweep_cod_p{4,8,16,32,64}.pt`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import (
    B_aging,
    DTheta_HS_R,
    DTheta_oil_R,
    E_act,
    K_PD_onset,
    LEGACY_V_ARR_CAP,
    PD_gain,
    R_load,
    T_HS_K_MAX,
    T_HS_K_MIN,
    T_HS_ref_C,
    T_ref,
    TW,
    alpha_Cu,
    k_dis,
    k_gen,
    m_exp,
    n_exp,
    tau_oil,
)
from cod.data.steady_state import true_fixed_point_torch
from cod.models.blocks import ModifiedMLP, build_trunk_feats
from cod.models.cascade import gas_integral

# Which analytic attractor the model is built on. "true_fixed_point" is the
# Phase 2 fix-1 default; "formula_C" is what v57 used and is required to
# reproduce the stored checkpoints.
THETA_SS_MODES = ("true_fixed_point", "formula_C")


class CODOperator(nn.Module):
    """COD: analytic thermal baseline + neural correction, then analytic gases.

    Gradient structure (audit M-1, confirmed statically and empirically in 26 of
    26 model versions):

    `forward` detaches the thermal grid before the gas integral, and
    `_gas_integral` contains no `nn.Parameter` in its autograd graph. Therefore
    `d L_gas / d theta = 0` exactly. Back-propagating each state's loss alone
    gives a total gradient norm of 1.255e+01 for theta_TO and **exactly zero**
    for all five gases; removing the `.detach()` makes them 5.65e-02 down to
    6.66e-06 (step4_gradient_flow.md §B).

    This is deliberate: the five gas physics-residual terms are computed,
    weighted by `lam_state[1:]`, summed into the loss, and contribute nothing to
    `loss.backward()`. Gas accuracy is inherited from thermal accuracy plus the
    fixed kinetics. Two consequences worth carrying into the paper:

      * the gas loss terms are 4-7 orders of magnitude below the thermal term, so
        even with gradients flowing they would contribute almost nothing to an
        unweighted sum — which is what `lam_state` exists to compensate for, and
        which supports presenting the decoupling as intended;
      * the gas states degrade first when the load leaves the training range,
        because their only route to accuracy is through theta_TO.

    KEEP the `.detach()`. If sections 6, 7 or Appendix C describe the gas
    residuals as training signal, the text is wrong, not the code.
    """

    def __init__(self, state_dim: int = 6, n_sensors: int = 100, d_h: int = 128,
                 p: int = 64, n_layers: int = 4, n_exp_feats: int = 12,
                 T: float = TW, x_mean=None, x_std=None,
                 theta_ss_mode: str = "true_fixed_point",
                 legacy_V_clamp: bool = False):
        super().__init__()
        if theta_ss_mode not in THETA_SS_MODES:
            raise ValueError(f"theta_ss_mode must be one of {THETA_SS_MODES}")
        # PHASE 2 FIX 1. `theta_ss_mode` is NOT a buffer, so it does not appear in
        # the state_dict and loading a v57 checkpoint will not silently reset it.
        # Pass "formula_C" whenever a v57 checkpoint is loaded.
        self.theta_ss_mode = theta_ss_mode
        # PHASE 2 FIX 6 (DECISIONS N-1). Same treatment: not a buffer, so a v57
        # checkpoint cannot silently switch the kinetics back. Pass True together
        # with theta_ss_mode="formula_C" when reproducing a stored checkpoint.
        self.legacy_V_clamp = legacy_V_clamp
        self.state_dim = state_dim
        self.p_dim = p
        self.n_exp_feats = n_exp_feats
        self.n_gas = state_dim - 1
        self.T = T
        self.n_sensors = n_sensors

        branch_th_in = 1 + 2 * n_sensors          # 201 at n_sensors = 100
        trunk_th_in = 2 * n_exp_feats + 4 + 4     # 32 = 28 + 4 K-history

        self.branch_th = ModifiedMLP(branch_th_in, d_h, p, n_layers)
        self.trunk_th = ModifiedMLP(trunk_th_in, d_h // 2, p, n_layers - 1)
        self.bias_th = nn.Parameter(torch.zeros(1))

        _os_raw = torch.log(torch.exp(torch.tensor([20.0])) - 1 + 1e-6)
        self.output_scale_raw = nn.Parameter(_os_raw)

        k_vals = torch.exp(torch.linspace(torch.log(torch.tensor(0.2)),
                                          torch.log(torch.tensor(5.0)), n_exp_feats))
        self.register_buffer("exp_decay_rates", k_vals)
        # The constructor parameter is `n_exp_feats`, so the module-global n_exp
        # (0.8) is NOT shadowed here and n_exp_buf is correct. The monolithic
        # baselines name the same parameter `n_exp` and so register ne = 12.0
        # instead of 0.8 (PORT_LOG J-8).
        for name, val in [
            ("tau_oil_buf", tau_oil), ("R_load_buf", R_load),
            ("n_exp_buf", n_exp), ("m_exp_buf", m_exp),
            ("DTheta_oil_R_buf", DTheta_oil_R), ("DTheta_HS_R_buf", DTheta_HS_R),
            ("alpha_Cu_buf", alpha_Cu), ("T_HS_ref_C_buf", T_HS_ref_C),
        ]:
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32))
        self.register_buffer("k_gen_buf", torch.tensor(k_gen, dtype=torch.float32))
        self.register_buffer("k_dis_buf", torch.tensor(k_dis, dtype=torch.float32))
        self.register_buffer("E_act_buf", torch.tensor(E_act, dtype=torch.float32))

        if x_mean is not None:
            self.register_buffer("x_mean_TO", torch.tensor([x_mean[0]], dtype=torch.float32))
            self.register_buffer("x_std_TO", torch.tensor([x_std[0]], dtype=torch.float32))

    def n_parameters(self) -> int:
        """Replaces the print in the source constructor (PORT_LOG J-12)."""
        return sum(q.numel() for q in self.parameters())

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def _norm_TO(self, x0_TO):
        if hasattr(self, "x_mean_TO"):
            return (x0_TO - self.x_mean_TO) / (self.x_std_TO + 1e-8)
        return x0_TO

    def _interp_at_t(self, u_sensors, t):
        ns = u_sensors.shape[-1] // 2
        t_sq = t.squeeze(-1)
        tn = torch.clamp(t_sq / self.T * (ns - 1), 0.0, ns - 2 + 1e-6)
        idx = torch.clamp(tn.long(), 0, ns - 2)
        frac = tn - idx.float()
        K_lo = torch.gather(u_sensors[:, :ns], 1, idx.unsqueeze(1)).squeeze(1)
        K_hi = torch.gather(u_sensors[:, :ns], 1, (idx + 1).unsqueeze(1)).squeeze(1)
        Ta_lo = torch.gather(u_sensors[:, ns:2 * ns], 1, idx.unsqueeze(1)).squeeze(1)
        Ta_hi = torch.gather(u_sensors[:, ns:2 * ns], 1, (idx + 1).unsqueeze(1)).squeeze(1)
        return K_lo * (1 - frac) + K_hi * frac, Ta_lo * (1 - frac) + Ta_hi * frac

    def _theta_ss(self, K_t, Ta_t):
        """The model's analytic attractor.

        PHASE 2 FIX 1. Two modes:

        `true_fixed_point` (default) — the actual fixed point of the
        data-generating ODE, by contraction iteration on the model's own
        registered buffers. This is the same quantity IC generation and the
        rollout now use, so all four sites finally agree.

        `formula_C` — what v57 used: m_exp for the hot-spot gradient inside the Rf
        estimate, n_exp for the thermal output. Coincides exactly with formula B at
        K = 1 (the load factor is 1, so every exponent gives 1) and diverges as K
        moves away from 1 in either direction, by -8.45 degC at K = 1.3,
        theta_a = 30 against the true fixed point. Required to reproduce the
        stored checkpoints.
        """
        R = self.R_load_buf
        ne = self.n_exp_buf
        me = self.m_exp_buf
        Do = self.DTheta_oil_R_buf
        Dhs = self.DTheta_HS_R_buf
        ac = self.alpha_Cu_buf
        Tr = self.T_HS_ref_C_buf

        if self.theta_ss_mode == "true_fixed_point":
            return true_fixed_point_torch(
                K_t, Ta_t, R_load_v=R, n_exp_v=ne, m_exp_v=me,
                DTheta_oil_R_v=Do, DTheta_HS_R_v=Dhs, alpha_Cu_v=ac,
                T_HS_ref_C_v=Tr)

        fac_m = ((1.0 + K_t ** 2 * R) / (1.0 + R)) ** me
        fac_n = ((1.0 + K_t ** 2 * R) / (1.0 + R)) ** ne
        th_HS0 = Ta_t + Do * fac_n + Dhs * fac_m
        Rf = (1.0 + ac * (th_HS0 - Tr)).clamp(0.8, 1.5)
        return Ta_t + Do * ((1.0 + K_t ** 2 * R * Rf) / (1.0 + R)) ** ne

    def _ode_baseline(self, x0_TO, u_sensors, t, theta_ss_grid=None):
        """Exact first-order ODE solution by trapezoid on the sensor grid.

        theta(t) = x0 * exp(-t/tau) + exp(-t/tau) * integral_0^t theta_ss(s) exp(s/tau)/tau ds

        Correct for time-varying K, unlike the v23 envelope it replaced. Note
        exp(s/tau) grows by exp(T/tau) across the window; at TW = 720 and
        tau = 150 that is exp(4.8) ~ 122, which 100 trapezoid points resolve.
        The battery model faces exp(19) ~ 2e8 on the same construction and had to
        switch to a closed-form recurrence (n15 cell 7).
        """
        ns = u_sensors.shape[-1] // 2
        B = x0_TO.shape[0]
        tau = self.tau_oil_buf
        t_sq = t.squeeze(-1)
        s_grid = torch.linspace(0.0, self.T, ns, device=x0_TO.device)
        # theta_ss on the sensor grid needs no gradient: it reaches the output only
        # through F_cum, and the t-gradient flows through `frac` and `decay`, not
        # through these values. So it is safe to read from a cache.
        theta_ss_s = (theta_ss_grid if theta_ss_grid is not None
                      else self._theta_ss(u_sensors[:, :ns], u_sensors[:, ns:2 * ns]))
        exp_s = torch.exp(s_grid / tau)
        integrand = theta_ss_s * exp_s.unsqueeze(0) / tau
        ds = self.T / (ns - 1)
        trap = 0.5 * (integrand[:, :-1] + integrand[:, 1:]) * ds
        F_cum = torch.cat(
            [torch.zeros(B, 1, device=x0_TO.device), torch.cumsum(trap, dim=1)], dim=1)
        tn = torch.clamp(t_sq / self.T * (ns - 1), 0.0, ns - 1 - 1e-6)
        idx = torch.clamp(tn.long(), 0, ns - 2)
        frac = tn - idx.float()
        F_t = (torch.gather(F_cum, 1, idx.unsqueeze(1)).squeeze(1) * (1 - frac)
               + torch.gather(F_cum, 1, (idx + 1).unsqueeze(1)).squeeze(1) * frac)
        decay = torch.exp(-t_sq / tau)
        return (x0_TO.squeeze(-1) * decay + decay * F_t).unsqueeze(-1)

    def _thermal_trunk_feat(self, t, u_sensors, x0_TO, theta_ss_grid=None):
        return build_trunk_feats(
            t, u_sensors, x0_TO, self.T, u_sensors.shape[-1] // 2,
            self.exp_decay_rates, self.tau_oil_buf, self.R_load_buf,
            self.n_exp_buf, self.m_exp_buf, self.DTheta_oil_R_buf,
            self.DTheta_HS_R_buf, self.alpha_Cu_buf, self.T_HS_ref_C_buf,
            theta_ss_mode=self.theta_ss_mode,      # PHASE 2 FIX 1
            theta_ss_grid=theta_ss_grid,
        )

    def _interp_grid_5d(self, F_grid, t):
        ns = F_grid.shape[1]
        t_sq = t.squeeze(-1)
        tn = torch.clamp(t_sq / self.T * (ns - 1), 0.0, ns - 1 - 1e-6)
        idx = torch.clamp(tn.long(), 0, ns - 2)
        frac = tn - idx.float()
        idx3 = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 5)
        F_lo = torch.gather(F_grid, 1, idx3).squeeze(1)
        F_hi = torch.gather(F_grid, 1, idx3 + 1).squeeze(1)
        return F_lo + frac.unsqueeze(1) * (F_hi - F_lo)

    def _thermal_predict_grid(self, x0_TO, u_sensors, b_th, n_grid=None,
                              theta_ss_grid=None):
        """theta_TO on a uniform grid, to drive the gas quadrature.

        `n_grid=20` while training gives 5x fewer trunk evaluations for a ~3-4x
        speedup; the gas integral error is under 0.0001% at 20 points because the
        integrand is smooth and tau_gas >> TW. `n_grid=None` uses all
        `n_sensors` points at eval time.
        """
        ns = n_grid if n_grid is not None else self.n_sensors
        B = x0_TO.shape[0]
        tau = self.tau_oil_buf
        norm = 1.0 - torch.exp(-self.T / tau)
        s_grid = torch.linspace(0.0, self.T, ns, device=x0_TO.device)
        s_BN = s_grid.unsqueeze(0).unsqueeze(-1).expand(B, ns, 1).reshape(B * ns, 1)
        x0e = x0_TO.unsqueeze(1).expand(B, ns, 1).reshape(B * ns, 1)

        if ns != self.n_sensors:
            tn = torch.clamp(s_grid / self.T * (self.n_sensors - 1),
                             0, self.n_sensors - 2 + 1e-6)
            idx_s = torch.clamp(tn.long(), 0, self.n_sensors - 2)
            frac_s = (tn - idx_s.float()).unsqueeze(0)
            K_full = u_sensors[:, :self.n_sensors]
            Ta_full = u_sensors[:, self.n_sensors:]
            K_sub = K_full[:, idx_s] * (1 - frac_s) + K_full[:, idx_s + 1] * frac_s
            Ta_sub = Ta_full[:, idx_s] * (1 - frac_s) + Ta_full[:, idx_s + 1] * frac_s
            u_sub = torch.cat([K_sub, Ta_sub], dim=1)
            ss_sub = None            # subsampled K, so the cache does not apply
        else:
            u_sub = u_sensors
            ss_sub = theta_ss_grid

        # Solve theta_ss on the (B, ns) sub-grid ONCE and expand, instead of on the
        # (B*ns, ns) expanded tensor. Expanding identical values is exact, so this
        # is a pure ns-fold saving with no numerical change.
        if ss_sub is None:
            ss_sub = self._theta_ss(u_sub[:, :ns], u_sub[:, ns:2 * ns])

        s_e = u_sub.unsqueeze(1).expand(B, ns, -1).reshape(B * ns, -1)
        ss_e = ss_sub.unsqueeze(1).expand(B, ns, -1).reshape(B * ns, -1)
        b_e = b_th.unsqueeze(1).expand(B, ns, -1).reshape(B * ns, -1)
        th_feat = self._thermal_trunk_feat(s_BN, s_e, x0e, theta_ss_grid=ss_e)
        tr_th = self.trunk_th(th_feat)
        delta = (b_e * tr_th).sum(-1) + self.bias_th.squeeze()
        delta = delta.reshape(B, ns)
        baseline = self._ode_baseline(x0e, s_e, s_BN, theta_ss_grid=ss_e).reshape(B, ns)
        t_exp = (1.0 - torch.exp(-s_grid / tau)) / norm
        return baseline + t_exp.unsqueeze(0) * self.output_scale * delta

    def _gas_integral(self, t, u_sensors, x0_gas, theta_TO_grid):
        """Analytic Arrhenius quadrature. No `nn.Parameter` appears here.

        c_i(t) = c_i(0) + trapz_0^t k_gen_i V_arr_i(theta_HS(s)) ds - k_dis_i c_i(0) t

        PHASE 2 FIX 6 (DECISIONS N-1). This quadrature is the model's entire gas
        prediction, so a cap here is a cap on the answer. It carried
        `V_arr.clamp(max=1e4)`; the reference ODE that produced every label caps
        the temperature at 573.15 K instead and never caps the rate. The
        quadrature now uses the reference's envelope, so model and ground truth
        integrate the same kinetics. `legacy_V_clamp=True` restores v57.

        Clamps remaining: `Rf.clamp(0.8, 1.5)` and
        `T_HS_K.clamp(313.15, 573.15)`, both matching `fast_rhs_np`.
        """
        # Delegated to `cod.models.cascade.gas_integral`, which is the one
        # definition of this quadrature. The C-11 in-cascade configurations of
        # FNO, MIONet and S-DeepONet need the same physics, and a second copy is
        # the defect CLAUDE.md names by example (`ode_physics_loss` existed three
        # times). The model's own buffers and interpolator are passed in, so the
        # delegation is bit-identical rather than merely equal to within float32
        # rounding — `audit_port/scripts/29_cascade_refactor_identical.py` checks
        # that against this file's pre-refactor version read out of git, and the
        # Phase 1 gates are the end-to-end check.
        return gas_integral(
            t, u_sensors, x0_gas, theta_TO_grid, self.T, self.n_sensors,
            self.k_gen_buf, self.k_dis_buf, self.E_act_buf,
            self._interp_grid_5d, self.legacy_V_clamp)

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        """`theta_ss_grid`, if given, is theta_ss on the sensor grid, (B, n_sensors).

        Supplying it skips the contraction solve that Phase 2 fix 1 introduced.
        Bit-exact — see `_ode_baseline` and `build_trunk_feats` for why those uses
        need no gradient. Omit it and the model computes everything itself.
        """
        B = x0.shape[0]
        x0_TO = x0[:, 0:1]
        x0_gas = x0[:, 1:]
        b_th = self.branch_th(torch.cat([self._norm_TO(x0_TO), u_sensors], dim=-1))
        th_feat = self._thermal_trunk_feat(t, u_sensors, x0_TO,
                                           theta_ss_grid=theta_ss_grid)
        tr_th = self.trunk_th(th_feat)
        delta_TO = (b_th * tr_th).sum(dim=-1, keepdim=True) + self.bias_th
        tau = self.tau_oil_buf
        norm = 1.0 - torch.exp(-self.T / tau)
        t_exp = (1.0 - torch.exp(-t / tau)) / norm
        baseline = self._ode_baseline(x0_TO, u_sensors, t,
                                     theta_ss_grid=theta_ss_grid)
        theta_TO_pred = baseline + t_exp * self.output_scale * delta_TO

        # INTENDED: the cascade is one-way. Detaching here is what makes
        # d L_gas / d theta exactly zero. Do not remove — see the class docstring
        # and reference/audit/results/step4_gradient_flow.md.
        theta_TO_grid = self._thermal_predict_grid(
            x0_TO, u_sensors, b_th, n_grid=20 if self.training else None,
            theta_ss_grid=theta_ss_grid).detach()

        gases_pred = self._gas_integral(t, u_sensors, x0_gas, theta_TO_grid)
        return torch.cat([theta_TO_pred, gases_pred], dim=-1)


class CODNoBaseline(CODOperator):
    """Ablation A: the analytic baseline H is replaced by the constant x0.

    Source: n00 cell 10 L438 (`COD_NoBaseline`). The network must learn the full
    thermal signal rather than a residual.

    Note the internal inconsistency the audit records (B-2): the same nominal
    Ablation A reports 18.5% in `COD_ablation_study` c10 and 13,437.1% in the
    standalone `PI_DeepONet_abl_A_done`, a factor of ~700. Neither checkpoint is
    among the supplied artifacts, so this class was ported for completeness.

    FIXED 2026-08-03, and the defect is worth recording. The override carried the
    pre-fix-1 signature `(self, x0, u, t)` while `forward` and
    `_thermal_predict_grid` both call `_ode_baseline(..., theta_ss_grid=...)`,
    which fix 1 added. **Every forward pass raised `TypeError`**, so this class
    had never run — the docstring's own "not exercised by any verification gate"
    was the reason it went unnoticed for as long as the argument had existed.
    O-12 could not have started against it. The signature now matches the parent
    exactly; `theta_ss_grid` is accepted and ignored, which is correct, because
    the whole point of this ablation is that the analytic attractor does not reach
    the output.

    This is the one-variable test N-8 and O-12 need: same network, same trunk,
    same cascade, same training pipeline as `CODOperator` — only `H` is replaced
    by the constant `x0`. The monolithic baselines are not a substitute because
    they change two things at once (no baseline *and* no cascade) and carry J-8.
    """

    def _ode_baseline(self, x0_TO, u_sensors, t, theta_ss_grid=None):
        return x0_TO


def cod_predict(model, x0, u, t, theta_ss_grid=None):
    """Uniform predict signature, so the sweep and the gates treat all models
    the same way (n15 cell 2 L293). `theta_ss_grid` is optional and ignored by the
    monolithic baselines, which have no analytic baseline."""
    return model(x0, u, t, theta_ss_grid=theta_ss_grid)


def steady_state_grid(model, u_sensors):
    """theta_ss on the sensor grid for a batch of profiles, (B, n_sensors).

    Computed with the model's own registered buffers and its own `_theta_ss`, so the
    result is bit-identical to what `forward` would have computed internally. Use
    this to build the dataset cache.
    """
    ns = model.n_sensors
    with torch.no_grad():
        return model._theta_ss(u_sensors[:, :ns], u_sensors[:, ns:2 * ns])


__all__ = ["CODOperator", "CODNoBaseline", "cod_predict", "steady_state_grid",
           "THETA_SS_MODES"]
