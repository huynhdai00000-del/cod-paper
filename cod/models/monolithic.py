"""Monolithic PI-DeepONet baselines.

PHASE 1 — FAITHFUL PORT. Three variants exist in the source and they are NOT
interchangeable — the audit's two conflicting monolithic headline numbers come
from two different classes with two different checkpoints:

    MonolithicFair       n15 cell 2 L298 (`PIDeepONet_Mono_Fair`)
                         checkpoint mono_fair_v2_perstate.pt  ->  13,199.7%
                         single p-dim bottleneck, per-state learnable output
                         scale initialised from x_std, phi(t) IC mask

    MonolithicMultiHead  n15 cell 8 L950 (`PIDeepONet_Mono_MultiHead`)
                         checkpoint mono_multihead.pt          ->  18,076.6%
                         branch emits p*SD, per-state dot product, no bottleneck

    MonolithicSoftIC     n00 cell 8 L386 (`PIDeepONet_Mono`)
                         checkpoint mono_fair_v1.pt            ->  18,933.3%
                         NO output scaling at all, and a sigmoid(10t/T) soft IC
                         mask instead of phi(t)

So the audit's open question 3 — "which monolithic run is cited, 13,199.7% or
18,933.3%?" — has a mechanical answer: they are different architectures, not two
runs of one experiment. `mono_fair_v1.pt` is not among the supplied artifacts, so
`MonolithicSoftIC` cannot be verified here; it is ported so the provenance of
18,933.3% is recorded in code rather than lost.

KNOWN DEFECT in all three (PORT_LOG J-8, NOT in the audit): the constructor
parameter is named `n_exp` and shadows the module-global thermal exponent
`n_exp = 0.8`, so the buffer registration loop binds `self.ne = 12.0` — the
number of exponential trunk features — as the thermal exponent. Every monolithic
checkpoint stores `ne = 12.0`, so this is what they were trained with. Ported
with the shadowing intact.

This matters for audit M-2. The claim "monolithic architectures fail" rests on
runs whose analytic trunk features were computed with a thermal exponent of 12
instead of 0.8, on top of the causal weights having underflowed to exactly zero
(`wm = 0.000`) and a final loss five orders above COD's. Three independent
reasons the comparison does not support the claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import (
    DTheta_HS_R,
    DTheta_oil_R,
    N_SENSORS,
    R_load,
    STATE_DIM_FAST,
    T_HS_ref_C,
    TW,
    alpha_Cu,
    m_exp,
    tau_oil,
)
from cod.models.blocks import ModifiedMLP, build_trunk_feats


def _register_physics_buffers(module: nn.Module, ne_value) -> None:
    """Register the abbreviated physics buffers the monolithic checkpoints use.

    `ne_value` is passed in explicitly so the shadowing defect is visible at the
    call site instead of hiding inside a closure over a parameter name. The
    source's effective value is the constructor's `n_exp` argument (12), not the
    thermal exponent (0.8).
    """
    for name, val in [
        ("tau", tau_oil), ("R", R_load), ("ne", ne_value), ("me", m_exp),
        ("Do", DTheta_oil_R), ("Dhs", DTheta_HS_R),
        ("ac", alpha_Cu), ("Tr", T_HS_ref_C),
    ]:
        module.register_buffer(name, torch.tensor(float(val), dtype=torch.float32))


def _per_state_output_scale_raw(x_std) -> nn.Parameter:
    """Per-state learnable output scale, initialised from the IC standard deviations.

    n15 cell 2 L306-L310. The `torch.where(init_scale > 20, init_scale,
    log(expm1(init_scale) + 1e-6))` is a softplus pre-image only on the second
    branch; above 20 the raw value is used directly because softplus is
    effectively the identity there. This is the "KEY FIX" the source labels as
    what separates Mono Fair from the earlier `PIDeepONet_Mono`.
    """
    if x_std is None:
        return nn.Parameter(torch.zeros(STATE_DIM_FAST))
    init_scale = torch.tensor(x_std, dtype=torch.float32).clamp(min=1.0)
    raw = torch.where(init_scale > 20, init_scale,
                      torch.log(torch.expm1(init_scale) + 1e-6))
    return nn.Parameter(raw)


class MonolithicFair(nn.Module):
    """Monolithic PI-DeepONet with a single p-dim bottleneck.

    b . tr -> Linear(p, 6) -> six outputs, all sharing one p-dim representation.

    What is the same as COD: the ModifiedMLP blocks, the 32-dim trunk features,
    the branch input, the adaptive loss weights, the causal weighting, 25,000
    epochs and batch size 128. What is removed — and this is precisely the COD
    contribution being ablated: the analytic baseline H, and the cascaded gas
    integral.

    Checkpoint: mono_fair_v2_perstate.pt, sweep_mono_fair_p{4,8,16,32,64}.pt.
    """

    def __init__(self, d_h: int = 128, p: int = 64, n_layers: int = 4,
                 n_exp: int = 12, x_mean=None, x_std=None):
        super().__init__()
        self.state_dim = STATE_DIM_FAST
        self.T = TW
        self.n_sensors = N_SENSORS
        self.p = p

        self.branch = ModifiedMLP(STATE_DIM_FAST + 2 * N_SENSORS, d_h, p, n_layers)
        self.trunk = ModifiedMLP(2 * n_exp + 4 + 4, d_h // 2, p, n_layers - 1)
        self.out_proj = nn.Linear(p, STATE_DIM_FAST)
        self.bias = nn.Parameter(torch.zeros(STATE_DIM_FAST))
        self.output_scale_raw = _per_state_output_scale_raw(x_std)

        k_vals = torch.exp(torch.linspace(torch.log(torch.tensor(0.2)),
                                          torch.log(torch.tensor(5.0)), n_exp))
        self.register_buffer("exp_rates", k_vals)
        # PORT_LOG J-8: `n_exp` here is the constructor argument (12), which is
        # what the source binds as the thermal exponent. Faithful, not correct.
        _register_physics_buffers(self, ne_value=n_exp)

        if x_mean is not None:
            self.register_buffer("xm", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("xs", torch.tensor(x_std + 1e-8, dtype=torch.float32))

    def n_parameters(self) -> int:
        return sum(q.numel() for q in self.parameters())

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def forward(self, x0, u, t):
        x0n = (x0 - self.xm) / self.xs if hasattr(self, "xm") else x0
        b = self.branch(torch.cat([x0n, u], dim=-1))
        tf = build_trunk_feats(t, u, x0[:, 0:1], self.T, N_SENSORS, self.exp_rates,
                               self.tau, self.R, self.ne, self.me,
                               self.Do, self.Dhs, self.ac, self.Tr)
        tr = self.trunk(tf)
        dot = b * tr
        raw_out = self.out_proj(dot) + self.bias
        phi = (1 - torch.exp(-t / self.tau)) / (1 - torch.exp(-self.T / self.tau))
        return x0 + phi * self.output_scale * raw_out


class MonolithicMultiHead(nn.Module):
    """Monolithic PI-DeepONet with no bottleneck: p basis functions per state.

    branch: (6 + 2*100) -> d_h -> p*6, reshaped to (B, 6, p)
    trunk:  (32)        -> d_h -> p,   shared across states
    output: raw_out[s] = sum_k branch[s,k] * trunk[k]

    Strictly more expressive than `MonolithicFair` — 6x the output capacity, no
    shared bottleneck — at the same depth, width and training pipeline. The
    source built it to answer a specific reviewer objection: if the 13,200%
    failure were an artefact of the bottleneck, this variant should recover. It
    scores 18,076.6%, i.e. worse, which is the source's own evidence that the
    failure is not the bottleneck.

    Checkpoint: mono_multihead.pt.
    """

    def __init__(self, d_h: int = 128, p: int = 64, n_layers: int = 4,
                 n_exp: int = 12, x_mean=None, x_std=None):
        super().__init__()
        self.state_dim = STATE_DIM_FAST
        self.T = TW
        self.n_sensors = N_SENSORS
        self.p = p

        self.branch = ModifiedMLP(STATE_DIM_FAST + 2 * N_SENSORS, d_h,
                                  p * STATE_DIM_FAST, n_layers)
        self.trunk = ModifiedMLP(2 * n_exp + 4 + 4, d_h // 2, p, n_layers - 1)
        self.bias = nn.Parameter(torch.zeros(STATE_DIM_FAST))
        self.output_scale_raw = _per_state_output_scale_raw(x_std)

        k_vals = torch.exp(torch.linspace(torch.log(torch.tensor(0.2)),
                                          torch.log(torch.tensor(5.0)), n_exp))
        self.register_buffer("exp_rates", k_vals)
        _register_physics_buffers(self, ne_value=n_exp)   # PORT_LOG J-8

        if x_mean is not None:
            self.register_buffer("xm", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("xs", torch.tensor(x_std + 1e-8, dtype=torch.float32))

    def n_parameters(self) -> int:
        return sum(q.numel() for q in self.parameters())

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def forward(self, x0, u, t):
        x0n = (x0 - self.xm) / self.xs if hasattr(self, "xm") else x0
        b = self.branch(torch.cat([x0n, u], dim=-1))
        b = b.reshape(b.shape[0], self.state_dim, self.p)
        tf = build_trunk_feats(t, u, x0[:, 0:1], self.T, N_SENSORS, self.exp_rates,
                               self.tau, self.R, self.ne, self.me,
                               self.Do, self.Dhs, self.ac, self.Tr)
        tr = self.trunk(tf)
        raw_out = (b * tr.unsqueeze(1)).sum(-1) + self.bias
        phi = (1 - torch.exp(-t / self.tau)) / (1 - torch.exp(-self.T / self.tau))
        return x0 + phi * self.output_scale * raw_out


class MonolithicSoftIC(nn.Module):
    """The earlier monolithic baseline: no output scaling, sigmoid soft IC.

    Source: n00 cell 8 L386 (`PIDeepONet_Mono`), checkpoint `mono_fair_v1.pt`,
    reported at 18,933.3%. Two differences from `MonolithicFair` explain the gap:

      * no `output_scale` at all, so the network must emit values spanning six
        orders of magnitude (theta_TO ~ 1e2 degC, CO2 ~ 1e3 ppm) from one
        unscaled linear head;
      * the IC is enforced softly by `sigmoid(10t/T)`, which equals 0.5 at t = 0
        rather than 0 — so `x(0) != x0` and the initial condition is violated by
        construction, unlike `phi(t)` which is exactly 0 at t = 0.

    The second point is worth stating in the paper: this variant cannot satisfy
    the exact-IC guarantee that COD gets structurally, so part of its error is
    definitional. `mono_fair_v1.pt` is not among the supplied artifacts, so this
    class is not exercised by any verification gate.
    """

    def __init__(self, d_h: int = 128, p: int = 64, n_layers: int = 4,
                 n_exp: int = 12, x_mean=None, x_std=None):
        super().__init__()
        self.state_dim = STATE_DIM_FAST
        self.T = TW
        self.n_sensors = N_SENSORS
        self.p = p

        self.branch = ModifiedMLP(STATE_DIM_FAST + 2 * N_SENSORS, d_h, p, n_layers)
        self.trunk = ModifiedMLP(2 * n_exp + 4 + 4, d_h // 2, p, n_layers - 1)
        self.out_proj = nn.Linear(p, STATE_DIM_FAST)
        self.bias = nn.Parameter(torch.zeros(STATE_DIM_FAST))

        k_vals = torch.exp(torch.linspace(torch.log(torch.tensor(0.2)),
                                          torch.log(torch.tensor(5.0)), n_exp))
        self.register_buffer("exp_rates", k_vals)
        _register_physics_buffers(self, ne_value=n_exp)   # PORT_LOG J-8

        if x_mean is not None:
            self.register_buffer("xm", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("xs", torch.tensor(x_std + 1e-8, dtype=torch.float32))

    def n_parameters(self) -> int:
        return sum(q.numel() for q in self.parameters())

    def forward(self, x0, u, t):
        x0n = (x0 - self.xm) / self.xs if hasattr(self, "xm") else x0
        b = self.branch(torch.cat([x0n, u], dim=-1))
        tf = build_trunk_feats(t, u, x0[:, 0:1], self.T, N_SENSORS, self.exp_rates,
                               self.tau, self.R, self.ne, self.me,
                               self.Do, self.Dhs, self.ac, self.Tr)
        tr = self.trunk(tf)
        out = self.out_proj(b * tr) + self.bias
        t_mask = torch.sigmoid(t * 10 / self.T)   # 0.5 at t = 0, not 0
        return x0 + t_mask * out


def mono_predict(model, x0, u, t):
    """Uniform predict signature shared by all monolithic variants."""
    return model(x0, u, t)


__all__ = ["MonolithicFair", "MonolithicMultiHead", "MonolithicSoftIC",
           "mono_predict"]
