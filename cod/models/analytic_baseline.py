"""The IEC 60076-7 analytic thermal baseline `H(t)`, as a shared component.

WHY THIS MODULE EXISTS. The 2x2 factorial (ANALYSIS_PLAN, and the C-11 extension)
runs every tier-1 architecture with and without the analytic baseline. That means
FNO, MIONet and S-DeepONet each need `H(t)`, and it already lives inside
`CODOperator._ode_baseline`. Copying it three times would put a fourth variant of
a physics routine in the repo, which is the defect CLAUDE.md names by example
(`ode_physics_loss` once existed in three versions), and the cascade extraction
already set the precedent for how to avoid it.

So it is extracted **once** here and `CODOperator._ode_baseline` delegates.
`audit_port/scripts/35_baseline_refactor_identical.py` asserts the delegation is
bit-identical against the pre-refactor file read out of git, and the Phase 1 gates
are the end-to-end check.

WHAT IT IS. The exact solution of the first-order top-oil ODE with time-varying
forcing, by trapezoid on the sensor grid:

    theta(t) = x0 exp(-t/tau)
             + exp(-t/tau) * integral_0^t theta_ss(s) exp(s/tau) / tau ds

`exp(s/tau)` grows by `exp(T/tau)` across the window — `exp(4.8) ~ 122` at
`TW = 720`, `tau = 150` — which 100 trapezoid points resolve.

NO `nn.Parameter`, AND THAT IS THE POINT. `H` is known physics, not learned. It
is what lets the with-baseline variants say the model degrades to the IEC standard
when the network output goes to zero, and what makes "does the analytic baseline
help this architecture" a one-variable question rather than a capacity question:
`n_parameters()` is identical with and without it.

NAMING. An architecture given this baseline is **not** that architecture any more;
it is a hybrid, and reporting it under the source paper's name would misattribute
its behaviour. The factorial labels these cells "FNO + IEC delta-learning" and so
on. See PORT_LOG J-92.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import (
    DTheta_HS_R,
    DTheta_oil_R,
    R_load,
    T_HS_ref_C,
    TW,
    alpha_Cu,
    m_exp,
    n_exp,
    tau_oil,
)
from cod.data.steady_state import true_fixed_point_torch


def ode_baseline(x0_TO, u_sensors, t, T, tau, theta_ss_grid):
    """`H(t)` given theta_ss already on the sensor grid.

    Split from the attractor computation so that a caller holding a cached
    `theta_ss_grid` — which the dataset provides, and which carries no gradient —
    pays nothing, and so that this arithmetic has one definition regardless of
    which attractor produced its input.
    """
    ns = u_sensors.shape[-1] // 2
    B = x0_TO.shape[0]
    t_sq = t.squeeze(-1)
    s_grid = torch.linspace(0.0, T, ns, device=x0_TO.device)
    exp_s = torch.exp(s_grid / tau)
    integrand = theta_ss_grid * exp_s.unsqueeze(0) / tau
    ds = T / (ns - 1)
    trap = 0.5 * (integrand[:, :-1] + integrand[:, 1:]) * ds
    F_cum = torch.cat(
        [torch.zeros(B, 1, device=x0_TO.device), torch.cumsum(trap, dim=1)], dim=1)
    tn = torch.clamp(t_sq / T * (ns - 1), 0.0, ns - 1 - 1e-6)
    idx = torch.clamp(tn.long(), 0, ns - 2)
    frac = tn - idx.float()
    F_t = (torch.gather(F_cum, 1, idx.unsqueeze(1)).squeeze(1) * (1 - frac)
           + torch.gather(F_cum, 1, (idx + 1).unsqueeze(1)).squeeze(1) * frac)
    decay = torch.exp(-t_sq / tau)
    return (x0_TO.squeeze(-1) * decay + decay * F_t).unsqueeze(-1)


class AnalyticBaseline(nn.Module):
    """`H(t)` with its own constants, for architectures that are not COD.

    `CODOperator` keeps its own buffers and calls `ode_baseline` directly; this
    wrapper exists so the factorial's with-baseline cells get the same physics
    without inheriting from `CODOperator`, which would drag in a branch and trunk
    they do not use.

    Parameter-free by construction. `n_parameters()` on a model containing one of
    these counts nothing here, so a with-baseline cell is not credited with
    capacity it does not have — which is what keeps the factorial's baseline
    factor a one-variable change.
    """

    def __init__(self, T: float = TW, n_sensors: int = 100):
        super().__init__()
        self.T = float(T)
        self.n_sensors = int(n_sensors)
        for name, val in [
            ("tau_oil_buf", tau_oil), ("R_load_buf", R_load),
            ("n_exp_buf", n_exp), ("m_exp_buf", m_exp),
            ("DTheta_oil_R_buf", DTheta_oil_R), ("DTheta_HS_R_buf", DTheta_HS_R),
            ("alpha_Cu_buf", alpha_Cu), ("T_HS_ref_C_buf", T_HS_ref_C),
        ]:
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32))

    def theta_ss(self, K_t, Ta_t):
        """The true fixed point of the data-generating ODE.

        Only `true_fixed_point` is offered here, not COD's `formula_C` mode:
        that mode exists solely to reproduce v57 checkpoints, and no factorial
        cell reproduces one.
        """
        return true_fixed_point_torch(
            K_t, Ta_t, R_load_v=self.R_load_buf, n_exp_v=self.n_exp_buf,
            m_exp_v=self.m_exp_buf, DTheta_oil_R_v=self.DTheta_oil_R_buf,
            DTheta_HS_R_v=self.DTheta_HS_R_buf, alpha_Cu_v=self.alpha_Cu_buf,
            T_HS_ref_C_v=self.T_HS_ref_C_buf)

    def forward(self, x0_TO, u_sensors, t, theta_ss_grid=None):
        ns = self.n_sensors
        if theta_ss_grid is None:
            theta_ss_grid = self.theta_ss(u_sensors[:, :ns],
                                          u_sensors[:, ns:2 * ns])
        return ode_baseline(x0_TO, u_sensors, t, self.T, self.tau_oil_buf,
                            theta_ss_grid)

    def on_grid(self, x0_TO, u_sensors, t_grid, theta_ss_grid=None):
        """`H` at every point of `t_grid` (n,1), returned (B, n).

        Needed by the in-cascade variants, whose quadrature integrates the
        thermal trajectory over the whole window rather than at one query time.
        """
        B = x0_TO.shape[0]
        n = t_grid.shape[0]
        ns = self.n_sensors
        if theta_ss_grid is None:
            theta_ss_grid = self.theta_ss(u_sensors[:, :ns],
                                          u_sensors[:, ns:2 * ns])
        x0e = x0_TO.unsqueeze(1).expand(B, n, 1).reshape(B * n, 1)
        ue = u_sensors.unsqueeze(1).expand(B, n, 2 * ns).reshape(B * n, 2 * ns)
        sse = theta_ss_grid.unsqueeze(1).expand(B, n, ns).reshape(B * n, ns)
        te = t_grid.reshape(1, n, 1).expand(B, n, 1).reshape(B * n, 1)
        return ode_baseline(x0e, ue, te, self.T, self.tau_oil_buf,
                            sse).reshape(B, n)


__all__ = ["AnalyticBaseline", "ode_baseline"]
