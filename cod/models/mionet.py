"""MIONet, adapted from its ODE/PDE benchmarks to the six-state transformer system.

REFERENCE. Jin, Meng, Lu, "MIONet: Learning multiple-input operators via tensor
product", 2022 (`reference/papers/2022.Jin_MIONet...pdf`). Nothing here is built
from memory.

WHY THIS ARCHITECTURE FITS THE PROBLEM UNUSUALLY WELL. MIONet exists to remove
DeepONet's restriction to a single input function (§1): a DeepONet branch takes one
function, so a problem whose operator depends on an initial condition *and* a
forcing history has to concatenate them and pretend they live on one domain. This
problem is exactly that case — `G: (x0, K, theta_a) -> x(.)`, with `x0` a point in
R^6 and `K`, `theta_a` functions on the window. The paper's own comparison method
("we simply concatenate all the input functions together as the input of DeepONet
branch net", §4) is what the existing `MonolithicFair` baseline already does, so
MIONet against that pair is a controlled test of the multi-input structure itself.

ARCHITECTURE (paper Eq. 16, Fig. 1) — the low-rank form, which §3.1 names the
default and the Remark prefers on cost grounds:

    G(v_1..v_n)(y) = S( g_1(v_1) . g_2(v_2) . ... . g_n(v_n) . f(y) ) + b

with `.` the Hadamard product, `S` the sum over components, and `b` a trainable
bias (Corollary 2). All branches and the trunk emit the same width `p`.

SIX OUTPUTS. Eq. (16) returns a scalar. Corollary 3(iii) covers a vector-valued
target: operators sharing the same input spaces may share the branch nets `g_i`
and differ only in the trunk coefficients `u^j`. So the branches are shared and
the trunk emits `p * 6`, reshaped to `(p, 6)`; the merged branch vector contracts
against it per state. This is the paper's construction, not a convenience.

TWO PAPER DEVICES DELIBERATELY NOT USED, both because their precondition fails:

* **Linear branch for the initial condition.** §4.3 makes the `u_0` branch a
  bias-free linear layer, justified by Corollary 4: if `G` is linear in `v_i`,
  linear `g_i` suffices. Our operator is *not* linear in `x0` — `theta_TO` enters
  `Rf`, `hot_spot_ETC` and `V_arr`, the last exponentially. Using a linear branch
  here would not be transferring a technique, it would be asserting a false
  property of the system.
* **Periodic trunk layer.** §4.3 feeds `(cos 2pi x, sin 2pi x, cos 4pi x,
  sin 4pi x)` to the trunk when the solution is periodic in the trunk variable,
  and reports the best accuracy of the paper from it (1.29% vs 1.98%). Our trunk
  variable is `t` on a 12 h window, and the trajectory is not periodic in it — the
  window is a slice of a 24 h day at random phase. Encoding a periodicity the
  system does not have would inject a false prior.

Both are recorded in PORT_LOG J-90 so that "MIONet was not given its best-case
tricks" is on the record before any number exists, with the reason each is
inapplicable rather than merely omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW, tau_oil
from cod.models.blocks import ic_mask, per_state_output_scale_raw
from cod.models.cascade import GasCascade

#: Paper §4.1, the ODE-system experiment — the closest of its three benchmarks to
#: this problem: "Depth 2, Width 200", ReLU, Adam at lr 1e-3. Depth counts hidden
#: layers. `p` is the shared branch/trunk output width.
MIONET_DEPTH = 2
MIONET_WIDTH = 200
MIONET_P = 200


def _fnn(in_dim: int, width: int, depth: int, out_dim: int) -> nn.Sequential:
    """Plain FNN with ReLU, per §4: "The branch and trunk nets are all chosen as
    fully-connected neural networks ... The activation in all networks is set to
    ReLU." No residual connections, no ModifiedMLP — using COD's block here would
    make the comparison about that block rather than about MIONet."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _MIONetCore(nn.Module):
    """Three branches, one trunk, Hadamard merge, per-state trunk coefficients.

    branch_x0   x0 (6, normalised)        -> p
    branch_K    K on the sensor grid      -> p
    branch_Ta   theta_a on the sensor grid-> p
    trunk       t                         -> p * out_dim

    `K` and `theta_a` get separate branches rather than one branch over the
    concatenated 200-vector. That is the paper's own point: they are different
    input functions, and §2 is explicit that the limitation being removed is
    precisely "all components of the input function must be defined on the same
    domain". They happen to share a grid here, but they are physically distinct
    inputs with different units and ranges, and giving each its own branch is what
    makes this MIONet rather than a DeepONet with a wider branch.
    """

    def __init__(self, out_dim: int, n_sensors: int = N_SENSORS,
                 width: int = MIONET_WIDTH, depth: int = MIONET_DEPTH,
                 p: int = MIONET_P):
        super().__init__()
        self.p, self.out_dim = int(p), int(out_dim)
        self.T_dom = float(TW)
        self.n_sensors = int(n_sensors)
        self.branch_x0 = _fnn(STATE_DIM_FAST, width, depth, p)
        self.branch_K = _fnn(n_sensors, width, depth, p)
        self.branch_Ta = _fnn(n_sensors, width, depth, p)
        # Trunk input is t/T, not t in minutes. The paper's output domain is
        # the unit interval and its trunk is tuned for that; feeding 0-720
        # would put the trunk two orders of magnitude outside the range its
        # depth-2 width-200 configuration was reported for.
        self.trunk = _fnn(1, width, depth, p * out_dim)
        # Corollary 2: "we take an additional bias b", which makes the constant M
        # in the error bound smaller. One per output state.
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x0n: torch.Tensor, u_sensors: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        ns = self.n_sensors
        g = (self.branch_x0(x0n)
             * self.branch_K(u_sensors[:, :ns])
             * self.branch_Ta(u_sensors[:, ns:2 * ns]))          # (B, p)
        f = self.trunk(t / self.T_dom).view(-1, self.p, self.out_dim)          # (B, p, out)
        # S(.) in Eq. (16): sum over the p merged components, per output state.
        return (g.unsqueeze(-1) * f).sum(dim=1) + self.bias


class _MIONetBase(nn.Module):
    """Shared plumbing: input normalisation, IC mask, output scaling.

    None of this is from the MIONet paper and none of it is architecture. It is
    the C-11 protocol every matrix model gets (see `blocks.ic_mask`): the six
    states span orders of magnitude and the initial condition is an operator
    input, neither of which is true of the paper's GRF benchmarks where the inputs
    are O(1) and the IC is fixed at zero.
    """

    def __init__(self, T: float, x_mean, x_std):
        super().__init__()
        self.T = float(T)
        # SharedPhysicsTrainer reads this to size the physics residual;
        # every model driven by that trainer must expose it.
        self.state_dim = STATE_DIM_FAST
        self.register_buffer("tau", torch.tensor(float(tau_oil)))
        if x_mean is not None:
            xm = torch.tensor(x_mean, dtype=torch.float32)
            xs = torch.tensor(x_std, dtype=torch.float32)
            self.register_buffer("xm", xm)
            self.register_buffer("xs", xs)
            self.register_buffer("x_mean_TO", xm[:1].clone())
            self.register_buffer("x_std_TO", xs[:1].clone())
        # Per-state output scale, from the shared definition so this model and
        # `MonolithicFair` are scaled identically and a difference between them is
        # a difference in architecture rather than in output parameterisation.
        # An earlier naive `log(expm1(x))` here overflowed float32 for c_CO2
        # (x_std ~ 100, and the expression overflows near 88), initialising that
        # scale to inf and returning NaN on the first forward pass.
        self.output_scale_raw = per_state_output_scale_raw(x_std, STATE_DIM_FAST)

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def _norm(self, x0):
        return (x0 - self.xm) / self.xs if hasattr(self, "xm") else x0

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MIONetMonolithic(_MIONetBase):
    """MIONet predicting all six states directly (C-11 monolithic configuration)."""

    def __init__(self, n_sensors: int = N_SENSORS, width: int = MIONET_WIDTH,
                 depth: int = MIONET_DEPTH, p: int = MIONET_P, T: float = TW,
                 x_mean=None, x_std=None):
        super().__init__(T, x_mean, x_std)
        self.n_sensors = int(n_sensors)
        self.core = _MIONetCore(STATE_DIM_FAST, n_sensors, width, depth, p)

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        raw = self.core(self._norm(x0), u_sensors, t)
        return x0 + ic_mask(t, self.tau, self.T) * self.output_scale * raw


class MIONetInCascade(_MIONetBase):
    """MIONet predicting `theta_TO`; gases by the analytic Arrhenius quadrature.

    The one-variable companion to `MIONetMonolithic`. The cascade needs the
    thermal trajectory on a grid, not just at `t`, so the trunk is evaluated on
    the sensor grid as well — a batched extra forward of the trunk only, since the
    branches do not depend on `t`. That is cheap and is the structural reason an
    operator-learning architecture suits the cascade: the branch encoding is
    computed once and reused for every query point.
    """

    def __init__(self, n_sensors: int = N_SENSORS, width: int = MIONET_WIDTH,
                 depth: int = MIONET_DEPTH, p: int = MIONET_P, T: float = TW,
                 x_mean=None, x_std=None):
        super().__init__(T, x_mean, x_std)
        self.n_sensors = int(n_sensors)
        self.core = _MIONetCore(1, n_sensors, width, depth, p)
        self.cascade = GasCascade(T=T, n_sensors=n_sensors)
        grid = torch.linspace(0.0, float(T), self.n_sensors).view(-1, 1)
        self.register_buffer("t_grid", grid)

    def _theta_grid(self, x0n, u_sensors, x0_TO):
        """theta_TO on the sensor grid: branches once, trunk at every grid point."""
        B, ns = x0n.shape[0], self.n_sensors
        g = (self.core.branch_x0(x0n)
             * self.core.branch_K(u_sensors[:, :ns])
             * self.core.branch_Ta(u_sensors[:, ns:2 * ns]))       # (B, p)
        f = self.core.trunk(self.t_grid / self.T).view(ns, self.core.p)     # (ns, p)
        raw = torch.einsum("bp,np->bn", g, f) + self.core.bias     # (B, ns)
        phi = ic_mask(self.t_grid.view(1, -1), self.tau, self.T)
        return x0_TO + phi * self.output_scale[0] * raw

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        x0n = self._norm(x0)
        x0_TO, x0_gas = x0[:, 0:1], x0[:, 1:]
        raw = self.core(x0n, u_sensors, t)
        theta_TO = x0_TO + ic_mask(t, self.tau, self.T) * self.output_scale[0] * raw
        # Detached exactly as in CODOperator: the cascade is one-way by design, so
        # the gas loss must not send gradient into the thermal branch.
        theta_grid = self._theta_grid(x0n, u_sensors, x0_TO).detach()
        gases = self.cascade(t, u_sensors, x0_gas, theta_grid)
        return torch.cat([theta_TO, gases], dim=-1)


def mionet_predict(model, x0, u, t, theta_ss_grid=None):
    """Uniform predict signature, matching `cod_predict` and `mono_predict`."""
    return model(x0, u, t)


__all__ = ["MIONetMonolithic", "MIONetInCascade", "mionet_predict",
           "MIONET_DEPTH", "MIONET_WIDTH", "MIONET_P"]
