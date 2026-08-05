"""Fourier Neural Operator, adapted from a PDE benchmark to a 6-state ODE.

REFERENCE. Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart,
Anandkumar, "Fourier Neural Operator for Parametric Partial Differential
Equations", ICLR 2021 (`reference/papers/2021.Li_FOURIER NEURAL OPERATOR...pdf`).
Nothing here is built from memory; every structural choice cites a section.

ARCHITECTURE (paper §3-4, Fig. 2). Lift the input to a channel space with `P`,
apply four Fourier layers

    v_{t+1} = sigma( W v_t + F^{-1}( R . F(v_t) ) )                    (paper Eq. 2, 4)

and project back with `Q`. `R` is a complex tensor acting on the lowest `k_max`
Fourier modes and zeroing the rest (Eq. 5); `W` is a local linear transform
applied in physical space.

WHAT THE DOMAIN IS HERE. The paper's domains are spatial (Burgers 1-d, Darcy 2-d,
Navier-Stokes 2-d+time). This problem has no space: the operator maps a load and
ambient history on a 12 h window to a six-state trajectory on that same window.
The domain is therefore **time**, and this is a 1-d FNO over `t` with the paper's
own 1-d hyperparameters (`k_max = 16`, `d_v = 64`, §5).

PERIODICITY, AND WHAT IT COSTS. The FFT treats the window as periodic and a 12 h
thermal window is not: `theta_TO(0) != theta_TO(T)` in general, because the window
is a slice of a 24 h day at a random phase (fix 7). The paper addresses exactly
this in §5.5:

    "Traditional Fourier methods work only with periodic boundary conditions.
     However, the Fourier neural operator does not have this limitation. This is
     due to the linear transform W (the bias term) which keeps the track of
     non-periodic boundary."

So the design already carries the answer: the spectral branch supplies the smooth,
resolved part and `W` — which never enters the FFT — carries whatever the periodic
extension cannot represent. The paper supports this empirically on Darcy flow and
on the time domain of Navier-Stokes, both non-periodic.

The cost is real and is not hidden by that argument. The spectral branch still
imposes a periodic extension, so it sees a discontinuity of size
`theta_TO(T) - theta_TO(0)` at the window edge and pays Gibbs ringing there, which
`W` must cancel. Two consequences worth stating before any number exists: error
should be worst at the window endpoints, and the burden on `W` grows with the
endpoint mismatch, i.e. with load swing — which is the regime the Jensen argument
cares about. `domain_padding` is provided, defaulting to **off**, because padding
the domain to make it artificially periodic is a later refinement of the reference
implementation and not in this paper; turning it on is a deviation and must be
recorded as one (see PORT_LOG J-90).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW, tau_oil
from cod.models.analytic_baseline import AnalyticBaseline
from cod.models.blocks import ic_mask, per_state_output_scale_raw
from cod.models.cascade import GasCascade

#: Paper §5, the 1-d configuration: "We set k_max,j = 16, d_v = 64 for the 1-d
#: problem". Our domain is 1-d (time), so these are the paper's values and not a
#: choice of ours.
FNO_MODES = 16
FNO_WIDTH = 64
#: Paper §5: "stacking four Fourier integral operator layers".
FNO_LAYERS = 4


class SpectralConv1d(nn.Module):
    """`F^{-1}(R . F(v))` on the lowest `modes` frequencies (paper Def. 3, Eq. 5).

    `R` is stored as a real tensor of shape (in, out, modes, 2) rather than a
    complex parameter so that optimiser state and checkpoint loading behave
    identically to every other module in the repo. Modes above `modes` are
    dropped, which is the truncation the paper's Eq. (5) performs; the paper notes
    the activations between layers still recover higher frequencies (§5.5,
    "Spectral analysis").
    """

    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, 2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, n)
        B, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        k = min(self.modes, x_ft.shape[-1])
        w = torch.view_as_complex(self.weight[:, :, :k].contiguous())
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        # einsum over the input channel, per retained mode: paper Eq. (5).
        out_ft[:, :, :k] = torch.einsum("bik,iok->bok", x_ft[:, :, :k], w)
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FourierLayer(nn.Module):
    """One update `v -> sigma(W v + K(v))` (paper Eq. 2 with Eq. 4 for K).

    `W` is a width-1 convolution, i.e. a per-point linear map — the paper's "local
    linear transform". It is the term §5.5 credits with handling non-periodic
    boundaries, and it is deliberately outside the FFT path.

    Batch normalisation follows the paper's §5 ("with the ReLU activation as well
    as batch normalization"). `activate=False` on the last layer follows the
    reference implementation's convention of not activating into the projection.
    """

    def __init__(self, width: int, modes: int, activate: bool = True):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, 1)
        self.norm = nn.BatchNorm1d(width)
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.spectral(x) + self.local(x))
        return torch.relu(y) if self.activate else y


class _FNOTrunk(nn.Module):
    """Lift, four Fourier layers, project — the whole operator on the time grid.

    Input channels, all sampled on the `n_sensors` time grid:
      `K(t)`, `theta_a(t)`      the forcing, which is what varies per case
      `x0` broadcast (6)        the initial condition, constant along the window
      `t / T`                   the positional coordinate the paper's grid carries

    Broadcasting `x0` as constant channels is the standard way an FNO receives a
    non-field conditioning input, and it is the only option that keeps the operator
    a function-to-function map on one domain. It costs six channels that carry no
    spatial information; the alternative — a separate branch network — is MIONet's
    design, not FNO's, and mixing them would make the comparison meaningless.
    """

    def __init__(self, out_dim: int, n_sensors: int = N_SENSORS,
                 width: int = FNO_WIDTH, modes: int = FNO_MODES,
                 n_layers: int = FNO_LAYERS, T: float = TW,
                 domain_padding: int = 0):
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.T = float(T)
        self.out_dim = int(out_dim)
        self.domain_padding = int(domain_padding)
        in_ch = 2 + STATE_DIM_FAST + 1
        self.lift = nn.Linear(in_ch, width)
        self.layers = nn.ModuleList([
            FourierLayer(width, modes, activate=(i < n_layers - 1))
            for i in range(n_layers)
        ])
        # Paper §3: "project back to the target dimension by a neural network Q".
        self.project = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                     nn.Linear(128, out_dim))
        grid = torch.linspace(0.0, 1.0, self.n_sensors).view(1, -1, 1)
        self.register_buffer("grid", grid)

    def forward(self, x0: torch.Tensor, u_sensors: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        ns = self.n_sensors
        K = u_sensors[:, :ns].unsqueeze(-1)
        Ta = u_sensors[:, ns:2 * ns].unsqueeze(-1)
        x0_b = x0.unsqueeze(1).expand(B, ns, STATE_DIM_FAST)
        a = torch.cat([K, Ta, x0_b, self.grid.expand(B, ns, 1)], dim=-1)
        v = self.lift(a).permute(0, 2, 1)               # (B, width, ns)
        if self.domain_padding:
            v = torch.nn.functional.pad(v, (0, self.domain_padding))
        for layer in self.layers:
            v = layer(v)
        if self.domain_padding:
            v = v[..., :ns]
        return self.project(v.permute(0, 2, 1))         # (B, ns, out_dim)


def _interp_time(grid_vals: torch.Tensor, t: torch.Tensor, T: float
                 ) -> torch.Tensor:
    """Linear interpolation of a (B, ns, C) grid at per-sample times t (B, 1).

    FNO is a function-to-function map: it produces the whole trajectory on the
    grid, and a query at an arbitrary `t` is an interpolation of that. This is a
    real difference from COD, which evaluates a trunk at `t` directly, and it has a
    consequence for the physics loss: `d/dt` through a linear interpolant is
    piecewise constant, with 100 pieces over a 720 min window, i.e. 7.2 min
    resolution against `tau_oil = 150` min. The thermal dynamics are resolved at
    that spacing, but the residual sees a staircase derivative rather than a smooth
    one. Recorded in PORT_LOG J-90 as a cost of the adaptation, not hidden.
    """
    ns = grid_vals.shape[1]
    C = grid_vals.shape[-1]
    pos = (t.squeeze(-1) / T).clamp(0.0, 1.0) * (ns - 1)
    lo = pos.floor().long().clamp(0, ns - 2)
    frac = (pos - lo.float()).unsqueeze(-1)
    idx = lo.view(-1, 1, 1).expand(-1, 1, C)
    v_lo = torch.gather(grid_vals, 1, idx).squeeze(1)
    v_hi = torch.gather(grid_vals, 1, idx + 1).squeeze(1)
    return v_lo * (1 - frac) + v_hi * frac


class FNOMonolithic(nn.Module):
    """FNO predicting all six states directly.

    The C-11 "monolithic" configuration: no cascade, no analytic baseline. The
    network must produce the gas concentrations itself, exactly as the existing
    `MonolithicFair` baseline does, so a failure here is attributable to the
    architecture plus the absence of the cascade — two variables, which is why the
    in-cascade configuration below exists alongside it.
    """

    def __init__(self, n_sensors: int = N_SENSORS, width: int = FNO_WIDTH,
                 modes: int = FNO_MODES, n_layers: int = FNO_LAYERS,
                 T: float = TW, x_mean=None, x_std=None,
                 domain_padding: int = 0, use_baseline: bool = False):
        super().__init__()
        self.T = float(T)
        # SharedPhysicsTrainer reads this to size the physics residual;
        # every model driven by that trainer must expose it.
        self.state_dim = STATE_DIM_FAST
        self.n_sensors = int(n_sensors)
        self.register_buffer("tau", torch.tensor(float(tau_oil)))
        self.trunk = _FNOTrunk(STATE_DIM_FAST, n_sensors, width, modes,
                               n_layers, T, domain_padding)
        # C-11 protocol, identical for every matrix model — see `blocks.ic_mask`.
        # Without the mask this model satisfied the initial condition only to
        # within 0.38 degC at t=0, which the smoke test caught; a baseline that
        # has to learn `x(0) = x0` while COD gets it by construction would fail
        # for a reason that is not its architecture.
        self.output_scale_raw = per_state_output_scale_raw(x_std, STATE_DIM_FAST)
        # Factorial baseline factor. `AnalyticBaseline` is parameter-free, so
        # `n_parameters()` is identical with and without it and the factor is
        # one variable rather than also a capacity change. PORT_LOG J-92.
        self.baseline = (AnalyticBaseline(T=T, n_sensors=n_sensors)
                         if use_baseline else None)
        if x_mean is not None:
            self.register_buffer("x_mean_TO",
                                 torch.tensor([x_mean[0]], dtype=torch.float32))
            self.register_buffer("x_std_TO",
                                 torch.tensor([x_std[0]], dtype=torch.float32))

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _anchor(self, x0, u_sensors, t, theta_ss_grid=None):
        """What the correction is added to, per state.

        Without the baseline this is the constant `x0`, which is the published
        form. With it, `theta_TO` is anchored on the IEC analytic solution `H(t)`
        and the gases stay anchored on their own initial values, because there is
        no analytic baseline for a gas concentration — the cascade is what plays
        that role, and this is the *monolithic* configuration.
        """
        anchor = x0
        if self.baseline is not None:
            H = self.baseline(x0[:, 0:1], u_sensors, t, theta_ss_grid)
            anchor = torch.cat([H, x0[:, 1:]], dim=-1)
        return anchor

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        raw = _interp_time(self.trunk(x0, u_sensors), t, self.T)
        anchor = self._anchor(x0, u_sensors, t, theta_ss_grid)
        return anchor + ic_mask(t, self.tau, self.T) * self.output_scale * raw


class FNOInCascade(nn.Module):
    """FNO predicting `theta_TO` only; the gases follow by Arrhenius quadrature.

    The C-11 one-variable test. If `FNOMonolithic` fails and this does not, the
    cascade is what the monolithic configuration was missing and the architecture
    is not at fault — which is the evidence a reviewer will ask for.

    The cascade is `cod.models.cascade.GasCascade`, the same analytic quadrature
    `CODOperator` uses, carrying no parameters. `n_parameters()` therefore counts
    only the FNO, so this configuration is not credited with capacity it does not
    have.
    """

    def __init__(self, n_sensors: int = N_SENSORS, width: int = FNO_WIDTH,
                 modes: int = FNO_MODES, n_layers: int = FNO_LAYERS,
                 T: float = TW, x_mean=None, x_std=None,
                 domain_padding: int = 0, use_baseline: bool = False):
        super().__init__()
        self.T = float(T)
        self.n_sensors = int(n_sensors)
        # SharedPhysicsTrainer reads this to size the physics residual; every
        # model driven by that trainer must expose it.
        self.state_dim = STATE_DIM_FAST
        self.register_buffer("tau", torch.tensor(float(tau_oil)))
        self.trunk = _FNOTrunk(1, n_sensors, width, modes, n_layers, T,
                               domain_padding)
        self.cascade = GasCascade(T=T, n_sensors=n_sensors)
        self.output_scale_raw = per_state_output_scale_raw(x_std, STATE_DIM_FAST)
        # Factorial baseline factor. `AnalyticBaseline` is parameter-free, so
        # `n_parameters()` is identical with and without it and the factor is
        # one variable rather than also a capacity change. PORT_LOG J-92.
        self.baseline = (AnalyticBaseline(T=T, n_sensors=n_sensors)
                         if use_baseline else None)
        t_grid = torch.linspace(0.0, float(T), self.n_sensors).view(1, -1, 1)
        self.register_buffer("t_grid", t_grid)
        if x_mean is not None:
            self.register_buffer("x_mean_TO",
                                 torch.tensor([x_mean[0]], dtype=torch.float32))
            self.register_buffer("x_std_TO",
                                 torch.tensor([x_std[0]], dtype=torch.float32))

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        x0_TO = x0[:, 0:1]
        x0_gas = x0[:, 1:]
        # The IC mask is applied on the grid too, so the trajectory the cascade
        # integrates starts at x0 exactly rather than only the queried point.
        raw_grid = self.trunk(x0, u_sensors)                          # (B, ns, 1)
        phi_grid = ic_mask(self.t_grid, self.tau, self.T)
        if self.baseline is None:
            anchor_grid = x0_TO.unsqueeze(1)                          # (B, 1, 1)
        else:
            # H over the whole grid, because the quadrature integrates the
            # trajectory rather than sampling it at the query time. Anchoring only
            # the queried point would leave the cascade integrating a trajectory
            # the model does not predict.
            anchor_grid = self.baseline.on_grid(
                x0_TO, u_sensors, self.t_grid.view(-1, 1),
                theta_ss_grid).unsqueeze(-1)                          # (B, ns, 1)
        theta_grid = anchor_grid + phi_grid * self.output_scale[0] * raw_grid
        theta_TO = _interp_time(theta_grid, t, self.T)
        # `.detach()` mirrors CODOperator: the cascade is one-way, so no gradient
        # flows from the gas loss back into the thermal branch. Keeping it makes
        # this a test of the cascade rather than of a different training signal.
        gases = self.cascade(t, u_sensors, x0_gas,
                             theta_grid.squeeze(-1).detach())
        return torch.cat([theta_TO, gases], dim=-1)


def fno_predict(model, x0, u, t, theta_ss_grid=None):
    """Uniform predict signature, matching `cod_predict` and `mono_predict`."""
    return model(x0, u, t)


__all__ = ["FNOMonolithic", "FNOInCascade", "SpectralConv1d", "FourierLayer",
           "fno_predict", "FNO_MODES", "FNO_WIDTH", "FNO_LAYERS"]
