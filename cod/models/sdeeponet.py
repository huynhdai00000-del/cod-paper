"""S-DeepONet: a recurrent branch over the load history, adapted to a time domain.

REFERENCE. He, Kushwaha, Park, Koric, Abueidda, Jasiuk, "Sequential Deep Operator
Networks (S-DeepONet) for Predicting Full-field Solutions Under Time-dependent
Loads", Engineering Applications of Artificial Intelligence 127 (2024) 107258
(`reference/papers/2024.He_Sequential deep operator networks...pdf`). Not built
from memory.

WHY IT IS IN THE MATRIX. It is the only tier-1 architecture designed for
time-dependent forcing, which is this problem. Its motivating claim (§1) is that
"a FNN in the branch network does not retain the causality of input data", so a
load *history* should be encoded by a recurrent net rather than flattened into one
vector. Our branch input is a 12 h load and ambient history, so that claim applies
here directly, and leaving the architecture out would invite the obvious question.

ARCHITECTURE (paper §2.1.2, Fig. 2). GRU encoder-decoder branch: encoder
GRU(256) returning sequences then GRU(128) compressing to a vector, a repeat
vector for shape matching, decoder GRU(128) then GRU(256) returning sequences, a
time-distributed dense layer with linear activation, reshaped to the hidden
dimension. tanh in all GRU layers. Trunk is an FNN with ReLU. Merge is the
DeepONet dot product with a bias, `sum_i b_i t_i + beta` (§2.1.1, Fig. 1). The
paper sets the branch hidden dimension equal to the number of input time steps
(101 there); here that is `n_sensors`.

═══════════════════════════════════════════════════════════════════════════
THE PRIMARY ADAPTATION, AND IT IS A REAL DEPARTURE
═══════════════════════════════════════════════════════════════════════════

**In the published design the trunk takes spatial coordinates.** The trunk input
is `(x, y)`, the nodal coordinates of a 2-D mesh (§2.1.1: "The 2D problem geometry
is described by N nodes within the domain, assembled into a N x 2 matrix, and fed
to the trunk network"). Time enters *only* through the branch, as the load history
the GRU consumes. The network predicts the field **at the end of the load step** —
one spatial field per case, not a trajectory: "predicts an output field of shape
N x 1 with the field value ... at the end of the load step defined at each node".

**Our problem has no spatial domain.** The state is six scalars evolving in time,
and the operator must be queryable at arbitrary `t` within the window. The only
available mapping is to let the trunk take `t`.

**That makes both the branch and the trunk temporal, which the paper's design
never does.** It is a change of what the architecture is for, not a change of a
hyperparameter. Two specific consequences:

1. The paper's division of labour disappears. There, the branch encodes *when* and
   the trunk encodes *where*, and the dot product combines two different kinds of
   information. Here both encode time, so the merge combines a summary of the
   whole history with a query point inside that same history. The recurrent
   branch's advantage — causal encoding of a sequence — is still available, but it
   is no longer complementary to the trunk in the way the paper relies on.
2. The output changes from a single end-of-window field to a trajectory. The paper
   never asks its network to be accurate at intermediate times, so its reported
   accuracy is not evidence about this use.

**What this means for reading the comparison.** If S-DeepONet underperforms in the
matrix, this adaptation is a live candidate explanation and must be weighed
against "the architecture is unsuited to the problem". The two are not
distinguishable from the headline number alone. The in-cascade configuration
partially separates them — it asks only for `theta_TO`, so a failure there is not
about the gas states — but nothing in the matrix separates "recurrent branch with a
temporal trunk is the wrong pairing" from "recurrent branch is the wrong encoder
here". Stating that before the numbers exist is the point; a reader who is told
only the error rate cannot weigh it.

An alternative worth naming and rejecting: keep the paper's setup exactly by
predicting only the window-end state, and roll windows to build a trajectory. That
would be faithful, and it would also make S-DeepONet the only model in the matrix
unable to answer a query at arbitrary `t`, unable to enter the swing-fidelity
measurement (C-11's honesty protocol requires a resolved trajectory) and scored on
a different quantity from every other cell. The comparison would be lost to save
the fidelity, so the departure is taken deliberately and disclosed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW, tau_oil
from cod.models.blocks import ic_mask, per_state_output_scale_raw
from cod.models.cascade import GasCascade

#: Paper §2.1.2, Fig. 2: encoder 256 -> 128, decoder 128 -> 256.
SDON_ENC = (256, 128)
SDON_DEC = (128, 256)
#: Paper §2.1.2: the trunk of the GRU-DeepONet is an FNN of seven layers,
#: [2, 101, 101, 101, 101, 101, 101] — six Linear layers into the hidden dim.
SDON_TRUNK_LAYERS = 6
#: Paper: hidden dimension "identical to the number of time steps in the input
#: load vector". Ours is `n_sensors`, so it is derived rather than chosen.
SDON_CELL = "gru"


class _SequentialBranch(nn.Module):
    """GRU (or LSTM) encoder-decoder over the forcing history (paper Fig. 2/3).

    Input is the history as a sequence: `(B, n_sensors, n_feat)`. The paper's is a
    scalar load at 101 steps; ours is `(K, theta_a)` at `n_sensors` steps, plus the
    initial condition broadcast along the sequence. Broadcasting `x0` is a choice
    the paper does not face — its problem has a uniform fixed IC (`T(x,0) = T_0`),
    so there is no IC input at all. Recorded in PORT_LOG J-90.

    The paper offers GRU and LSTM and reports both; GRU is the default here
    because it is the cheaper of the two at nearly equal reported accuracy (0.06%
    vs 0.06% on heat transfer, 792k vs 1,039k parameters), and the matrix budget
    is wall clock.
    """

    def __init__(self, n_feat: int, hidden: int, cell: str = SDON_CELL):
        super().__init__()
        rnn = nn.GRU if cell.lower() == "gru" else nn.LSTM
        self.cell = cell.lower()
        e0, e1 = SDON_ENC
        d0, d1 = SDON_DEC
        # tanh is the GRU/LSTM default activation in torch and is what the paper
        # states ("All GRU layers use a tanh activation function").
        self.enc0 = rnn(n_feat, e0, batch_first=True)
        self.enc1 = rnn(e0, e1, batch_first=True)
        self.dec0 = rnn(e1, d0, batch_first=True)
        self.dec1 = rnn(d0, d1, batch_first=True)
        # "a time-distributed dense layer with linear activation is used to output
        # the results to the larger DeepONet architecture" — per-step, no bias-free
        # trick, linear.
        self.head = nn.Linear(d1, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L, _ = seq.shape
        h, _ = self.enc0(seq)
        h, _ = self.enc1(h)
        # The paper's second encoder block "compresses the output into 1D": take
        # the final step, then a repeat vector to restore the sequence length.
        z = h[:, -1, :]
        z = z.unsqueeze(1).expand(B, L, z.shape[-1])
        d, _ = self.dec0(z)
        d, _ = self.dec1(d)
        return self.head(d).squeeze(-1)          # (B, L) == (B, hidden)


def _trunk_fnn(in_dim: int, hidden: int, out_dim: int,
               n_layers: int = SDON_TRUNK_LAYERS) -> nn.Sequential:
    """FNN trunk with ReLU (paper §2.1.1, §2.1.2)."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _SDeepONetBase(nn.Module):
    """Shared protocol: normalisation, IC mask, per-state output scaling.

    As with FNO and MIONet, none of this is architecture and all of it is the
    C-11 protocol every matrix model receives (`blocks.ic_mask`).
    """

    def __init__(self, T: float, n_sensors: int, x_mean, x_std):
        super().__init__()
        self.T = float(T)
        self.n_sensors = int(n_sensors)
        self.state_dim = STATE_DIM_FAST
        self.register_buffer("tau", torch.tensor(float(tau_oil)))
        if x_mean is not None:
            xm = torch.tensor(x_mean, dtype=torch.float32)
            xs = torch.tensor(x_std, dtype=torch.float32)
            self.register_buffer("xm", xm)
            self.register_buffer("xs", xs)
            self.register_buffer("x_mean_TO", xm[:1].clone())
            self.register_buffer("x_std_TO", xs[:1].clone())
        self.output_scale_raw = per_state_output_scale_raw(x_std, STATE_DIM_FAST)
        t_grid = torch.linspace(0.0, float(T), self.n_sensors).view(-1, 1)
        self.register_buffer("t_grid", t_grid)

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def _norm(self, x0):
        return (x0 - self.xm) / self.xs if hasattr(self, "xm") else x0

    def _sequence(self, x0n, u_sensors):
        """The branch input: (K, theta_a) per step, with x0 broadcast alongside."""
        ns = self.n_sensors
        K = u_sensors[:, :ns].unsqueeze(-1)
        Ta = u_sensors[:, ns:2 * ns].unsqueeze(-1)
        x0_b = x0n.unsqueeze(1).expand(-1, ns, STATE_DIM_FAST)
        return torch.cat([K, Ta, x0_b], dim=-1)

    def _branch_dedup(self, x0n, u_sensors):
        """Run the recurrent branch once per distinct case, not once per row.

        The branch encodes `(x0, K, theta_a)` and does **not** depend on the query
        time. `ode_physics_loss_shared` evaluates the model at `n_col` collocation
        points by

            x0e = x0.unsqueeze(1).expand(B, n_col, n).reshape(B * n_col, n)

        so every case appears `n_col` times in contiguous rows. Recomputing a
        4-layer GRU over a 100-step sequence for each of those is pure waste — the
        same waste the rollout's shared `cyc_cache` removed, and for the same
        reason: the quantity does not depend on what varies across the repeats.
        Measured, it is what exhausted CPU memory at the config's
        `batch_size = 64, n_collocation = 60`, where activations scale as
        `B * n_col * n_sensors * 256`.

        Shrinking the batch was the alternative and is worse: at equal wall clock
        a smaller batch is a *different optimisation problem*, which would then
        need defending in the paper. This is exact instead of a compromise.

        Exactness. Runs are detected on `(x0, u)` — 206 numbers per row, trivial
        against the GRU — and a row starts a new group only when it differs from
        its predecessor, so the grouping is correct for any input where identical
        rows are contiguous, and degrades to computing every row when they are
        not. No approximation and no assumption about `n_col`.

        Not applicable to FNO, which carries batch normalisation: there the batch
        composition changes the output, so deduplicating rows would change the
        answer rather than save work. S-DeepONet has no normalisation layer.
        """
        key = torch.cat([x0n, u_sensors], dim=-1)
        if key.shape[0] > 1:
            diff = (key[1:] != key[:-1]).any(dim=-1)
            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=key.device),
                torch.nonzero(diff, as_tuple=False).squeeze(-1) + 1])
            if starts.numel() < key.shape[0]:
                gid = torch.cumsum(
                    torch.cat([torch.zeros(1, dtype=torch.long,
                                           device=key.device), diff.long()]), 0)
                b_unique = self.branch(
                    self._sequence(x0n[starts], u_sensors[starts]))
                return b_unique[gid]
        return self.branch(self._sequence(x0n, u_sensors))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SDeepONetMonolithic(_SDeepONetBase):
    """S-DeepONet predicting all six states (C-11 monolithic configuration)."""

    def __init__(self, n_sensors: int = N_SENSORS, cell: str = SDON_CELL,
                 trunk_layers: int = SDON_TRUNK_LAYERS, T: float = TW,
                 x_mean=None, x_std=None):
        super().__init__(T, n_sensors, x_mean, x_std)
        hd = self.n_sensors      # paper: hidden dim == number of input time steps
        self.hd = hd
        self.branch = _SequentialBranch(2 + STATE_DIM_FAST, hd, cell)
        self.trunk = _trunk_fnn(1, hd, hd * STATE_DIM_FAST, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(STATE_DIM_FAST))

    def _raw(self, x0n, u_sensors, t):
        b = self._branch_dedup(x0n, u_sensors)                       # (B, hd)
        # Trunk input is t/T. The paper's trunk takes nodal coordinates; ours
        # takes the query time, normalised to the unit interval.
        f = self.trunk(t / self.T).view(-1, self.hd, STATE_DIM_FAST)
        return (b.unsqueeze(-1) * f).sum(dim=1) + self.bias

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        raw = self._raw(self._norm(x0), u_sensors, t)
        return x0 + ic_mask(t, self.tau, self.T) * self.output_scale * raw


class SDeepONetInCascade(_SDeepONetBase):
    """S-DeepONet predicting `theta_TO`; gases by Arrhenius quadrature.

    The recurrent branch is evaluated once per case and reused at every grid
    point, which is the structural reason an operator architecture suits the
    cascade: the expensive encoding does not depend on the query time.
    """

    def __init__(self, n_sensors: int = N_SENSORS, cell: str = SDON_CELL,
                 trunk_layers: int = SDON_TRUNK_LAYERS, T: float = TW,
                 x_mean=None, x_std=None):
        super().__init__(T, n_sensors, x_mean, x_std)
        hd = self.n_sensors
        self.hd = hd
        self.branch = _SequentialBranch(2 + STATE_DIM_FAST, hd, cell)
        self.trunk = _trunk_fnn(1, hd, hd, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))
        self.cascade = GasCascade(T=T, n_sensors=n_sensors)

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        x0n = self._norm(x0)
        x0_TO, x0_gas = x0[:, 0:1], x0[:, 1:]
        b = self._branch_dedup(x0n, u_sensors)                        # (B, hd)

        f_q = self.trunk(t / self.T)                                  # (B, hd)
        raw_q = (b * f_q).sum(dim=-1, keepdim=True) + self.bias
        theta_TO = x0_TO + ic_mask(t, self.tau, self.T) * self.output_scale[0] * raw_q

        f_g = self.trunk(self.t_grid / self.T)                        # (ns, hd)
        raw_g = torch.einsum("bh,nh->bn", b, f_g) + self.bias
        phi_g = ic_mask(self.t_grid.view(1, -1), self.tau, self.T)
        theta_grid = x0_TO + phi_g * self.output_scale[0] * raw_g
        # Detached exactly as in CODOperator: the cascade is one-way by design.
        gases = self.cascade(t, u_sensors, x0_gas, theta_grid.detach())
        return torch.cat([theta_TO, gases], dim=-1)


def sdeeponet_predict(model, x0, u, t, theta_ss_grid=None):
    """Uniform predict signature, matching the other matrix models."""
    return model(x0, u, t)


__all__ = ["SDeepONetMonolithic", "SDeepONetInCascade", "sdeeponet_predict",
           "SDON_ENC", "SDON_DEC", "SDON_TRUNK_LAYERS", "SDON_CELL"]
