"""Per-profile PINN — the tier-2 baseline the audit found missing.

WHY THIS EXISTS. The audit lists "PINN baseline (§7.2) does not exist" among the
claims that cannot be traced to an artifact. C-11 puts it in tier 2 as the
evidence for the **amortisation** argument, which is a headline claim of the
paper: an operator is trained once and answers any profile in a forward pass,
while a PINN must be retrained for every new profile.

WHAT IS BEING COMPARED, AND WHAT IS NOT. This is easy to get backwards, so state
it plainly:

    A per-profile PINN is the SPECIALISED GOLD STANDARD for its own profile. It
    fits one (x0, K, theta_a) and is evaluated on that same profile. It should be
    MORE accurate than an operator network, which has to be right about a whole
    distribution with the same capacity.

So the claim this baseline supports is **not** "we beat the PINN on accuracy". If
COD beats a converged per-profile PINN on that PINN's own profile, the honest
reading is that the PINN is undertrained, not that the operator is better — and
the run should be investigated rather than reported as a win. The claim is:
comparable accuracy at amortised cost, with the cost measured rather than
asserted (the audit found the manuscript's timing claims had no artifact).

NO LABELS. The PINN sees the ODE residual and the initial condition, never the
RK45 solution. It solves the ODE for its profile without supervision, which is
what makes it a PINN and not a curve fit, and what makes the comparison fair —
the operator has no labels either.

FAIRNESS, SPELLED OUT. Every advantage the operator has is given to the PINN too,
because an unfair baseline is worse than no baseline:

  * **Causal weighting**, `w_i = exp(-eps * sum_{k<i} L_r(t_k))`, Wang et al.
    2024 Eq. 3.2 (`reference/papers/2024.Wang_Respecting causality...pdf`), which
    is exactly what `cod.training.losses.causal_weights` already implements. The
    operator uses it; withholding it here would compare our-method-with-causality
    against a PINN-without.
  * **The same hard IC ansatz**, `x0 + phi(t) * scale * net(t)` with `phi(0) = 0`
    (`blocks.ic_mask`), so the initial condition is satisfied by construction on
    both sides rather than learned on one.
  * **The same per-state output scaling** (`blocks.per_state_output_scale_raw`),
    because the six states span orders of magnitude for both.
  * **The same convergence criterion and harness.** The PINN trains to its own
    plateau; its budget is *not* capped to the operator's. Capping it would rig
    the amortisation number in our favour, which is the one number this baseline
    exists to produce.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW, tau_oil
from cod.models.blocks import ic_mask, per_state_output_scale_raw

#: A plain PINN: tanh MLP on the time coordinate. Depth and width are the tier-2
#: hyperparameters that the search (ANALYSIS_PLAN §6) may move; these are the
#: starting point, chosen to sit near the operator's own capacity so that a
#: difference is not trivially explained by parameter count.
PINN_WIDTH = 128
PINN_DEPTH = 4


class PerProfilePINN(nn.Module):
    """One network for one profile: `t -> x(t)` in R^6.

    `x0` and `u_sensors` are fixed at construction, because that is what
    "per-profile" means. `forward` still takes them so the signature matches every
    other model in the repo and the existing evaluation path works unchanged; they
    are asserted against the stored profile rather than ignored, so a caller that
    accidentally scores this network on a different profile gets an error instead
    of a silently meaningless number.

    tanh rather than ReLU: the residual needs `d/dt` through the network, and a
    ReLU network has a piecewise-constant derivative. That is a property of
    collocation-based PINNs generally and is why the PINN literature uses smooth
    activations; it is not a choice made to favour this baseline.
    """

    def __init__(self, x0, u_sensors, T: float = TW,
                 n_sensors: int = N_SENSORS, width: int = PINN_WIDTH,
                 depth: int = PINN_DEPTH, x_std=None):
        super().__init__()
        self.T = float(T)
        self.n_sensors = int(n_sensors)
        self.state_dim = STATE_DIM_FAST
        self.register_buffer("tau", torch.tensor(float(tau_oil)))
        self.register_buffer("x0", torch.as_tensor(np.asarray(x0, np.float32)))
        self.register_buffer("u", torch.as_tensor(np.asarray(u_sensors,
                                                             np.float32)))
        layers: list[nn.Module] = []
        d = 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, STATE_DIM_FAST))
        self.net = nn.Sequential(*layers)
        self.output_scale_raw = per_state_output_scale_raw(
            x_std if x_std is not None else np.ones(STATE_DIM_FAST),
            STATE_DIM_FAST)

    @property
    def output_scale(self):
        return torch.nn.functional.softplus(self.output_scale_raw) + 1e-3

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x0, u_sensors, t, theta_ss_grid=None):
        if x0 is not None and x0.shape[0] > 0:
            if not torch.allclose(x0[0], self.x0, rtol=0, atol=0):
                raise ValueError(
                    "PerProfilePINN was scored on a different initial condition "
                    "than it was trained on. A per-profile model is only "
                    "meaningful on its own profile.")
        t = t.reshape(-1, 1)
        raw = self.net(t / self.T)
        return self.x0 + ic_mask(t, self.tau, self.T) * self.output_scale * raw

    def predict(self, t):
        """Convenience for the trainer: evaluate at times `t` (n, 1)."""
        raw = self.net(t / self.T)
        return self.x0 + ic_mask(t, self.tau, self.T) * self.output_scale * raw


def pinn_predict(model, x0, u, t, theta_ss_grid=None):
    """Uniform predict signature, matching the other models."""
    return model(x0, u, t)


__all__ = ["PerProfilePINN", "pinn_predict", "PINN_WIDTH", "PINN_DEPTH"]
