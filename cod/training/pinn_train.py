"""Trainer for the per-profile PINN (C-11 tier 2, the amortisation baseline).

Separate from `train.py` because the shape of the problem is genuinely different:
`CODTrainer` and `SharedPhysicsTrainer` batch over *profiles*, and a per-profile
PINN has exactly one. Its only batch dimension is collocation time. Folding it
into either of those would mean a batch axis that is always 1 and a set of
branches that never fire.

Everything the two share is imported rather than restated: `fast_rhs_torch`,
`interp_sensors`, `causal_weights`, `EpsilonSchedule`, and the same
`ConvergenceCriterion` through the same `harness.train`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cod.data.physics import N_SENSORS, STATE_DIM_FAST, TW, fast_rhs_torch
from cod.models.blocks import interp_sensors
from cod.training.losses import CAUSAL_WEIGHT_FLOOR, causal_weights
from cod.training.train import EpsilonSchedule


class PerProfilePINNTrainer:
    """One PINN, one profile. Satisfies the `Trainable` protocol.

    The residual is the same one every other model is trained on —
    `dx/dt - fast_rhs_torch(x, u(t))` — with the same causal weighting over
    chunks in time. The differences from `SharedPhysicsTrainer` are only those
    forced by there being a single profile:

      * collocation points are drawn fresh each step over `[t_min, T]` rather
        than fixed, since with one profile there is no batch to provide variety
        and a fixed grid would let the network memorise those points;
      * `validation_loss` is the residual on a **held-out set of collocation
        times**, disjoint from the training draw. A PINN has no held-out data, so
        held-out *points* are the only honest analogue, and the plateau criterion
        needs something it did not just descend on.

    NOT clamped. `SharedPhysicsTrainer` truncates the state before forming the
    RHS; that exists because a randomly initialised operator can produce wild
    values across a whole distribution. A per-profile PINN with a hard IC ansatz
    starts at `x0` and stays near it, and adding a clamp would import the very
    defect that `30_scalar500_clamp.py` found — evaluating the residual at a state
    the model did not predict. If a clamp turns out to be needed the run is
    reported as needing it rather than getting it silently.
    """

    def __init__(self, model, x0, sensors, device=None, lr: float = 1e-3,
                 n_col: int = 256, n_chunks: int = 8, seed: int = 0,
                 t_min_frac: float = 0.005,
                 causal_log_space: bool = True,
                 causal_floor: float = CAUSAL_WEIGHT_FLOOR,
                 causal_schedule_shared: bool = True,
                 T: float = TW, n_sensors: int = N_SENSORS):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.T = float(T)
        self.ns = int(n_sensors)
        self.n_col = int(n_col)
        self.n_chunks = int(n_chunks)
        self.t_min = float(t_min_frac) * self.T
        self.causal_log_space = bool(causal_log_space)
        self.causal_floor = float(causal_floor)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.eps_schedule = EpsilonSchedule(shared=causal_schedule_shared)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.sensors = torch.as_tensor(
            np.asarray(sensors, np.float32), device=self.device).unsqueeze(0)
        self.x0 = torch.as_tensor(np.asarray(x0, np.float32),
                                  device=self.device).unsqueeze(0)
        # Held-out collocation times, fixed once, never trained on.
        self._val_t = self._draw_times(self.n_col, val=True)

    def _draw_times(self, n: int, val: bool = False) -> torch.Tensor:
        """Sorted collocation times, log-spaced draws over [t_min, T].

        Log spacing matches `ode_physics_loss_shared`'s
        `exp(linspace(log(0.005 T), log T, n_col))`: the thermal response is
        fastest early, so uniform sampling under-resolves exactly where the
        residual is largest. Sorted because the causal weighting is defined over
        chunks in increasing time.
        """
        g = torch.Generator(device="cpu").manual_seed(12345) if val else self.gen
        u = torch.rand(n, generator=g)
        lo, hi = math.log(self.t_min), math.log(self.T)
        t = torch.exp(lo + (hi - lo) * u).sort().values
        return t.to(self.device).unsqueeze(-1)

    def _residual(self, t: torch.Tensor):
        t = t.detach().requires_grad_(True)
        xp = self.model.predict(t)
        dxdt = torch.cat([
            torch.autograd.grad(xp[:, i].sum(), t, create_graph=True,
                                retain_graph=True)[0]
            for i in range(STATE_DIM_FAST)
        ], dim=1)
        u_t = interp_sensors(self.sensors, t, self.T, self.ns)
        f_rhs = fast_rhs_torch(xp, u_t)
        return (dxdt - f_rhs) ** 2, xp

    def _weighted(self, t: torch.Tensor):
        res, xp = self._residual(t)
        n_use = (t.shape[0] // self.n_chunks) * self.n_chunks
        r2c = res[:n_use].reshape(self.n_chunks, -1, STATE_DIM_FAST).mean(dim=1)
        r2m = r2c.mean(dim=-1).unsqueeze(0)                # (1, n_chunks)
        w, wm = causal_weights(r2m, self.eps_schedule.eps,
                               log_space=self.causal_log_space,
                               floor=self.causal_floor)
        loss = (w * r2m).sum() / (w.sum() + 1e-20)
        return loss, wm, xp

    def train_step(self) -> dict:
        self.model.train()
        t = self._draw_times(self.n_col)
        loss, wm, xp = self._weighted(t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.eps_schedule.observe(wm)
        return {"loss": float(loss.detach()),
                "causal_weight_min": float(wm),
                "eps_causal": float(self.eps_schedule.eps),
                # Reported for parity with the other trainers. A per-profile PINN
                # is not clamped, so this measures whether one would have been
                # needed, not whether one was applied.
                "clamp_frac_would_hit_state_hi": float(
                    (xp.detach() > 500).any(dim=-1).float().mean())}

    def validation_loss(self) -> float:
        self.model.eval()
        loss, _, _ = self._weighted(self._val_t)
        return float(loss.detach())

    def grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return total ** 0.5


__all__ = ["PerProfilePINNTrainer"]
