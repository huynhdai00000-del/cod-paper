"""Training loops, restructured to satisfy the `Trainable` protocol.

PHASE 1 — FAITHFUL PORT of two genuinely different loops:

    CODTrainer            n12 cell 1 L1028 (`train_v34`)
                          trained transformer_pideepOnet_v57.pt
    SharedPhysicsTrainer  n15 cell 2 L334 | n00 cell 4 L167 (`train_physics`)
                          trained every monolithic baseline and every
                          sweep_{cod,mono_fair}_p*.pt

They are not two spellings of one loop. The differences are enumerated in
`losses.ode_physics_loss_shared`; the headline COD model and the capacity-sweep
COD models were trained by different loops with different collocation grids and
different state clamps.

The restructuring: the source runs `for ep in trange(n_epochs)` with the
optimiser, the schedule, the adaptive weights and the epsilon schedule all inside
one function body. `harness.train` owns the loop instead, so a trainer exposes
`train_step()`, `validation_loss()` and `grad_norm()`. The per-epoch arithmetic is
unchanged; only who calls it changed.

Two things the harness adds that the source did not have, both required by
README rules 4 and 5:

  * a validation split, drawn from the training distribution with its own fixed
    seed and never touched by the optimiser. The source has no validation set at
    all, so "converged" was never checked against anything — every run simply ran
    its epoch budget out. That is exactly how a baseline came to be reported at
    171 of 25,000 epochs.
  * clamp and causal-weight diagnostics on every step, so the `wm = 0.000` versus
    `wm = 0.988` asymmetry is recorded rather than discovered later in a log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim

from cod.data.physics import STATE_DIM_FAST
from cod.training.losses import (
    CAUSAL_WEIGHT_FLOOR,
    ode_physics_loss,
    ode_physics_loss_shared,
)


@dataclass
class EpsilonSchedule:
    """The causal-weighting epsilon schedule, n12 cell 1 L1037-L1039.

    PHASE 2 FIX 3 (audit B-1), the second half of the fix. v57 started `eps_causal`
    at 0.01 and multiplied it by 1.10 once `wm` had stayed above 0.50 for 200
    consecutive epochs, capped at 50.0 — a schedule driven by **each model's own**
    `wm`, so two models being compared did not follow the same epsilon trajectory.
    COD's weights stayed high, so its epsilon climbed all the way to the 50.0 cap;
    Mono Fair's underflowed to zero, so its epsilon froze near the start. The two
    arms were optimising measurably different objectives, which is what makes that
    comparison unusable however the weights themselves are computed.

    `shared=True` (default) advances epsilon purely on elapsed epochs: every
    `patience_needed` epochs, unconditionally. Two models trained for the same
    number of epochs then see the identical epsilon trajectory regardless of how
    their residuals behave, so the objective is a property of the protocol rather
    than of the model.

    `shared=False` restores the v57 wm-driven behaviour for comparison.

    Fixing the weights without fixing the schedule would not have been enough:
    with a floor in place Mono Fair's `wm` would no longer hit zero, but it would
    still sit far below COD's, so the two epsilon trajectories would still
    diverge. Both halves are needed.
    """

    eps: float = 0.01
    patience_needed: int = 200
    factor: float = 1.10
    eps_max: float = 50.0
    wm_threshold: float = 0.50
    shared: bool = True
    _patience: int = field(default=0, repr=False)
    _epoch: int = field(default=0, repr=False)

    def observe(self, wm: float) -> None:
        self._epoch += 1
        if self.shared:
            # Model-independent: advance on elapsed epochs alone.
            if self._epoch % self.patience_needed == 0 and self.eps < self.eps_max:
                self.eps = min(self.eps * self.factor, self.eps_max)
            return
        if wm > self.wm_threshold:
            self._patience += 1
        else:
            self._patience = 0
        if self._patience >= self.patience_needed and self.eps < self.eps_max:
            self.eps = min(self.eps * self.factor, self.eps_max)
            self._patience = 0


class _BatchSource:
    """Random batches from the training set, plus a held-out validation split.

    The validation indices are carved out with a dedicated RandomState and are
    never returned by `train_batch`, so the convergence criterion is evaluated on
    data the optimiser has not seen. The source has no such split (PORT_LOG J-13).
    """

    def __init__(self, x0s, sensors, device, batch_size: int = 128,
                 val_fraction: float = 0.05, seed: int = 0):
        n = len(x0s)
        rs = np.random.RandomState(seed)
        perm = rs.permutation(n)
        n_val = max(1, int(round(n * val_fraction)))
        self.val_idx = np.sort(perm[:n_val])
        self.train_idx = np.sort(perm[n_val:])

        self.x0s = torch.as_tensor(x0s, dtype=torch.float32, device=device)
        self.sensors = torch.as_tensor(sensors, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.device = device
        self._train_idx_t = torch.as_tensor(self.train_idx, device=device)
        self._val_idx_t = torch.as_tensor(self.val_idx, device=device)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed + 1)

    def train_batch(self, n: int | None = None):
        n = n or self.batch_size
        pick = torch.randint(0, len(self.train_idx), (n,), generator=self._gen)
        idx = self._train_idx_t[pick.to(self.device)]
        return self.x0s[idx], self.sensors[idx]

    def val_batch(self, n: int | None = None):
        n = min(n or self.batch_size, len(self.val_idx))
        idx = self._val_idx_t[:n]
        return self.x0s[idx], self.sensors[idx]


class CODTrainer:
    """`train_v34`: plain Adam, cosine schedule, per-state adaptive weights.

    Source: n12 cell 1 L1028. AugLag was dropped in v34 in favour of direct Adam.

    Per step:
        L_states, wm = ode_physics_loss(..., return_per_state=True)
        Lv    = per-state loss values (detached)
        lraw  = clamp(Lv / mean(Lv), 0.1, 50)
        lam   = 0.70 * lam + 0.30 * lraw
        lam[0]  = clamp(lam[0], min=5.0)
        lam[1:] = clamp(lam[1:], min=0.0)
        loss  = sum_s lam[s] * L_states[s]
        clip_grad_norm_(1.0); opt.step(); sch.step()

    KNOWN DEFECT (audit M-4): `lam_state[0]` is initialised at 5.0 and floored at
    5.0, so the "learned" thermal weight of ~5 that the manuscript reports is the
    floor, not a fitted value. And `lam_state[1:]` multiplies the five gas terms,
    which carry exactly zero gradient because of the intended cascade detach
    (audit M-1) — so those five weights have no effect on training at all.
    """

    def __init__(self, model, x0s, sensors, device=None, lr: float = 1e-3,
                 n_fb: int = 128, n_col: int = 80, n_chunks: int = 5,
                 max_epochs: int = 25_000, seed: int = 0,
                 val_fraction: float = 0.05,
                 causal_log_space: bool = True,
                 causal_floor: float = CAUSAL_WEIGHT_FLOOR,
                 causal_schedule_shared: bool = True):
        self.device = device or next(model.parameters()).device
        self.model = model
        self.n_fb = n_fb
        self.n_col = n_col
        self.n_chunks = n_chunks
        self.n_states = model.state_dim

        self.data = _BatchSource(x0s, sensors, self.device, batch_size=n_fb,
                                 val_fraction=val_fraction, seed=seed)
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.sch = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max_epochs, eta_min=1e-5)

        self.lam_state = torch.ones(self.n_states, device=self.device)
        self.lam_state[0] = 5.0
        # PHASE 2 FIX 3
        self.causal_log_space = causal_log_space
        self.causal_floor = causal_floor
        self.eps_schedule = EpsilonSchedule(shared=causal_schedule_shared)
        self._last_grad_norm = float("nan")

    def train_step(self) -> dict:
        self.model.train()
        x0_b, s_b = self.data.train_batch()
        diag: dict = {}

        L_states, wm = ode_physics_loss(
            self.model, x0_b, s_b, n_col=self.n_col, n_chunks=self.n_chunks,
            eps_causal=self.eps_schedule.eps, return_per_state=True,
            diagnostics=diag, causal_log_space=self.causal_log_space,
            causal_floor=self.causal_floor,
        )

        with torch.no_grad():
            Lv = torch.tensor([l.item() for l in L_states], device=self.device)
            Lm = Lv.mean() + 1e-20
            lraw = torch.clamp(Lv / (Lm + 1e-20), 0.1, 50.0)
            self.lam_state = 0.70 * self.lam_state + 0.30 * lraw
            self.lam_state[0] = self.lam_state[0].clamp(min=5.0)     # audit M-4
            self.lam_state[1:] = self.lam_state[1:].clamp(min=0.0)

        loss = sum(self.lam_state[s] * L_states[s] for s in range(self.n_states))
        f_val = float(loss.item())

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.opt.step()
        self.sch.step()
        self.eps_schedule.observe(wm)

        out = {
            "loss": f_val,
            "causal_weight_min": wm,
            "eps_causal": self.eps_schedule.eps,
            "lr": float(self.opt.param_groups[0]["lr"]),
            "grad_norm": self._last_grad_norm,
        }
        for s in range(self.n_states):
            out[f"lam_state_{s}"] = float(self.lam_state[s])
        for s, l in enumerate(L_states):
            out[f"L_state_{s}"] = float(l.item())
        out.update({k: v for k, v in diag.items() if k.startswith("clamp_frac_")})
        return out

    @torch.no_grad()
    def _val_no_grad_guard(self):
        return None

    def validation_loss(self) -> float:
        """Residual loss on the held-out split.

        Cannot run under `no_grad`: the residual needs `autograd.grad` through t.
        The optimiser is never stepped here and gradients are discarded.
        """
        self.model.eval()
        x0_v, s_v = self.data.val_batch()
        L_states, _ = ode_physics_loss(
            self.model, x0_v, s_v, n_col=self.n_col, n_chunks=self.n_chunks,
            eps_causal=self.eps_schedule.eps, return_per_state=True,
            causal_log_space=self.causal_log_space,
            causal_floor=self.causal_floor)
        val = float(sum(self.lam_state[s] * L_states[s]
                        for s in range(self.n_states)).item())
        self.model.zero_grad(set_to_none=True)
        return val

    def grad_norm(self) -> float:
        return self._last_grad_norm


class SharedPhysicsTrainer:
    """`train_physics`: the loop that trained the baselines and the sweep.

    Source: n15 cell 2 L334, n00 cell 4 L167. Differs from `CODTrainer` in the
    collocation grid, the RHS state clamp and the adaptive-weight target — see
    `losses.ode_physics_loss_shared` for the enumeration.

    Per step:
        r2c, w, wm = ode_physics_loss_shared(...)
        Lv    = [r2c[:,:,s].mean() for s]          # plain mean, not weighted
        lraw  = clamp(Lv / mean(Lv), 0.1, 50)
        lam   = 0.70 * lam + 0.30 * lraw
        lam[0] = clamp(lam[0], min=5.0)            # lam[1:] NOT floored here
        loss  = sum_s lam[s] * (w * r2c[:,:,s]).sum() / (w.sum() + 1e-20)
    """

    def __init__(self, model, predict_fn, x0s, sensors, device=None,
                 lr: float = 1e-3, n_fb: int = 128, n_col: int = 60,
                 n_chunks: int = 5, max_epochs: int = 25_000, seed: int = 0,
                 val_fraction: float = 0.05,
                 causal_log_space: bool = True,
                 causal_floor: float = CAUSAL_WEIGHT_FLOOR,
                 causal_schedule_shared: bool = True):
        self.device = device or next(model.parameters()).device
        self.model = model
        self.predict_fn = predict_fn
        self.n_fb = n_fb
        self.n_col = n_col
        self.n_chunks = n_chunks
        self.n_states = getattr(model, "state_dim", STATE_DIM_FAST)

        self.data = _BatchSource(x0s, sensors, self.device, batch_size=n_fb,
                                 val_fraction=val_fraction, seed=seed)
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.sch = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max_epochs, eta_min=1e-5)

        self.lam = torch.ones(self.n_states, device=self.device)
        self.lam[0] = 5.0
        # PHASE 2 FIX 3
        self.causal_log_space = causal_log_space
        self.causal_floor = causal_floor
        self.eps_schedule = EpsilonSchedule(shared=causal_schedule_shared)
        self._last_grad_norm = float("nan")

    def _loss_terms(self, x0_b, s_b, diag: dict | None = None):
        r2c, w, wm = ode_physics_loss_shared(
            self.model, self.predict_fn, x0_b, s_b, n_col=self.n_col,
            n_chunks=self.n_chunks, eps_causal=self.eps_schedule.eps,
            diagnostics=diag, causal_log_space=self.causal_log_space,
            causal_floor=self.causal_floor)
        with torch.no_grad():
            Lv = torch.tensor([r2c[:, :, s].mean().item()
                               for s in range(self.n_states)], device=self.device)
            Lm = Lv.mean() + 1e-20
            lraw = torch.clamp(Lv / (Lm + 1e-20), 0.1, 50.0)
            self.lam = 0.7 * self.lam + 0.3 * lraw
            self.lam[0] = self.lam[0].clamp(min=5.0)
        loss = sum(self.lam[s] * (w * r2c[:, :, s]).sum() / (w.sum() + 1e-20)
                   for s in range(self.n_states))
        return loss, wm

    def train_step(self) -> dict:
        self.model.train()
        x0_b, s_b = self.data.train_batch()
        diag: dict = {}
        loss, wm = self._loss_terms(x0_b, s_b, diag)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.opt.step()
        self.sch.step()
        self.eps_schedule.observe(wm)

        out = {
            "loss": float(loss.item()),
            "causal_weight_min": wm,
            "eps_causal": self.eps_schedule.eps,
            "lr": float(self.opt.param_groups[0]["lr"]),
            "grad_norm": self._last_grad_norm,
        }
        for s in range(self.n_states):
            out[f"lam_state_{s}"] = float(self.lam[s])
        out.update({k: v for k, v in diag.items() if k.startswith("clamp_frac_")})
        return out

    def validation_loss(self) -> float:
        self.model.eval()
        x0_v, s_v = self.data.val_batch()
        loss, _ = self._loss_terms(x0_v, s_v)
        val = float(loss.item())
        self.model.zero_grad(set_to_none=True)
        return val

    def grad_norm(self) -> float:
        return self._last_grad_norm


__all__ = ["EpsilonSchedule", "CODTrainer", "SharedPhysicsTrainer"]
