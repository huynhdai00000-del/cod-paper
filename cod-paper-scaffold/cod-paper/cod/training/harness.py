"""Shared training harness.

Every model, including baselines, goes through this one loop. The audit found
that baselines were compared on unequal terms: one stopped at 171 of 25,000
epochs and was still reported, another trained with its causal weights
underflowed to exactly zero while the proposed model had them near one, and
equal epoch counts hid a 4.6x difference in wall clock.

This module makes those conditions visible and recorded rather than silent:

* a convergence criterion stated in the config, applied identically to every
  model;
* a wall-clock budget alongside the epoch budget, since equal epochs is not
  equal compute;
* pathology checks that run every epoch and are written into the result file,
  so an unfair comparison is detectable afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol


class Trainable(Protocol):
    """Minimal interface a model must satisfy to use this harness."""

    def train_step(self) -> dict:
        """Run one optimisation step.

        Returns a dict that must contain ``loss`` (float) and may contain any
        number of diagnostic scalars, which are recorded verbatim.
        """
        ...

    def validation_loss(self) -> float:
        ...

    def grad_norm(self) -> float:
        ...


@dataclass
class ConvergenceCriterion:
    """Stated in the config before any run. Applied identically to all models.

    Parameters
    ----------
    max_epochs
        Hard epoch ceiling.
    max_wall_seconds
        Hard wall-clock ceiling. Reached first for expensive models, which is
        the point: it makes the compute budget explicit rather than implied by
        an epoch count.
    patience
        Stop if validation loss has not improved by ``min_delta_rel`` for this
        many consecutive checks.
    min_delta_rel
        Relative improvement that counts as progress.
    check_every
        Epochs between validation checks.
    """

    max_epochs: int = 25_000
    max_wall_seconds: float = 3600.0
    patience: int = 20
    min_delta_rel: float = 1e-3
    check_every: int = 100


@dataclass
class PathologyReport:
    """Conditions that invalidate a comparison if left unreported."""

    causal_weight_min: float | None = None
    causal_weight_underflowed: bool = False
    clamp_hit_fraction: dict[str, float] = field(default_factory=dict)
    nonfinite_loss_epochs: list[int] = field(default_factory=list)
    grad_norm_final: float | None = None

    def warnings(self) -> list[str]:
        out = []
        if self.causal_weight_underflowed:
            out.append(
                "Causal weights underflowed to zero: this model was effectively "
                "trained on the early part of the window only. Comparing it "
                "against a model whose weights stayed near one is not a fair "
                "comparison."
            )
        for name, frac in self.clamp_hit_fraction.items():
            if frac > 0.05:
                out.append(
                    f"Clamp '{name}' active on {frac:.1%} of samples: the loss "
                    "is being evaluated at a clamped state, not the predicted one."
                )
        if self.nonfinite_loss_epochs:
            out.append(
                f"Non-finite loss at {len(self.nonfinite_loss_epochs)} epoch(s), "
                f"first at epoch {self.nonfinite_loss_epochs[0]}."
            )
        return out


@dataclass
class TrainingOutcome:
    """What actually happened, as opposed to what was budgeted."""

    converged: bool
    stop_reason: str
    epochs_reached: int
    epochs_budgeted: int
    wall_seconds: float
    wall_budgeted: float
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    loss_history: list[float]
    pathology: PathologyReport

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pathology_warnings"] = self.pathology.warnings()
        return d

    def is_fair_comparison_candidate(self) -> bool:
        """A model that did not converge, or trained under a pathology, must
        not be reported as a performance number without disclosure."""
        return self.converged and not self.pathology.warnings()


def train(
    model: Trainable,
    criterion: ConvergenceCriterion,
    pathology_hook: Callable[[int, dict], None] | None = None,
    log_every: int = 500,
) -> TrainingOutcome:
    """Run one training job under a stated convergence criterion.

    ``pathology_hook`` receives ``(epoch, step_metrics)`` each epoch and may
    mutate the report it closes over. The built-in checks below cover the
    conditions the audit found; add model-specific ones through the hook.
    """
    report = PathologyReport()
    history: list[float] = []

    best_val = float("inf")
    stale_checks = 0
    stop_reason = "budget_exhausted"
    converged = False

    t0 = time.monotonic()
    epoch = 0

    for epoch in range(1, criterion.max_epochs + 1):
        metrics = model.train_step()
        loss = float(metrics["loss"])
        history.append(loss)

        if not _is_finite(loss):
            report.nonfinite_loss_epochs.append(epoch)

        # Built-in pathology checks.
        wm = metrics.get("causal_weight_min")
        if wm is not None:
            wm = float(wm)
            report.causal_weight_min = wm
            if wm <= 0.0:
                report.causal_weight_underflowed = True

        for key, value in metrics.items():
            if key.startswith("clamp_frac_"):
                report.clamp_hit_fraction[key[len("clamp_frac_"):]] = float(value)

        if pathology_hook is not None:
            pathology_hook(epoch, metrics)

        elapsed = time.monotonic() - t0
        if elapsed > criterion.max_wall_seconds:
            stop_reason = "wall_clock_budget"
            break

        if epoch % criterion.check_every == 0:
            val = float(model.validation_loss())
            improved = val < best_val * (1.0 - criterion.min_delta_rel)
            if improved:
                best_val = val
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= criterion.patience:
                    stop_reason = "converged_plateau"
                    converged = True
                    break

        if epoch % log_every == 0:
            print(f"[train] epoch {epoch:>6d}  loss {loss:.4e}  {elapsed:6.0f}s")

    wall = time.monotonic() - t0
    final_val = float(model.validation_loss())
    report.grad_norm_final = float(model.grad_norm())

    if stop_reason == "budget_exhausted" and epoch >= criterion.max_epochs:
        # Reached the epoch ceiling without plateauing. Not the same as
        # converged, and must not be reported as if it were.
        stop_reason = "epoch_budget"

    outcome = TrainingOutcome(
        converged=converged,
        stop_reason=stop_reason,
        epochs_reached=epoch,
        epochs_budgeted=criterion.max_epochs,
        wall_seconds=wall,
        wall_budgeted=criterion.max_wall_seconds,
        final_train_loss=history[-1] if history else float("nan"),
        final_val_loss=final_val,
        best_val_loss=best_val,
        loss_history=history,
        pathology=report,
    )

    for w in report.warnings():
        print(f"[train] PATHOLOGY: {w}")
    if not converged:
        print(
            f"[train] NOT CONVERGED (stop_reason={stop_reason}). This result "
            "must be reported as non-converged, not as a performance number."
        )

    return outcome


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
