"""Error metrics, physical units first.

The audit found that normalising error by per-case signal range produced
figures like "34,558% NMAE on acetylene" that correspond to an absolute error
of 0.23 ppm, which is 0.7% of the IEC 60599 diagnostic threshold. The
denominator hit its floor on 38% of C2H2 cases because the gas states barely
move over a 12 h window.

This module therefore reports absolute error in named physical units as the
primary metric, and normalised error only alongside the fraction of cases
where the denominator was degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class StateSpec:
    """One state of the system, with the units its error should be read in."""

    name: str
    unit: str
    engineering_threshold: float | None = None
    threshold_note: str = ""


# Thresholds are the accuracy a practitioner would need, not a target the model
# is tuned against. Sources belong in the manuscript; where a value is assumed
# rather than sourced, say so.
TRANSFORMER_STATES = (
    StateSpec("theta_TO", "degC", 2.0, "identification uncertainty of "
              "Delta_theta_oil_R from an IEC 60076-2 heat-run report"),
    StateSpec("c_H2", "ppm", 2.0, "typical DGA laboratory repeatability"),
    StateSpec("c_C2H2", "ppm", 1.0, "typical DGA laboratory repeatability"),
    StateSpec("c_C2H4", "ppm", 2.0, "typical DGA laboratory repeatability"),
    StateSpec("c_CO", "ppm", 5.0, "typical DGA laboratory repeatability"),
    StateSpec("c_CO2", "ppm", 10.0, "typical DGA laboratory repeatability"),
    StateSpec("DP", "DP units", None, ""),
)

BATTERY_STATES = (
    StateSpec("T_cell", "K", 1.0, ""),
    StateSpec("L_SEI", "nm", None, ""),
)


@dataclass
class StateMetrics:
    """Per-state error. Never aggregate these blindly across states: the states
    of a one-way coupled system carry different physical meaning and differ in
    magnitude by orders of magnitude."""

    name: str
    unit: str
    mae: float
    rmse: float
    max_abs_error: float
    within_threshold_frac: float | None
    nmae_mean: float
    nmae_median: float
    denominator_median: float
    denominator_floor_hit_frac: float
    n_cases: int

    def to_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> str:
        s = f"{self.name}: MAE {self.mae:.4g} {self.unit}"
        if self.within_threshold_frac is not None:
            s += f" ({self.within_threshold_frac:.0%} within threshold)"
        if self.denominator_floor_hit_frac > 0.01:
            s += (
                f"  [NMAE {self.nmae_median:.2%} median, but denominator hit the "
                f"floor on {self.denominator_floor_hit_frac:.0%} of cases: read "
                "the absolute figure, not the normalised one]"
            )
        else:
            s += f"  [NMAE {self.nmae_median:.2%} median]"
        return s


def evaluate_state(
    pred: np.ndarray,
    true: np.ndarray,
    spec: StateSpec,
    denom_floor: float | None = None,
) -> StateMetrics:
    """Compute metrics for one state across a set of test cases.

    Parameters
    ----------
    pred, true
        Arrays of shape ``(n_cases, n_timesteps)``.
    spec
        The state being evaluated, carrying its physical unit.
    denom_floor
        Floor applied to the normalisation denominator. If ``None``, it is set
        to a value far below any physically meaningful variation so that it
        effectively never binds, and the floor-hit fraction reports whether
        that assumption held. Do not raise this to make a metric look better.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 2:
        raise ValueError("expected arrays of shape (n_cases, n_timesteps)")

    abs_err = np.abs(pred - true)
    per_case_mae = abs_err.mean(axis=1)

    variation = true.max(axis=1) - true.min(axis=1)
    if denom_floor is None:
        # Six orders of magnitude below the median variation, so it binds only
        # when a case is genuinely degenerate.
        median_var = float(np.median(variation))
        denom_floor = max(median_var * 1e-6, np.finfo(float).tiny)

    floor_hit = variation < denom_floor
    denom = np.maximum(variation, denom_floor)
    per_case_nmae = per_case_mae / denom

    within = None
    if spec.engineering_threshold is not None:
        within = float((per_case_mae <= spec.engineering_threshold).mean())

    return StateMetrics(
        name=spec.name,
        unit=spec.unit,
        mae=float(per_case_mae.mean()),
        rmse=float(np.sqrt((abs_err ** 2).mean())),
        max_abs_error=float(abs_err.max()),
        within_threshold_frac=within,
        nmae_mean=float(per_case_nmae.mean()),
        nmae_median=float(np.median(per_case_nmae)),
        denominator_median=float(np.median(variation)),
        denominator_floor_hit_frac=float(floor_hit.mean()),
        n_cases=int(pred.shape[0]),
    )


def evaluate_system(
    pred: dict[str, np.ndarray],
    true: dict[str, np.ndarray],
    specs=TRANSFORMER_STATES,
) -> dict[str, StateMetrics]:
    out = {}
    for spec in specs:
        if spec.name not in pred:
            continue
        out[spec.name] = evaluate_state(pred[spec.name], true[spec.name], spec)
    return out


def report(metrics: dict[str, StateMetrics], tier: str = "") -> str:
    """Human-readable summary. Always states which test tier it refers to,
    because in-distribution, parameter-extrapolation and out-of-family results
    must never be merged into a single number."""
    header = f"Test tier: {tier}" if tier else "Test tier: UNSPECIFIED (say which)"
    lines = [header, "-" * len(header)]
    lines.extend(m.headline() for m in metrics.values())

    degenerate = [m.name for m in metrics.values()
                  if m.denominator_floor_hit_frac > 0.05]
    if degenerate:
        lines.append("")
        lines.append(
            "Note: " + ", ".join(degenerate) + " have near-constant ground truth "
            "over the evaluation window. Normalised error for these states is "
            "not a physical error measure. Report the absolute figure."
        )
    return "\n".join(lines)
