"""The benchmark NMAE as the source computes it — for reproduction only.

PHASE 1 — FAITHFUL PORT of:
    compute_metrics_one   n12 cell 3 L1538
    evaluate_v44          n12 cell 3 L1901   (gate 1: Table 2)
    metrics_one           n15 cell 2 L157 | n00 cell 4 L97
    evaluate_100          n15 cell 2 L369 | n00 cell 4 L213  (gates 2 and 3)

This module exists to reproduce the stored numbers, not to measure anything well.
`cod/eval/metrics.py` is the metric the paper should report.

KNOWN DEFECT (audit M-3): the normalisation denominator is
`max(x_true.max() - x_true.min(), 1e-4)`. Over a 12 h window the gas states barely
move, so that floor binds on 38% of C2H2 cases, 19% of C2H4, 10% of H2. The
resulting figures are not physical error measures: the monolithic baseline's
"34,558% NMAE on acetylene" is an absolute error of 0.23 ppm, which is 0.7% of the
IEC 60599 attention level of 35 ppm. Its genuinely large error is thermal, 13.9 degC.

The battery section of the same repository already does this correctly, with a
floor of 1e-12 and absolute units reported alongside (audit M-14). Whenever these
numbers are quoted, quote `metrics.py`'s absolute MAE and floor-hit rate too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from cod.data.generate import TestCase, build_test_set, rk45_ground_truth
from cod.data.physics import STATE_DIM_FAST, STATE_NAMES_FAST, TW

NMAE_DENOM_FLOOR = 1e-4   # audit M-3: this is the defect, kept for reproduction


def compute_metrics_one(x_true: np.ndarray, x_pred: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-state NMAE and NRMSE for one case.

    NMAE_s  = mean|err| / max(range(x_true_s), 1e-4)
    NRMSE_s = ||err|| / max(||x_true_s - mean(x_true_s)||, 1e-8)
    """
    nmae = np.zeros(STATE_DIM_FAST)
    nrmse = np.zeros(STATE_DIM_FAST)
    for s in range(STATE_DIM_FAST):
        err = x_true[:, s] - x_pred[:, s]
        mae = np.abs(err).mean()
        variation = x_true[:, s].max() - x_true[:, s].min()
        nmae[s] = mae / max(variation, NMAE_DENOM_FLOOR)
        x_dev = x_true[:, s] - x_true[:, s].mean()
        nrmse[s] = np.linalg.norm(err) / max(np.linalg.norm(x_dev), 1e-8)
    return nmae, nrmse


@dataclass
class BenchmarkResult:
    """Everything the source prints, plus the absolute figures it does not."""

    label: str
    nmae: np.ndarray          # (n_cases, 6)
    nrmse: np.ndarray         # (n_cases, 6)
    is_tv: np.ndarray         # (n_cases,) bool
    mae_abs: np.ndarray       # (n_cases, 6) in physical units
    denom: np.ndarray         # (n_cases, 6) normalisation denominators used
    cases: list[TestCase] = field(default_factory=list)

    @property
    def per_state_pct(self) -> np.ndarray:
        return self.nmae.mean(axis=0) * 100

    @property
    def per_state_std_pct(self) -> np.ndarray:
        return self.nmae.std(axis=0) * 100

    @property
    def overall_pct(self) -> float:
        return float(self.nmae.mean() * 100)

    @property
    def ck_pct(self) -> float:
        return float(self.nmae[~self.is_tv].mean() * 100)

    @property
    def tv_pct(self) -> float:
        return float(self.nmae[self.is_tv].mean() * 100)

    @property
    def n_within_10pct(self) -> int:
        return int((self.nmae.mean(axis=1) < 0.10).sum())

    @property
    def median_pct(self) -> float:
        return float(np.median(self.nmae.mean(axis=1)) * 100)

    def floor_hit_frac(self) -> np.ndarray:
        """Fraction of cases per state where the 1e-4 denominator floor bound."""
        return (self.denom <= NMAE_DENOM_FLOOR).mean(axis=0)

    def summary(self) -> str:
        lines = [f"=== {self.label} (N={len(self.nmae)}) ==="]
        for i, nm in enumerate(STATE_NAMES_FAST):
            lines.append(f"  {nm:10s}: {self.per_state_pct[i]:8.1f}% "
                         f"+/- {self.per_state_std_pct[i]:7.1f}%")
        lines.append(f"  Overall: {self.overall_pct:.1f}% | CK: {self.ck_pct:.1f}% "
                     f"| TV: {self.tv_pct:.1f}% | <10%: {self.n_within_10pct}/"
                     f"{len(self.nmae)}")
        return "\n".join(lines)

    def physical_summary(self) -> str:
        """What audit M-3 says should be reported instead of, or beside, NMAE."""
        units = ["degC", "ppm", "ppm", "ppm", "ppm", "ppm"]
        floor = self.floor_hit_frac()
        lines = [f"--- {self.label}: absolute error, physical units ---"]
        for i, (nm, unit) in enumerate(zip(STATE_NAMES_FAST, units)):
            lines.append(
                f"  {nm:10s}: MAE {self.mae_abs[:, i].mean():12.4g} {unit:4s}"
                f"  median denom {np.median(self.denom[:, i]):10.4g}"
                f"  floor hit {floor[i]:5.1%}")
        return "\n".join(lines)


@torch.no_grad()
def evaluate(model, predict_fn, cases: list[TestCase] | None = None,
             label: str = "model", n_eval: int = 50, T: float = TW,
             t_clip_frac: float = 0.9999, device=None) -> BenchmarkResult:
    """Score a model on the seed-999 benchmark exactly as the source does.

    `t_clip_frac` selects the eval harness: 0.9999 is n12's `evaluate_v44`
    (gate 1), 0.999 is n15/n00's `evaluate_100` (gates 2 and 3). The two draw the
    same 100 cases and differ only in the sensor interpolant's right-edge guard
    (PORT_LOG J-10).
    """
    if cases is None:
        cases = build_test_set(n_test=100, seed=999, T=T)
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n = len(cases)
    all_nmae = np.zeros((n, STATE_DIM_FAST))
    all_nrmse = np.zeros((n, STATE_DIM_FAST))
    mae_abs = np.zeros((n, STATE_DIM_FAST))
    denom = np.zeros((n, STATE_DIM_FAST))
    is_tv = np.zeros(n, dtype=bool)

    t_eval = np.linspace(0, T, n_eval)
    t_q = torch.tensor(t_eval, dtype=torch.float32, device=device).unsqueeze(-1)

    for k, c in enumerate(cases):
        x_true = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval,
                                   T=T, t_clip_frac=t_clip_frac)
        s_k = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                           dtype=torch.float32, device=device).unsqueeze(0)
        x0_t = torch.tensor(c.x0, dtype=torch.float32, device=device).unsqueeze(0)
        x_pred = predict_fn(model,
                            x0_t.expand(n_eval, -1).contiguous(),
                            s_k.expand(n_eval, -1).contiguous(),
                            t_q).cpu().numpy()

        nmae, nrmse = compute_metrics_one(x_true, x_pred)
        all_nmae[k] = nmae
        all_nrmse[k] = nrmse
        mae_abs[k] = np.abs(x_true - x_pred).mean(axis=0)
        denom[k] = np.maximum(x_true.max(axis=0) - x_true.min(axis=0),
                              NMAE_DENOM_FLOOR)
        is_tv[k] = c.kind == "TV"

    return BenchmarkResult(label=label, nmae=all_nmae, nrmse=all_nrmse,
                           is_tv=is_tv, mae_abs=mae_abs, denom=denom,
                           cases=list(cases))


def outliers(result: BenchmarkResult, threshold: float = 0.10) -> list[dict]:
    """Cases above `threshold` overall, as n12 cell 4 lists them."""
    out = []
    per_case = result.nmae.mean(axis=1)
    for k, c in enumerate(result.cases):
        if per_case[k] > threshold:
            out.append({
                "idx": c.idx, "type": c.kind, "x0_TO": float(c.x0[0]),
                "K": c.K_mean, "overall": float(per_case[k]),
                "nmae": result.nmae[k].copy(),
            })
    return sorted(out, key=lambda d: -d["overall"])


__all__ = ["NMAE_DENOM_FLOOR", "compute_metrics_one", "BenchmarkResult",
           "evaluate", "outliers"]
