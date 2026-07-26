"""Dataset generation and RK45 ground truth.

PHASE 1 — FAITHFUL PORT.

Training set   n12 cell 0 L995-L1007 (`transformer_training_v57.npz`, seed 42)
Test set       n12 cell 3 L1901-L1943 (`evaluate_v44`, seed 999, N=100)
               n15 cell 2 L369-L391  (`evaluate_100`, seed 999, N=100)

The two test-set builders draw the same random numbers in the same order and so
produce the same 100 cases; they differ only in how the sensor interpolant is
clamped at the right edge of the window (`TW*0.9999` in n12, `TW*0.999` in
n15/n00). Both are available through `t_clip_frac` because gate 1 comes from the
first and gates 2-3 from the second, and the port must not silently pick one.

KNOWN GAP (audit B-5): this test set is not out of distribution. The TV waveform
(amp 0.20, period TW, phase pi/3, K_base ~ U(0.5,1.2)) sits inside every training
marginal. KNOWN GAP (audit M-8): 40% of the constant-K cases lie outside the
training load support — training draws K_base ~ U(0.5,1.2) while the CK test
draws K ~ U(0.4,1.4), so 20 of 50 CK cases fall in [0.40,0.50) u (1.20,1.40].
The tier labels in `cod/eval/metrics.py` exist so this is stated, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d as sci_interp1d

from cod.data.physics import (
    N_SENSORS,
    STATE_DIM_FAST,
    TW,
    fast_rhs_np,
)
from cod.data.profiles import N_IC, make_sensor_profile, sample_consistent_ic

TRAIN_FILE = "transformer_training_v57.npz"


# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════
def rk45_ground_truth(x0: np.ndarray, K_sensors: np.ndarray, Ta_sensors: np.ndarray,
                      t_eval: np.ndarray, T: float = TW,
                      rtol: float = 1e-8, atol: float = 1e-10,
                      t_clip_frac: float = 0.9999) -> np.ndarray:
    """Solve the reference ODE on a sensor-driven profile.

    Returns an array of shape (len(t_eval), 6).

    `t_clip_frac` reproduces the source's right-edge guard: the sensor
    interpolant is evaluated at `min(t, T*t_clip_frac)` so that the solver never
    asks for a time past the last sensor sample. n12 uses 0.9999, n15/n00 use
    0.999. Keep this explicit — it is the only difference between the two eval
    harnesses that produced the stored gate numbers.
    """
    grid = np.linspace(0, T, N_SENSORS)
    Kf = sci_interp1d(grid, K_sensors, kind="linear", fill_value="extrapolate")
    Taf = sci_interp1d(grid, Ta_sensors, kind="linear", fill_value="extrapolate")
    t_cap = T * t_clip_frac

    sol = solve_ivp(
        lambda t, x: fast_rhs_np(x, float(Kf(min(t, t_cap))), float(Taf(min(t, t_cap)))),
        [0, T], x0, method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
    )
    return sol.y.T


# ═══════════════════════════════════════════════════════════════════════════
# Training set
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TrainingSet:
    """The 8000 (IC, profile) pairs the checkpoints were trained on."""

    x0s: np.ndarray        # (N, 6)
    sensors: np.ndarray    # (N, 200) = [K(100), Ta(100)]
    x_mean: np.ndarray     # (6,)
    x_std: np.ndarray      # (6,)

    def __len__(self) -> int:
        return len(self.x0s)


def generate_training_set(n_ic: int = N_IC, seed: int = 42,
                          randomise_ambient_phase: bool = False) -> TrainingSet:
    """Regenerate the training set from scratch.

    n12 cell 0 L1002-L1006: one RandomState(42) draws all ICs first, then all
    profiles. Not interleaved — replaying it in any other order gives a different
    dataset. `x_std` carries the `+ 1e-8` the source adds.
    """
    rng = np.random.RandomState(seed)
    x0s = np.array([sample_consistent_ic(rng) for _ in range(n_ic)])
    sensors = np.array([
        make_sensor_profile(rng, randomise_ambient_phase=randomise_ambient_phase)
        for _ in range(n_ic)
    ])
    x_mean = x0s.mean(axis=0)
    x_std = x0s.std(axis=0) + 1e-8
    return TrainingSet(x0s=x0s, sensors=sensors, x_mean=x_mean, x_std=x_std)


def load_training_set(path: str | Path) -> TrainingSet:
    """Load the stored `transformer_training_v57.npz`.

    Use this, not `generate_training_set`, whenever a checkpoint is involved:
    `x_mean` / `x_std` are baked into the checkpoints as the `x_mean_TO`,
    `x_std_TO`, `xm` and `xs` buffers.
    """
    d = np.load(path)
    return TrainingSet(x0s=d["x0s"], sensors=d["sensors"],
                       x_mean=d["x_mean"], x_std=d["x_std"])


# ═══════════════════════════════════════════════════════════════════════════
# Test set
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TestCase:
    """One evaluation case. `kind` is 'CK' (constant K) or 'TV' (time-varying)."""

    idx: int
    kind: str
    x0: np.ndarray
    K_sensors: np.ndarray
    Ta_sensors: np.ndarray
    K_mean: float


def build_test_set(n_test: int = 100, seed: int = 999,
                   T: float = TW) -> list[TestCase]:
    """The seed-999 benchmark: first half constant K, second half time-varying.

    Draw order per case, which must not change:
        1. sample_consistent_ic(rng)                      (5 rng calls)
        2. CK:  rng.uniform(0.4, 1.4)  then rng.uniform(15, 45)
           TV:  rng.uniform(0.5, 1.2)  then rng.uniform(15, 40)

    Identical in n12 `evaluate_v44` and n15/n00 `evaluate_100`, which is why all
    three gates score the same 100 cases.

    Note the ambient phase of pi/3 on the TV branch. Training fixes the ambient
    phase at 0 (see `make_sensor_profile`); Phase 2 fix 4 closes that gap.
    """
    rng = np.random.RandomState(seed)
    tau = np.linspace(0, T, N_SENSORS)
    cases: list[TestCase] = []

    for k in range(n_test):
        x0_k = sample_consistent_ic(rng)
        if k < n_test // 2:
            K_k = rng.uniform(0.4, 1.4)
            Ta_k = rng.uniform(15, 45)
            K_s = np.full(N_SENSORS, K_k, dtype=np.float32)
            Ta_s = np.full(N_SENSORS, Ta_k, dtype=np.float32)
            kind = "CK"
        else:
            K_k = rng.uniform(0.5, 1.2)
            K_s = (K_k + 0.2 * np.sin(2 * np.pi * tau / T)).clip(0.3, 1.5).astype(np.float32)
            Ta_k = rng.uniform(15, 40)
            Ta_s = (Ta_k + 5 * np.sin(2 * np.pi * tau / T + np.pi / 3)).astype(np.float32)
            kind = "TV"

        cases.append(TestCase(idx=k, kind=kind, x0=x0_k, K_sensors=K_s,
                              Ta_sensors=Ta_s, K_mean=float(K_s.mean())))
    return cases


def solve_test_set(cases: list[TestCase], n_eval: int = 50, T: float = TW,
                   t_clip_frac: float = 0.9999) -> np.ndarray:
    """RK45 ground truth for every case: (n_cases, n_eval, 6)."""
    t_eval = np.linspace(0, T, n_eval)
    out = np.zeros((len(cases), n_eval, STATE_DIM_FAST))
    for i, c in enumerate(cases):
        out[i] = rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval,
                                   T=T, t_clip_frac=t_clip_frac)
    return out


__all__ = [
    "TRAIN_FILE", "TrainingSet", "TestCase",
    "rk45_ground_truth", "generate_training_set", "load_training_set",
    "build_test_set", "solve_test_set",
]
