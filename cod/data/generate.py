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
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d as sci_interp1d

from cod.data.physics import (
    N_SENSORS,
    STATE_DIM_FAST,
    TW,
    fast_rhs_np,
)
from cod.data.profiles import N_IC, make_sensor_profile, sample_consistent_ic
from cod.data.realistic import build_realistic_set
from cod.data.steady_state import true_fixed_point_torch

TRAIN_FILE = "transformer_training_v57.npz"


def steady_state_on_grid(sensors: np.ndarray,
                         n_sensors: int = N_SENSORS) -> np.ndarray:
    """theta_ss on the sensor grid for every profile, (N, n_sensors), float32.

    Phase 2 fix 1 replaced a closed-form formula with a contraction solve, which
    made a training epoch 5.1x more expensive. The fixed point depends only on
    (K, theta_a) — fixed inputs per sample — so it is solved once here and stored
    with the dataset instead of being re-solved every forward pass.

    Computed with `true_fixed_point_torch` in float32, the same function and dtype
    the model uses, so the cached values are bit-identical to what the model would
    have computed. `audit_port/scripts/11_check_ss_cache.py` asserts that.

    Safe to cache because these grid values carry no gradient: they reach the output
    only through `F_cum` in `_ode_baseline` and through `dm`/`dr` in
    `build_trunk_feats`, none of which is differentiated. The query-time theta_ss,
    which does need a t-gradient, is still solved live.
    """
    K = torch.tensor(sensors[:, :n_sensors], dtype=torch.float32)
    Ta = torch.tensor(sensors[:, n_sensors:2 * n_sensors], dtype=torch.float32)
    with torch.no_grad():
        return true_fixed_point_torch(K, Ta).numpy().astype(np.float32)


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
    theta_ss: np.ndarray | None = None   # (N, 100) cached steady state

    def __len__(self) -> int:
        return len(self.x0s)

    def ensure_theta_ss(self) -> np.ndarray:
        """Cached theta_ss on the sensor grid, computing it once if absent."""
        if self.theta_ss is None:
            self.theta_ss = steady_state_on_grid(self.sensors)
        return self.theta_ss


def generate_realistic_training_set(n_ic: int, seed: int,
                                    params) -> TrainingSet:
    """Training set from the operationally realistic sampler (`realistic.py`).

    This is the default path since the sampler was put on the config: every knob
    comes from the hashed `distribution.sampler.params` block via
    `RealisticParams.from_config`, so there is no parameter the frozen hash does
    not cover.

    Unlike `generate_training_set` below, the day is drawn first and the initial
    condition is derived from it, which is the whole point of the sampler — the
    IC is the periodic state of that day's own load pattern rather than an
    independent draw (audit M-9).
    """
    x0s, sensors = build_realistic_set(n_ic, seed, params)
    x_mean = x0s.mean(axis=0)
    x_std = x0s.std(axis=0) + 1e-8
    return TrainingSet(x0s=x0s, sensors=sensors, x_mean=x_mean, x_std=x_std,
                       theta_ss=steady_state_on_grid(sensors))


def generate_training_set(n_ic: int = N_IC, seed: int = 42,
                          randomise_ambient_phase: bool = True,
                          steady_state=None, clip_step: bool = True) -> TrainingSet:
    """DEPRECATED for new work: the v57-era sampler, kept to reproduce v57.

    **Do not point a new experiment at this.** Audit M-9 is against it: it draws
    every IC before any profile, so `theta_TO(0)` is
    `steady_state(K, theta_a) + U(-30, 30)` with `K` and `theta_a` unrelated to
    the load profile that then drives the window. `generate_realistic_training_set`
    is the path for anything whose numbers go in the paper, and
    `configs/example_cod_seed1.yaml` selects it.

    This path survives for exactly one reason: `configs/v57_faithful.yaml` and the
    Phase 1 gates have to reproduce `transformer_training_v57.npz` byte for byte.
    Its ranges are therefore **deliberately hardcoded** in
    `cod/data/profiles.py` (`make_sensor_profile`, `sample_consistent_ic`) and are
    deliberately *not* configurable. Making them configurable would let an edit to
    a YAML file silently change what "reproduces v57" means, which is the opposite
    of what a reproduction gate is for. They are a frozen historical artifact, not
    a second source of truth competing with the config.

    n12 cell 0 L1002-L1006: one RandomState(42) draws all ICs first, then all
    profiles. Not interleaved — replaying it in any other order gives a different
    dataset. `x_std` carries the `+ 1e-8` the source adds.

    To reproduce `transformer_training_v57.npz` exactly, pass
    `steady_state=formula_A`, `randomise_ambient_phase=False` and
    `clip_step=False`. The defaults are the Phase 2 fixed behaviour and produce a
    different dataset again — neither is the realistic sampler.
    """
    rng = np.random.RandomState(seed)
    x0s = np.array([sample_consistent_ic(rng, steady_state=steady_state)
                    for _ in range(n_ic)])
    sensors = np.array([
        make_sensor_profile(rng, randomise_ambient_phase=randomise_ambient_phase,
                            clip_step=clip_step)
        for _ in range(n_ic)
    ])
    x_mean = x0s.mean(axis=0)
    x_std = x0s.std(axis=0) + 1e-8
    return TrainingSet(x0s=x0s, sensors=sensors, x_mean=x_mean, x_std=x_std,
                       theta_ss=steady_state_on_grid(sensors))


def load_training_set(path: str | Path) -> TrainingSet:
    """Load the stored `transformer_training_v57.npz`.

    Use this, not `generate_training_set`, whenever a checkpoint is involved:
    `x_mean` / `x_std` are baked into the checkpoints as the `x_mean_TO`,
    `x_std_TO`, `xm` and `xs` buffers.
    """
    d = np.load(path)
    return TrainingSet(x0s=d["x0s"], sensors=d["sensors"],
                       x_mean=d["x_mean"], x_std=d["x_std"],
                       theta_ss=(d["theta_ss"] if "theta_ss" in d.files else None))


# ═══════════════════════════════════════════════════════════════════════════
# Test set
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TestCase:
    """One evaluation case. `kind` is 'CK' (constant K) or 'TV' (time-varying).

    `family` names the sampler family the case came from where there is one. The
    v57 benchmark has no families — its two halves *are* CK and TV — so it leaves
    this None. The realistic sampler does, and losing it would make it impossible
    to say which load families a stratified result is really about.
    """

    idx: int
    kind: str
    x0: np.ndarray
    K_sensors: np.ndarray
    Ta_sensors: np.ndarray
    K_mean: float
    family: str | None = None


def build_test_set(n_test: int = 100, seed: int = 999, T: float = TW,
                   steady_state=None) -> list[TestCase]:
    """The seed-999 benchmark: first half constant K, second half time-varying.

    **This is the v57-era benchmark and it is not in the distribution of anything
    trained on the realistic sampler.** It consults no config: the IC comes from
    `profiles.sample_consistent_ic` (audit M-9) and both waveforms are hardcoded
    below at a 720 min load period, which is the N-6 defect fix 7 removed from
    training. `audit_port/TEST_SET_PROVENANCE.md` measures the gap on nine axes.
    Use `build_realistic_test_set` for `T1_in_distribution`; this stays as the
    Phase 1 reproduction benchmark and as a deliberate out-of-family tier.

    Draw order per case, which must not change:
        1. sample_consistent_ic(rng)                      (5 rng calls)
        2. CK:  rng.uniform(0.4, 1.4)  then rng.uniform(15, 45)
           TV:  rng.uniform(0.5, 1.2)  then rng.uniform(15, 40)

    Identical in n12 `evaluate_v44` and n15/n00 `evaluate_100`, which is why all
    three gates score the same 100 cases.

    Note the ambient phase of pi/3 on the TV branch. Training fixes the ambient
    phase at 0 (see `make_sensor_profile`); Phase 2 fix 4 closes that gap.

    `steady_state` reaches `sample_consistent_ic`. Pass `formula_A` for the v57
    test set; the default is the Phase 2 fixed behaviour, which shifts every
    theta_TO(0) and every gas IC and therefore changes what the benchmark measures.
    """
    rng = np.random.RandomState(seed)
    tau = np.linspace(0, T, N_SENSORS)
    cases: list[TestCase] = []

    for k in range(n_test):
        x0_k = sample_consistent_ic(rng, steady_state=steady_state)
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


#: Half peak-to-peak of the in-window load below which a case is called constant.
#: 5e-3 in K, which at the sampler's operating point is well under 0.1 degC of
#: hot-spot swing — far below anything the Jensen argument can resolve.
CK_THRESHOLD = 5e-3


def build_realistic_test_set(n_test: int = 100, seed: int = 999,
                             params=None, T: float = TW,
                             n_sensors: int = N_SENSORS) -> list[TestCase]:
    """The in-distribution benchmark: the frozen sampler at a held-out seed.

    This is what `T1_in_distribution` has to mean once the training data comes
    from `build_realistic_set`. `build_test_set` below cannot serve that role and
    the reason is not a detail — it calls `profiles.sample_consistent_ic`, which
    draws `theta_TO(0)` from a load unrelated to the profile (audit M-9), and it
    hardcodes its own two waveforms at a 720 min load period. Measured against the
    frozen sampler in `audit_port/TEST_SET_PROVENANCE.md`, seven of nine
    distributional axes fall outside training support, including 37% of cases
    whose mean hot-spot is outside the band the sampler clips to and 30% whose ICs
    sit above the physics-loss state clamp. Scoring a realistic-sampler model on
    it and calling the result in-distribution is the train/test mismatch the
    freeze work exists to prevent.

    `seed` must differ from `distribution.seed`, which is what draws the training
    set. Same sampler, same frozen parameters, different draw — that is what makes
    it a test set rather than a subset of training.

    `kind` is assigned from the *realised* in-window load variation rather than
    from the family name, because a family that varies over the day can still
    produce a nearly flat 12 h window (fix 7 made that a real part of the
    population). A case is 'CK' when its half peak-to-peak load is below
    `CK_THRESHOLD`. Consumers that split CK from TV — `eval/benchmark.py`,
    `18_swing_fidelity.py` — therefore keep working unchanged, and `family`
    carries the sampler label alongside.
    """
    from cod.data.realistic import DEFAULTS

    if params is None:
        params = DEFAULTS
    x0s, sensors, days = build_realistic_set(n_test, seed, params, T=T,
                                             n_sensors=n_sensors,
                                             return_days=True)
    cases: list[TestCase] = []
    for k in range(n_test):
        K_s = sensors[k, :n_sensors]
        Ta_s = sensors[k, n_sensors:2 * n_sensors]
        swing = 0.5 * float(K_s.max() - K_s.min())
        cases.append(TestCase(
            idx=k, kind=("CK" if swing < CK_THRESHOLD else "TV"),
            x0=x0s[k], K_sensors=K_s, Ta_sensors=Ta_s,
            K_mean=float(K_s.mean()), family=days[k]["kind"]))
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


def save_training_set(ts: TrainingSet, path: str | Path) -> None:
    """Persist a generated dataset, cache included."""
    np.savez(path, x0s=ts.x0s, sensors=ts.sensors, x_mean=ts.x_mean,
             x_std=ts.x_std, theta_ss=ts.ensure_theta_ss())


__all__ = [
    "TRAIN_FILE", "TrainingSet", "TestCase", "CK_THRESHOLD",
    "steady_state_on_grid", "save_training_set",
    "rk45_ground_truth", "generate_training_set",
    "generate_realistic_training_set", "load_training_set",
    "build_test_set", "build_realistic_test_set", "solve_test_set",
]
