"""Initial-condition sampler and load/ambient profile generator.

PHASE 1 — FAITHFUL PORT of n12 cell 0:
    sample_consistent_ic   L909
    make_sensor_profile    L926

Both consume a `numpy.random.RandomState` and must draw in exactly the source
order, because the stored `transformer_training_v57.npz` (8000 ICs, seed 42) and
the seed-999 test set are both reproduced by replaying these draw sequences. Any
reordering of `rng` calls silently changes the dataset.
"""

from __future__ import annotations

import numpy as np

from cod.data.physics import (
    B_aging,
    E_act,
    N_GAS,
    N_SENSORS,
    T_ref,
    TW,
    hot_spot_ETC_np,
    k_dis,
    k_gen,
)
from cod.data.steady_state import formula_A

# n12 cell 0 L902. Two later `N_IC = 4000` assignments exist (cell 2 L1298 and
# L1508) but data generation runs in cell 0 before them, and the stored .npz
# holds 8000 ICs. The two 4000 assignments are dead (audit MINOR).
N_IC = 8000

# The ten profile families and their weights, n12 cell 0 L936-L939.
# The manuscript says "eight load profile types"; there are ten. `daily` and
# `tv_ramp_sin` carry 15% of the training mass and are undeclared (audit §5).
PROFILE_KINDS = (
    "flat", "ramp", "daily", "sinusoidal", "step", "overload_spike",
    "multi_step", "peak_then_drop", "tv_high_amp", "tv_ramp_sin",
)
PROFILE_WEIGHTS = (0.11, 0.10, 0.08, 0.12, 0.11, 0.11, 0.11, 0.11, 0.08, 0.07)


def sample_consistent_ic(rng: np.random.RandomState,
                         steady_state=formula_A) -> np.ndarray:
    """Sample one initial condition: [theta_TO, c_H2, c_C2H2, c_C2H4, c_CO, c_CO2].

    Draw order, which must not change:
        1. rng.uniform()            the 20% high-K oversample coin
        2. rng.uniform(...)         K
        3. rng.uniform(15, 40)      theta_a
        4. rng.uniform(-30, 30)     offset from the steady state
        5. rng.uniform(0.5, 1.2, 5) per-gas multiplier on c_eq

    KNOWN DEFECT (audit M-6): `steady_state` defaults to formula A, which is off
    by up to -23.8 degC against the true fixed point. The docstring in the source
    calls the result "ETC-consistent"; it is not. Left as the default so the
    stored dataset is reproducible; Phase 2 fix 1 switches the default to
    `true_fixed_point`, which invalidates the checkpoints.

    KNOWN DEFECT (audit M-9): the resulting ICs are far outside the physical
    envelope — theta_TO(0) spans 20-150 degC, H2 reaches 1.14e5 ppm and C2H2
    2.52e5 ppm against IEC attention levels of 100 and 35 ppm. 37.0% of ICs
    exceed IEC attention levels and 23.2% exceed the `_hi` clamp inside the
    physics loss, so nearly a quarter of the training set is truncated before the
    residual is formed. Not corrected: it is a property of this sampler, and
    changing it would also invalidate the checkpoints.
    """
    # 20% oversample of high K, added in v38d to fix a specific failing test case
    K = rng.uniform(1.0, 1.3) if rng.uniform() < 0.2 else rng.uniform(0.4, 1.3)
    theta_a = rng.uniform(15, 40)
    theta_TO = steady_state(K, theta_a) + rng.uniform(-30, 30)
    theta_TO = float(np.clip(theta_TO, theta_a + 5, 150.0))
    theta_HS = hot_spot_ETC_np(theta_TO, K)
    T_HS_K = theta_HS + 273.15
    V_arr = np.exp(B_aging * E_act * (1.0 / T_ref - 1.0 / T_HS_K))
    c_eq = k_gen * V_arr / k_dis
    noise = rng.uniform(0.5, 1.2, N_GAS)
    dgas = np.maximum(c_eq * noise, 0.0)
    return np.concatenate([[theta_TO], dgas]).astype(np.float32)


def make_sensor_profile(rng: np.random.RandomState,
                        randomise_ambient_phase: bool = False) -> np.ndarray:
    """Sample one (K, theta_a) sensor profile, returned flat as [K(100), Ta(100)].

    Draw order, which must not change:
        1. rng.choice(kinds, p=weights)   the family
        2. rng.uniform(0.5, 1.2)          K_base
        3. family-specific draws
        4. rng.uniform(15, 45)            Ta_base
        5. rng.uniform(0, 8)              Ta amplitude

    KNOWN DEFECT (audit MINOR): the 'step' branch lacks the `.clip(0.3, 1.4)`
    that its siblings have. That is why the stored sensors reach K = 0.2571,
    below the documented floor of 0.3 — confirmed against the stored .npz
    (min 0.257134, 0.15% of profiles below the floor). Left unclipped here;
    Phase 2 fix 5 adds the clip.

    KNOWN DEFECT (audit B-5, fix 4): the ambient phase is fixed at 0 while the
    seed-999 test set uses a phase of pi/3, so training never sees the test
    waveform's phase. `randomise_ambient_phase` is the Phase 2 fix-4 switch and
    defaults to False to reproduce v57.

    KNOWN GAP (audit B-5): the training family was widened with the annotated
    goal `target TV gap: 2.7% -> <2.0%`, and 'sinusoidal' was added with the
    comment that the test set uses sin(2*pi*t/TW) so the training set should
    cover it. The test waveform sits inside every training marginal.
    """
    tau = np.linspace(0, TW, N_SENSORS)
    kind = rng.choice(PROFILE_KINDS, p=list(PROFILE_WEIGHTS))
    K_base = rng.uniform(0.5, 1.2)

    if kind == "flat":
        K_prof = np.full(N_SENSORS, K_base)
    elif kind == "ramp":
        K_prof = np.linspace(K_base, K_base + rng.uniform(-0.3, 0.3),
                             N_SENSORS).clip(0.3, 1.4)
    elif kind == "daily":
        # Period 1440 min against a 720 min window: only half a period is seen.
        # This is the v23 bug that motivated adding 'sinusoidal'.
        K_prof = K_base + 0.15 * np.sin(2 * np.pi * tau / (24 * 60))
    elif kind == "sinusoidal":
        period = rng.uniform(0.5, 2.0) * TW
        amp = rng.uniform(0.05, 0.40)   # widened from 0.10-0.30 in v57
        phase = rng.uniform(0, 2 * np.pi)
        K_prof = (K_base + amp * np.sin(2 * np.pi * tau / period + phase)).clip(0.3, 1.5)
    elif kind == "step":
        t_step = rng.uniform(0.2, 0.8) * TW
        # KNOWN DEFECT: no .clip() here, unlike every sibling branch.
        K_prof = np.where(tau < t_step, K_base, K_base + rng.uniform(-0.3, 0.3))
    elif kind == "overload_spike":
        K_prof = np.full(N_SENSORS, K_base)
        t_start = rng.uniform(0.2, 0.6) * TW
        t_end = t_start + rng.uniform(0.1, 0.3) * TW
        K_spike = rng.uniform(1.2, 1.5)
        K_prof = np.where((tau >= t_start) & (tau <= t_end), K_spike, K_base).clip(0.3, 1.5)
    elif kind == "peak_then_drop":
        amp = rng.uniform(0.10, 0.30)
        K_prof = (K_base + amp * np.sin(2 * np.pi * tau / TW + np.pi)).clip(0.3, 1.5)
    elif kind == "tv_high_amp":
        period = rng.uniform(0.5, 1.5) * TW
        amp = rng.uniform(0.30, 0.50)
        phase = rng.uniform(0, 2 * np.pi)
        K_prof = (K_base + amp * np.sin(2 * np.pi * tau / period + phase)).clip(0.3, 1.5)
    elif kind == "tv_ramp_sin":
        K_ramp = np.linspace(K_base, K_base + rng.uniform(0.1, 0.3), N_SENSORS)
        amp = rng.uniform(0.10, 0.25)
        K_sin = amp * np.sin(2 * np.pi * tau / TW + rng.uniform(0, 2 * np.pi))
        K_prof = (K_ramp + K_sin).clip(0.3, 1.5)
    else:  # multi_step
        K_prof = np.full(N_SENSORS, K_base)
        n_steps = rng.randint(2, 5)
        for ts, kv in zip(np.sort(rng.uniform(0, TW, n_steps)),
                          rng.uniform(0.4, 1.3, n_steps + 1)):
            K_prof = np.where(tau >= ts, kv, K_prof)
        K_prof = K_prof.clip(0.3, 1.4).astype(float)

    Ta_base = rng.uniform(15, 45)
    Ta_amp = rng.uniform(0, 8)
    if randomise_ambient_phase:
        # Phase 2 fix 4. Drawn last so that with the flag off the draw sequence,
        # and therefore the stored dataset, is bit-identical to v57.
        Ta_phase = rng.uniform(0, 2 * np.pi)
    else:
        Ta_phase = 0.0
    Ta_prof = Ta_base + Ta_amp * np.sin(2 * np.pi * tau / TW + Ta_phase)
    return np.concatenate([K_prof, Ta_prof]).astype(np.float32)


__all__ = ["N_IC", "PROFILE_KINDS", "PROFILE_WEIGHTS",
           "sample_consistent_ic", "make_sensor_profile"]
