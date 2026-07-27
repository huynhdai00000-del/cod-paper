"""An operationally realistic sampler, replacing the one audit M-9 found.

Why this exists. `profiles.sample_consistent_ic` draws `theta_TO(0)` as
`steady_state(K, theta_a) + U(-30, 30)` where `K` and `theta_a` are drawn
**independently of the load profile that will drive the window**. The initial
thermal state therefore does not match the load, and the window is spent relaxing
from a mismatched start. Measured on the seed-999 test set
(`audit_port/scripts/12_swing_decomposition.py`):

    IC offset from the profile-consistent value   median 28.3 degC, max 74.3
    realised hot-spot swing                       median 21.4 degC, 68% above 15
    of which removed by a consistent IC alone     76%

That inflates the Jensen gap, which depends on swing, and it drives `theta_TO(0)`
to 150 degC, which pushes the hot-spot past 187 degC where the model's `V_arr`
clamp diverges from the reference (DECISIONS N-1). It also puts 37.0% of gas ICs
above IEC 60599 attention levels, because `c_eq = k_gen V_arr / k_dis` is
exponential in a temperature that should never have been that high.

Two things are needed, and the decomposition above says why neither is sufficient
alone:

* `sample_realistic_ic` — `theta_TO(0)` from the thermal state the profile would
  actually produce, plus a bounded perturbation. On its own this takes the
  constant-load cases to *exactly zero* swing, since constant forcing means a
  constant trajectory.
* `make_realistic_profile` — load and ambient variation calibrated to what a
  transformer actually sees in a day. On its own it cannot help, because the IC
  transient dominates.

Together they give a swing distribution centred where operation puts it.

Nothing here is frozen. The parameters live in `RealisticParams` so they can be
argued with before anything is committed to `DISTRIBUTION_FREEZE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cod.data.physics import (
    B_aging,
    E_act,
    IEC_ATTENTION,
    N_GAS,
    N_SENSORS,
    T_ref,
    TW,
    hot_spot_ETC_np,
    k_dis,
    k_gen,
)
from cod.data.steady_state import true_fixed_point_np
from cod.models.daily_mean import DailyMeanArrhenius


@dataclass
class RealisticParams:
    """Every knob, in one place, with the operational reasoning attached.

    Load. `K_amp` is the daily swing of the load factor about its own mean, and
    the calibrated range +-12-28% is at the **upper end** of what a real feeder
    does. That is worth stating plainly rather than dressing up: it is what a
    hot-spot swing of 10-15 degC requires, and it is the amplitude the target
    implies, not an independent estimate of fleet behaviour. A base-loaded unit
    swings far less, which is why `base_load` exists as a family and why the p10 of
    the realised swing sits near 5 degC.

    The honest reading is that the Jensen gap matters for cycled units. A unit that
    truly runs flat has almost no gap, and no method can beat a mean-temperature
    calculation on it.

    Ambient. A diurnal air-temperature cycle of 6-16 degC peak to peak, which is
    ordinary. Bounded away from zero, unlike the U(0, 8) the old sampler used, and
    with a random phase (Phase 2 fix 4).

    Initial condition. `hist_sigma` is the spread of `theta_TO(0)` about the
    profile's own periodic steady state, representing that the preceding hours
    were not identical to this window. `hist_clip` bounds it so no case can start
    the way the old sampler did. `sensor_sigma` is instrument error on the
    top-oil reading.

    Gases. Long-run equilibrium at the unit's own mean hot-spot, times a
    service-history factor. `fault_prob` gives a minority of units with an
    incipient fault, which is what makes a DGA benchmark non-trivial.
    """

    # operating point. Loading is not drawn directly: a fleet is loaded so that
    # temperature stays in band, so the *hot-spot* is drawn and the load that
    # achieves it is solved for. IEC 60076-7 puts rated hot-spot at 98 degC, the
    # normal-cyclic ceiling at 120 and long-time emergency at 140.
    hot_spot_mean: float = 86.0
    hot_spot_sd: float = 11.0
    hot_spot_bounds: tuple[float, float] = (62.0, 122.0)
    K_bounds: tuple[float, float] = (0.30, 1.40)
    # load shape about that operating point
    K_amp: tuple[float, float] = (0.12, 0.28)
    base_load_frac: float = 0.50      # how flat the flattest family really is
    overload_K: tuple[float, float] = (1.15, 1.40)
    # ambient
    Ta_base: tuple[float, float] = (15.0, 40.0)
    Ta_amp: tuple[float, float] = (3.0, 8.0)
    # initial condition
    hist_sigma: float = 3.0
    hist_clip: float = 8.0
    sensor_sigma: float = 0.5
    theta_TO_bounds: tuple[float, float] = (30.0, 130.0)
    burn_in_cycles: int = 3
    # gases
    service_factor: tuple[float, float] = (0.45, 1.35)
    fault_prob: float = 0.08
    fault_gases: tuple[int, ...] = (0, 1, 2)      # H2, C2H2, C2H4
    fault_factor: tuple[float, float] = (2.0, 7.0)
    gas_floor_frac: float = 0.02                  # fraction of c_eq, keeps ICs > 0

    families: tuple[str, ...] = ("base_load", "daily", "ramp", "shift_change",
                                 "evening_peak", "overload_spike", "multi_step")
    weights: tuple[float, ...] = (0.18, 0.22, 0.12, 0.14, 0.16, 0.08, 0.10)


DEFAULTS = RealisticParams()


# ═══════════════════════════════════════════════════════════════════════════
# Load and ambient
# ═══════════════════════════════════════════════════════════════════════════
def make_realistic_profile(rng: np.random.RandomState,
                           p: RealisticParams = DEFAULTS,
                           T: float = TW,
                           n_sensors: int = N_SENSORS) -> np.ndarray:
    """One (K, theta_a) profile, returned flat as [K(n), Ta(n)].

    Same operational shapes as the v57 families, with amplitudes calibrated to a
    real daily cycle rather than to closing a test-set gap. Every profile carries a
    diurnal ambient cycle with a random phase, because every site does.
    """
    tau = np.linspace(0.0, T, n_sensors)
    kind = rng.choice(p.families, p=list(p.weights))

    # Draw the intended operating temperature, then the load that achieves it at
    # this site's ambient. See `solve_K_for_hot_spot`.
    Ta_base = rng.uniform(*p.Ta_base)
    target_hs = float(np.clip(rng.normal(p.hot_spot_mean, p.hot_spot_sd),
                              *p.hot_spot_bounds))
    K_base = float(np.clip(solve_K_for_hot_spot(target_hs, Ta_base),
                           *p.K_bounds))
    amp = rng.uniform(*p.K_amp)
    phase = rng.uniform(0.0, 2.0 * np.pi)

    if kind == "base_load":
        K = K_base + p.base_load_frac * amp * np.sin(2 * np.pi * tau / T + phase)
    elif kind == "daily":
        K = K_base + amp * np.sin(2 * np.pi * tau / T + phase)
    elif kind == "ramp":
        K = np.linspace(K_base, K_base + rng.uniform(-1.5, 1.5) * amp, n_sensors)
    elif kind == "shift_change":
        t_step = rng.uniform(0.3, 0.7) * T
        K = np.where(tau < t_step, K_base, K_base + rng.uniform(-1.5, 1.5) * amp)
    elif kind == "evening_peak":
        # one broad peak inside the window, the usual residential shape
        centre = rng.uniform(0.45, 0.7) * T
        width = rng.uniform(0.18, 0.30) * T
        K = K_base + 1.4 * amp * np.exp(-0.5 * ((tau - centre) / width) ** 2)
    elif kind == "overload_spike":
        K = np.full(n_sensors, K_base)
        t0 = rng.uniform(0.2, 0.6) * T
        t1 = t0 + rng.uniform(0.08, 0.20) * T
        K = np.where((tau >= t0) & (tau <= t1), rng.uniform(*p.overload_K), K)
    else:  # multi_step
        K = np.full(n_sensors, K_base)
        for ts in np.sort(rng.uniform(0, T, rng.randint(1, 4))):
            K = np.where(tau >= ts, K_base + rng.uniform(-1.2, 1.2) * amp, K)

    K = np.clip(K, *p.K_bounds)
    Ta = (Ta_base + rng.uniform(*p.Ta_amp)
          * np.sin(2 * np.pi * tau / T + rng.uniform(0, 2 * np.pi)))
    return np.concatenate([K, Ta]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Initial condition
# ═══════════════════════════════════════════════════════════════════════════
_dm = DailyMeanArrhenius()


def steady_hot_spot(K: float, Ta: float) -> float:
    """Hot-spot temperature at the steady state of constant (K, theta_a)."""
    return float(hot_spot_ETC_np(float(true_fixed_point_np(K, Ta)), K))


def solve_K_for_hot_spot(target_C: float, Ta: float,
                         bounds: tuple[float, float] = (0.05, 2.0),
                         tol: float = 1e-4) -> float:
    """Load factor whose steady state gives `target_C` at the hot spot.

    `steady_hot_spot` is strictly increasing in K, so bisection is enough and
    needs no derivative. This inversion is the point of the design: a utility
    loads a transformer to keep it in temperature band, so the load is a
    consequence of the intended operating temperature and the ambient, not an
    independent draw. Drawing them independently is what put 37% of the old
    sampler's gas ICs above IEC attention.
    """
    lo, hi = bounds
    f_lo, f_hi = steady_hot_spot(lo, Ta), steady_hot_spot(hi, Ta)
    if target_C <= f_lo:
        return lo
    if target_C >= f_hi:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if steady_hot_spot(mid, Ta) < target_C:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def periodic_steady_theta0(K_s: np.ndarray, Ta_s: np.ndarray,
                           n_cycles: int = DEFAULTS.burn_in_cycles,
                           T: float = TW) -> float:
    """theta_TO(0) for a unit that has been running this pattern day after day.

    Integrates the thermal ODE over `n_cycles` copies of the window and returns the
    end value, which is the periodic steady state. With tau_oil = 150 min against a
    720 min window, three cycles leaves a transient of exp(-3*720/150) = 6e-7 of
    the initial mismatch.

    Reuses `DailyMeanArrhenius.theta_TO_trajectory` rather than restating the
    recurrence, so the closed-form solution has one definition in the package.
    """
    theta = float(true_fixed_point_np(float(K_s[0]), float(Ta_s[0])))
    for _ in range(n_cycles):
        _, traj = _dm.theta_TO_trajectory(theta, K_s, Ta_s)
        theta = float(traj[-1])
    return theta


def sample_realistic_ic(rng: np.random.RandomState, sensors: np.ndarray,
                        p: RealisticParams = DEFAULTS,
                        T: float = TW,
                        n_sensors: int = N_SENSORS) -> np.ndarray:
    """One initial condition consistent with the profile that will drive the window.

    `sensors` is the flat [K(n), Ta(n)] array from `make_realistic_profile`. The
    dependence on it is the whole point: the old sampler drew `theta_TO(0)` from a
    load unrelated to the profile.

    theta_TO(0) = periodic steady state of this profile
                  + recent-history offset, N(0, hist_sigma) clipped to +-hist_clip
                  + sensor noise, N(0, sensor_sigma)

    The offset is what stops the distribution being degenerate: without it every
    constant-load window would start exactly at its steady state and never move,
    which is no more realistic than the old +-30 degC draw and would make the
    Jensen gap identically zero on those cases.

    Gas ICs are the long-run equilibrium at the unit's own mean hot-spot times a
    service-history factor, with a minority carrying an incipient fault. Drawing
    them at a *consistent* temperature is what removes the 37% IEC exceedance:
    `c_eq` is exponential in temperature, so it was the 150 degC ICs producing it.
    """
    K_s = sensors[:n_sensors].astype(float)
    Ta_s = sensors[n_sensors:2 * n_sensors].astype(float)

    theta0 = periodic_steady_theta0(K_s, Ta_s, p.burn_in_cycles, T)
    offset = float(np.clip(rng.normal(0.0, p.hist_sigma), -p.hist_clip, p.hist_clip))
    noise = float(rng.normal(0.0, p.sensor_sigma))
    theta_TO = float(np.clip(theta0 + offset + noise, *p.theta_TO_bounds))

    # The unit's typical operating hot-spot over this window, which is what its
    # dissolved-gas equilibrium reflects.
    _, traj = _dm.theta_TO_trajectory(theta_TO, K_s, Ta_s)
    hs = np.array([hot_spot_ETC_np(float(traj[i]), float(K_s[i]))
                   for i in range(n_sensors)])
    hs_mean = float(hs.mean())

    T_HS_K = hs_mean + 273.15
    V_arr = np.exp(B_aging * E_act * (1.0 / T_ref - 1.0 / T_HS_K))
    c_eq = k_gen * V_arr / k_dis

    gases = c_eq * rng.uniform(*p.service_factor, size=N_GAS)
    if rng.uniform() < p.fault_prob:
        f = rng.uniform(*p.fault_factor)
        for i in p.fault_gases:
            gases[i] = gases[i] * f
    gases = np.maximum(gases, c_eq * p.gas_floor_frac)

    return np.concatenate([[theta_TO], gases]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════
def build_realistic_set(n: int, seed: int, p: RealisticParams = DEFAULTS,
                        T: float = TW, n_sensors: int = N_SENSORS):
    """`n` (IC, profile) pairs. The profile is drawn first, then the IC from it.

    Order matters and is the opposite of the old sampler's, which drew every IC
    before any profile and so could not have made them consistent.
    """
    rng = np.random.RandomState(seed)
    sensors = np.empty((n, 2 * n_sensors), dtype=np.float32)
    x0s = np.empty((n, 1 + N_GAS), dtype=np.float32)
    for i in range(n):
        sensors[i] = make_realistic_profile(rng, p, T, n_sensors)
        x0s[i] = sample_realistic_ic(rng, sensors[i], p, T, n_sensors)
    return x0s, sensors


def iec_exceedance(x0s: np.ndarray) -> dict:
    """Fraction of ICs above IEC 60599 attention, overall and per gas."""
    att = np.asarray(IEC_ATTENTION, dtype=float)
    over = x0s[:, 1:] > att
    from cod.data.physics import GAS_NAMES
    out = {g: float(over[:, i].mean()) for i, g in enumerate(GAS_NAMES)}
    out["any_gas"] = float(over.any(axis=1).mean())
    return out


__all__ = ["RealisticParams", "DEFAULTS", "make_realistic_profile",
           "periodic_steady_theta0", "sample_realistic_ic",
           "build_realistic_set", "iec_exceedance"]
