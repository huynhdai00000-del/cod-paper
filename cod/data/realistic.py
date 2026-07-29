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

from dataclasses import dataclass, field, fields

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

    # The load pattern is a DAY. The 12 h window is a slice of it at a random
    # time of day (DECISIONS N-6). Before this, every family completed a full
    # cycle inside the 720 min window, i.e. a 12 h load period, and a real one is
    # 24 h. See `make_realistic_day` for why that mattered more than it looks.
    cycle_period: float = 1440.0
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

    # ── config binding ─────────────────────────────────────────────────────
    # Every field above is a knob somebody could be tempted to widen to make a
    # number look better, so every field above must live inside the hashed
    # `distribution` block. `from_config` enforces that in both directions, which
    # is the whole point: a hash that silently omits a knob is worse than no hash,
    # because it certifies something it did not check.

    _TUPLE_FIELDS = {
        "hot_spot_bounds": float, "K_bounds": float, "K_amp": float,
        "overload_K": float, "Ta_base": float, "Ta_amp": float,
        "theta_TO_bounds": float, "service_factor": float,
        "fault_gases": int, "fault_factor": float,
        "families": str, "weights": float,
    }

    @classmethod
    def from_config(cls, block: dict, where: str = "distribution.sampler.params"
                    ) -> "RealisticParams":
        """Build from a hashed config block, refusing anything incomplete.

        Two checks, and both matter:

        **Missing.** Every dataclass field must appear in `block`. Falling back to
        a Python default would mean the hash certifies a distribution whose
        parameters are not in the hashed text — exactly the failure
        `DISTRIBUTION_FREEZE.md` §2.1 recorded, where `K_base` sat in the YAML,
        moved the hash when edited, and reached no sampler.

        **Unknown.** Any key in `block` that is not a field is also an error. That
        is the same failure seen from the other side: a knob that looks
        authoritative, changes the hash, and does nothing. A typo in a field name
        would otherwise land here silently and leave the real field on its default.

        Adding a field to this dataclass therefore breaks every config until the
        config declares it. That is intended — it is what keeps the hash honest.
        """
        if not isinstance(block, dict):
            raise TypeError(f"{where} must be a mapping, got {type(block).__name__}")

        names = {f.name for f in fields(cls)}
        given = set(block)
        missing = sorted(names - given)
        unknown = sorted(given - names)
        if missing or unknown:
            msg = [f"{where} does not match RealisticParams."]
            if missing:
                msg.append(
                    f"  MISSING ({len(missing)}): {', '.join(missing)}\n"
                    "    The sampler reads these. Absent from the hashed config, "
                    "they would silently take Python defaults and the frozen hash "
                    "would certify a distribution it never saw.")
            if unknown:
                msg.append(
                    f"  UNKNOWN ({len(unknown)}): {', '.join(unknown)}\n"
                    "    No sampler parameter has these names. Editing them moves "
                    "the hash and changes no data. Check for a typo.")
            raise ValueError("\n".join(msg))

        kw = {}
        for name, value in block.items():
            if name in cls._TUPLE_FIELDS:
                cast = cls._TUPLE_FIELDS[name]
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"{where}.{name} must be a list, got "
                                    f"{type(value).__name__}")
                kw[name] = tuple(cast(v) for v in value)
            else:
                kw[name] = value

        p = cls(**kw)
        if len(p.families) != len(p.weights):
            raise ValueError(
                f"{where}: families has {len(p.families)} entries but weights has "
                f"{len(p.weights)}; they index each other in make_realistic_day.")
        total = sum(p.weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{where}.weights sums to {total!r}, not 1.0 "
                             "(rng.choice would renormalise it silently).")
        unknown_fam = sorted(set(p.families) - set(KNOWN_FAMILIES))
        if unknown_fam:
            raise ValueError(
                f"{where}.families contains {unknown_fam}, which "
                "make_realistic_day has no branch for. It would fall through to "
                "multi_step without saying so.")
        return p

    def to_config_dict(self) -> dict:
        """The block that `from_config` would accept, for writing a config out."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


# Every `kind` make_realistic_day branches on. The final `else` is multi_step, so
# an unrecognised name silently becomes multi_step; `from_config` rejects it
# instead.
KNOWN_FAMILIES = ("base_load", "daily", "ramp", "shift_change", "evening_peak",
                  "overload_spike", "multi_step")

DEFAULTS = RealisticParams()


# ═══════════════════════════════════════════════════════════════════════════
# Load and ambient
# ═══════════════════════════════════════════════════════════════════════════
def make_realistic_day(rng: np.random.RandomState,
                       p: RealisticParams = DEFAULTS,
                       n_day: int | None = None) -> dict:
    """One full 24 h (K, theta_a) pattern, plus the time of day the window starts.

    PHASE 2 FIX 7 (DECISIONS N-6, N-7). The load pattern is now a **day** and the
    evaluation window is a slice of it. Previously every family completed a whole
    cycle inside the 720 min window, which is a 12 h load period against a real
    one of 24 h, and that is not a cosmetic difference:

    a first-order thermal system attenuates a sinusoid by
    `1/sqrt(1 + (omega tau_oil)^2)`, which is 0.607 at a 12 h period and 0.837 at
    24 h — a ratio of 1.378. Forcing at the wrong period therefore forced the
    calibration to assume 1.378x more load swing than reality to reach a given
    hot-spot swing. `K_amp = 12-28%` divided by 1.378 is 8.7-20.3%, which is the
    range ETT actually measures (`ETT_LOAD_CALIBRATION.md`: ETTh2 median 8.7%,
    ETTh1 non-back-feeding days 17.8%). The amplitude was never the error; the
    period was. **`K_amp` is deliberately not touched.**

    Every family is built over the day, not only the periodic ones. A shift change
    or an overload spike happens at a *time of day*, so a 12 h window may contain
    it or may not — which is the point. Windows that happen to contain little
    variation are a real and previously absent part of the population.

    Returns the day arrays on a uniform grid over `[0, cycle_period)`, treated as
    periodic, together with `offset`, the time of day at which the window begins.
    """
    P = p.cycle_period
    if n_day is None:
        n_day = 4 * N_SENSORS
    t = np.linspace(0.0, P, n_day, endpoint=False)
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
        K = K_base + p.base_load_frac * amp * np.sin(2 * np.pi * t / P + phase)
    elif kind == "daily":
        K = K_base + amp * np.sin(2 * np.pi * t / P + phase)
    elif kind == "ramp":
        # A ramp is a within-day trend, so it rises across the day and returns:
        # a monotone ramp over 24 h that never comes back is not a daily pattern.
        K = K_base + rng.uniform(-1.5, 1.5) * amp * np.sin(np.pi * t / P)
    elif kind == "shift_change":
        # Two shifts a day, changing at a drawn time of day.
        t_step = rng.uniform(0.0, 1.0) * P
        dK = rng.uniform(-1.5, 1.5) * amp
        half = np.mod(t - t_step, P) < 0.5 * P
        K = np.where(half, K_base + dK, K_base)
    elif kind == "evening_peak":
        # One broad peak per day, the usual residential shape. Absolute widths are
        # unchanged from the window-local version (130-216 min); only the position
        # is now drawn over the day, so a window may miss the peak entirely.
        centre = rng.uniform(0.0, 1.0) * P
        width = rng.uniform(0.09, 0.15) * P
        d = np.abs(t - centre)
        d = np.minimum(d, P - d)                       # circular distance
        K = K_base + 1.4 * amp * np.exp(-0.5 * (d / width) ** 2)
    elif kind == "overload_spike":
        # Absolute duration unchanged (58-144 min), position drawn over the day.
        K = np.full(n_day, K_base)
        t0 = rng.uniform(0.0, 1.0) * P
        dur = rng.uniform(0.04, 0.10) * P
        d = np.mod(t - t0, P)
        K = np.where(d <= dur, rng.uniform(*p.overload_K), K)
    else:  # multi_step
        K = np.full(n_day, K_base)
        for ts in np.sort(rng.uniform(0, P, rng.randint(2, 7))):
            K = np.where(t >= ts, K_base + rng.uniform(-1.2, 1.2) * amp, K)
        # close the loop so the pattern is genuinely periodic
        K[t >= P - 1e-9] = K[0]

    K = np.clip(K, *p.K_bounds)
    Ta = (Ta_base + rng.uniform(*p.Ta_amp)
          * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi)))
    offset = float(rng.uniform(0.0, P))
    return {"kind": str(kind), "t": t, "K": K, "Ta": Ta, "offset": offset,
            "period": P}


def window_from_day(day: dict, T: float = TW,
                    n_sensors: int = N_SENSORS,
                    offset: float | None = None) -> np.ndarray:
    """The `T`-long slice of a day pattern starting at `offset`, flat [K(n), Ta(n)].

    `np.interp(..., period=P)` wraps, so a window that straddles midnight is
    handled without a special case.
    """
    P = day["period"]
    off = day["offset"] if offset is None else offset
    t_win = off + np.linspace(0.0, T, n_sensors)
    K = np.interp(t_win, day["t"], day["K"], period=P)
    Ta = np.interp(t_win, day["t"], day["Ta"], period=P)
    return np.concatenate([K, Ta]).astype(np.float32)


def make_realistic_profile(rng: np.random.RandomState,
                           p: RealisticParams = DEFAULTS,
                           T: float = TW,
                           n_sensors: int = N_SENSORS) -> np.ndarray:
    """One (K, theta_a) window, returned flat as [K(n), Ta(n)].

    Convenience wrapper: draw a day, take the window. Callers that also need a
    consistent initial condition want `make_realistic_day` plus `window_from_day`,
    because `sample_realistic_ic` needs the whole day to find the periodic state
    (see its `day` argument).
    """
    return window_from_day(make_realistic_day(rng, p), T, n_sensors)


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
    """theta_TO(0) for a unit that has been running this pattern cycle after cycle.

    Integrates the thermal ODE over `n_cycles` copies of `(K_s, Ta_s)` and returns
    the end value, which is the periodic steady state. With tau_oil = 150 min
    against a 720 min cycle, three cycles leaves a transient of
    exp(-3*720/150) = 6e-7 of the initial mismatch.

    Reuses `DailyMeanArrhenius.theta_TO_trajectory` rather than restating the
    recurrence, so the closed-form solution has one definition in the package.

    NOTE (fix 7): `(K_s, Ta_s)` must be a full **cycle**. Since N-6 the cycle is
    the 24 h day, not the 12 h window, so callers pass the day arrays and read the
    result at the window's offset — see `day_steady_theta0`. Repeating a 12 h
    window would assert a 12 h load period, which is the error fix 7 removes.
    """
    theta = float(true_fixed_point_np(float(K_s[0]), float(Ta_s[0])))
    dm = DailyMeanArrhenius(T=T, n_sensors=len(K_s))
    for _ in range(n_cycles):
        _, traj = dm.theta_TO_trajectory(theta, K_s, Ta_s)
        theta = float(traj[-1])
    return theta


def day_theta_cycle(day: dict, n_cycles: int = DEFAULTS.burn_in_cycles):
    """The periodic 24 h top-oil trajectory of a unit running this day pattern.

    Returns `(s, K_c, Ta_c, theta)` on the closed cycle grid — closed meaning the
    endpoint repeats the start, which the recurrence needs and `np.interp`'s
    `period` argument then wraps correctly.

    Both the initial condition and the gas equilibrium are read off this one
    trajectory, so they cannot disagree about what the unit is doing.
    """
    P = day["period"]
    K_c = np.append(day["K"], day["K"][0])
    Ta_c = np.append(day["Ta"], day["Ta"][0])
    dm = DailyMeanArrhenius(T=P, n_sensors=len(K_c))
    theta = float(true_fixed_point_np(float(K_c[0]), float(Ta_c[0])))
    for _ in range(n_cycles):
        s, traj = dm.theta_TO_trajectory(theta, K_c, Ta_c)
        theta = float(traj[-1])
    s, traj = dm.theta_TO_trajectory(theta, K_c, Ta_c)
    return s, K_c, Ta_c, traj


def day_steady_theta0(day: dict,
                      n_cycles: int = DEFAULTS.burn_in_cycles) -> float:
    """theta_TO at the window's start, for a unit already cycling on this day.

    The periodic state of the whole 24 h pattern, read at `day["offset"]`. This is
    what makes the initial condition consistent with the profile once the profile
    is a day and the window is a slice of it.
    """
    s, _, _, traj = day_theta_cycle(day, n_cycles)
    return float(np.interp(day["offset"], s, traj, period=day["period"]))


def sample_realistic_ic(rng: np.random.RandomState, sensors: np.ndarray,
                        p: RealisticParams = DEFAULTS,
                        T: float = TW,
                        n_sensors: int = N_SENSORS,
                        day: dict | None = None) -> np.ndarray:
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

    if day is not None:
        # Fix 7: the periodic state belongs to the 24 h cycle, sampled at the
        # window's time of day.
        theta0 = day_steady_theta0(day, p.burn_in_cycles)
    else:
        # Fallback for callers that only hold a window. It asserts a load period
        # equal to the window, which is exactly what N-6 identifies as wrong, so
        # it is kept only so old scripts still run and is not used by
        # `build_realistic_set`.
        theta0 = periodic_steady_theta0(K_s, Ta_s, p.burn_in_cycles, T)
    offset = float(np.clip(rng.normal(0.0, p.hist_sigma), -p.hist_clip, p.hist_clip))
    noise = float(rng.normal(0.0, p.sensor_sigma))
    theta_TO = float(np.clip(theta0 + offset + noise, *p.theta_TO_bounds))

    # The unit's typical operating hot-spot, which is what its dissolved-gas
    # equilibrium reflects. Fix 7: averaged over the whole DAY when one is
    # available, not over the 12 h window. Dissolved gas equilibrates on a
    # timescale of weeks; a window that happens to fall on the night trough should
    # not be given the gas loading of a permanently cool unit.
    if day is not None:
        _, K_c, _, traj_d = day_theta_cycle(day, p.burn_in_cycles)
        hs = np.array([hot_spot_ETC_np(float(traj_d[i]), float(K_c[i]))
                       for i in range(len(K_c))])
    else:
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
                        T: float = TW, n_sensors: int = N_SENSORS,
                        return_days: bool = False):
    """`n` (IC, profile) pairs. The day is drawn first, then the window, then the IC.

    Order matters and is the opposite of the old sampler's, which drew every IC
    before any profile and so could not have made them consistent. Since fix 7 the
    day is the primary object and the IC is the periodic state of that day read at
    the window's time of day.
    """
    rng = np.random.RandomState(seed)
    sensors = np.empty((n, 2 * n_sensors), dtype=np.float32)
    x0s = np.empty((n, 1 + N_GAS), dtype=np.float32)
    days = []
    for i in range(n):
        day = make_realistic_day(rng, p)
        sensors[i] = window_from_day(day, T, n_sensors)
        x0s[i] = sample_realistic_ic(rng, sensors[i], p, T, n_sensors, day=day)
        days.append(day)
    if return_days:
        return x0s, sensors, days
    return x0s, sensors


def iec_exceedance(x0s: np.ndarray) -> dict:
    """Fraction of ICs above IEC 60599 attention, overall and per gas."""
    att = np.asarray(IEC_ATTENTION, dtype=float)
    over = x0s[:, 1:] > att
    from cod.data.physics import GAS_NAMES
    out = {g: float(over[:, i].mean()) for i, g in enumerate(GAS_NAMES)}
    out["any_gas"] = float(over.any(axis=1).mean())
    return out


__all__ = ["RealisticParams", "DEFAULTS", "make_realistic_day",
           "window_from_day", "make_realistic_profile",
           "periodic_steady_theta0", "day_theta_cycle", "day_steady_theta0",
           "sample_realistic_ic",
           "build_realistic_set", "iec_exceedance", "solve_K_for_hot_spot",
           "steady_hot_spot"]
