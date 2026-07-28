"""The practitioner baseline: Arrhenius evaluated at the mean hot-spot temperature.

Tier 0 of the baseline matrix (DECISIONS C-11) and the paper's motivating gap
(C-10). Current industry practice takes a daily-mean temperature and multiplies by
a single Arrhenius factor. Because `V_arr` is convex in temperature, that
systematically *understates* both ageing and gas generation:

    (1/T) integral_0^T V(theta(s)) ds   >   V( (1/T) integral_0^T theta(s) ds )

by Jensen's inequality, with equality only for a constant trajectory. The gap grows
with activation energy and with swing amplitude, so it is largest for C2H2 — the
arc-discharge marker.

Two separate things live here, and they answer different questions:

`DailyMeanArrhenius`
    A model, usable as a baseline. IEC 60076-7 analytic thermal trajectory, then a
    single Arrhenius factor at the window-mean hot-spot for every gas and for DP.
    Its error against the reference mixes thermal error with the convexity gap.

`jensen_gap_from_trajectory`
    A measurement. Given a hot-spot trajectory it returns the ratio above, per gas
    and for DP. Fed the *true* trajectory it isolates convexity with no model error
    in it at all. This is what O-8 asks for.

On "daily": the forecast window is 12 h (C-4), so the averaging window here is the
12 h window, not 24 h. The distinction does not affect the ratio for a trajectory
averaged over a whole number of periods, which is the case for the time-varying test
profiles (period = TW). Stated because the name says "daily".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cod.data.physics import (
    B_aging,
    DP0,
    E_act,
    K_PD_onset,
    N_GAS,
    N_SENSORS,
    PD_gain,
    T_ref,
    TW,
    hot_spot_ETC_np,
    k0_aging,
    k_dis,
    k_gen,
    tau_oil,
)
from cod.data.steady_state import true_fixed_point_np

# DP ages under the un-weighted Arrhenius factor, i.e. an effective E_act of 1.0
# in the same normalisation the gases use. B_aging = 15000 K = Ea/R, so
# Ea = 15000 * 8.314 = 124.7 kJ/mol, matching the DP row of C-10.
E_ACT_DP = 1.0
R_GAS = 8.314


def activation_energies_kJ() -> dict[str, float]:
    """Ea in kJ/mol for the five gases and DP, from `B_aging` and `E_act`.

    Reproduces the Ea column of DECISIONS C-10 from the code's own constants, so
    the analytical table can be checked rather than trusted.
    """
    from cod.data.physics import GAS_NAMES
    out = {g: float(e * B_aging * R_GAS / 1000.0) for g, e in zip(GAS_NAMES, E_act)}
    out["DP"] = float(E_ACT_DP * B_aging * R_GAS / 1000.0)
    return out


def arrhenius(theta_HS_C, e_act) -> np.ndarray:
    """V_arr = exp(B_aging * e_act * (1/T_ref - 1/T_HS)).

    Deliberately unclamped, matching `fast_rhs_np`, the reference ODE. Since
    Phase 2 fix 6 the model path in `cod/models/cod.py` agrees: it bounds the
    temperature at [313.15, 573.15] K, as the reference does, rather than capping
    the rate (DECISIONS N-1, now resolved). Not even the temperature bound is
    applied here, because the Jensen measurements live around 100 degC where it
    is inert, and applying it would silently flatten any future measurement taken
    outside the envelope instead of making it visible.
    """
    T_HS_K = np.asarray(theta_HS_C, dtype=float) + 273.15
    e = np.asarray(e_act, dtype=float)
    return np.exp(B_aging * e * (1.0 / T_ref - 1.0 / T_HS_K[..., None]))


def jensen_gap_from_trajectory(theta_HS_C, t=None, e_act=None
                              ) -> tuple[np.ndarray, float]:
    """Ratio of the trajectory-resolved Arrhenius integral to the mean-temperature one.

    Parameters
    ----------
    theta_HS_C
        Hot-spot temperature trajectory, degC, shape (n_t,).
    t
        Times, shape (n_t,). Uniform spacing assumed if omitted.
    e_act
        Activation energies in the `E_act` normalisation. Defaults to the five
        gases followed by DP.

    Returns
    -------
    gap
        Ratio per activation energy. Always >= 1 by convexity; exactly 1 for a
        constant trajectory.
    theta_mean
        The mean hot-spot temperature the denominator was evaluated at.
    """
    theta = np.asarray(theta_HS_C, dtype=float)
    if t is None:
        t = np.linspace(0.0, 1.0, len(theta))
    t = np.asarray(t, dtype=float)
    if e_act is None:
        e_act = np.concatenate([E_act, [E_ACT_DP]])
    e_act = np.asarray(e_act, dtype=float)

    span = t[-1] - t[0]
    theta_mean = float(np.trapz(theta, t) / span)
    resolved = np.trapz(arrhenius(theta, e_act), t, axis=0) / span   # (n_e,)
    at_mean = arrhenius(np.array([theta_mean]), e_act)[0]            # (n_e,)
    return resolved / at_mean, theta_mean


def jensen_gap_sinusoidal(amplitude_C: float, theta_0_C: float = 100.0,
                          n: int = 20001, e_act=None) -> np.ndarray:
    """The analytical prediction: gap for a full-period sinusoid of given amplitude.

    This is what DECISIONS C-10 tabulates. Computed by quadrature over one period,
    which is exact to the resolution of `n`, rather than by the Bessel-function
    small-amplitude approximation.
    """
    phase = np.linspace(0.0, 2.0 * np.pi, n)
    theta = theta_0_C + amplitude_C * np.sin(phase)
    gap, _ = jensen_gap_from_trajectory(theta, phase, e_act=e_act)
    return gap


@dataclass
class DailyMeanResult:
    """One window of the practitioner baseline."""

    theta_TO: np.ndarray      # (n_t,) degC
    gases: np.ndarray         # (n_t, 5) ppm
    DP: np.ndarray            # (n_t,)
    theta_HS_mean: float      # degC, the single temperature everything used
    V_arr_at_mean: np.ndarray # (5,) the frozen Arrhenius factors


class DailyMeanArrhenius:
    """IEC 60076-7 thermal trajectory + one Arrhenius factor at the mean hot-spot.

    Not a network. No parameters, nothing to train, so it needs no seed and no
    convergence criterion — which is the point of a Tier 0 baseline.

    The thermal channel is the exact first-order solution of
    `d(theta_TO)/dt = (theta_ss(K(t), Ta(t)) - theta_TO) / tau_oil`, integrated on
    the sensor grid. `theta_ss` is the true fixed point (Phase 2 fix 1), so this
    baseline is *not* handicapped by the steady-state error the surrogate used to
    carry; the only thing it does differently from the reference is freeze Arrhenius
    at the mean temperature. That makes the comparison a clean measurement of
    convexity rather than a straw man.

    The gas channel then holds the generation rate constant:

        c_i(t) = c_i(0) + k_gen_i * V_arr_i(theta_HS_mean) * t - k_dis_i * c_i(0) * t

    and DP uses the same frozen factor.
    """

    def __init__(self, T: float = TW, n_sensors: int = N_SENSORS,
                 tau: float = tau_oil):
        self.T = T
        self.n_sensors = n_sensors
        self.tau = tau

    # ── thermal ────────────────────────────────────────────────────────────
    def theta_TO_trajectory(self, x0_TO: float, K_s: np.ndarray,
                            Ta_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact piecewise-linear-theta_ss solution on the sensor grid.

        Uses the closed-form recurrence for a linear driving term over each
        interval, which avoids the quadrature blow-up that the trapezoid form of
        `exp(s/tau)` suffers when T/tau is large:

            theta_{k+1} = theta_k e^{-r} + ss_k (1 - e^{-r})
                          + (ss_{k+1} - ss_k) (r - 1 + e^{-r}) / r,   r = ds/tau
        """
        s = np.linspace(0.0, self.T, self.n_sensors)
        ss = true_fixed_point_np(K_s, Ta_s)
        ds = self.T / (self.n_sensors - 1)
        r = ds / self.tau
        er = np.exp(-r)
        coef = (r - 1.0 + er) / r

        theta = np.empty(self.n_sensors)
        theta[0] = x0_TO
        for k in range(self.n_sensors - 1):
            theta[k + 1] = (theta[k] * er + ss[k] * (1.0 - er)
                            + (ss[k + 1] - ss[k]) * coef)
        return s, theta

    # ── the baseline ───────────────────────────────────────────────────────
    def predict(self, x0: np.ndarray, K_s: np.ndarray, Ta_s: np.ndarray,
                t_eval: np.ndarray, DP_0: float = DP0) -> DailyMeanResult:
        s, theta_TO_grid = self.theta_TO_trajectory(float(x0[0]), K_s, Ta_s)
        theta_HS_grid = np.array([hot_spot_ETC_np(float(theta_TO_grid[i]),
                                                  float(K_s[i]))
                                  for i in range(self.n_sensors)])
        theta_HS_mean = float(np.trapz(theta_HS_grid, s) / (s[-1] - s[0]))

        # The single frozen Arrhenius factor. The partial-discharge factor is
        # likewise taken at the mean load, as a practitioner would.
        V = arrhenius(np.array([theta_HS_mean]), E_act)[0]              # (5,)
        K_mean = float(np.trapz(K_s, s) / (s[-1] - s[0]))
        V = V.copy()
        V[1] = V[1] * (1.0 + PD_gain * max(K_mean - K_PD_onset, 0.0) ** 2)

        t_eval = np.asarray(t_eval, dtype=float)
        theta_TO = np.interp(t_eval, s, theta_TO_grid)
        gases = (x0[1:][None, :]
                 + (k_gen * V)[None, :] * t_eval[:, None]
                 - (k_dis * x0[1:])[None, :] * t_eval[:, None])

        V_DP = float(arrhenius(np.array([theta_HS_mean]), np.array([E_ACT_DP]))[0][0])
        DP = 1.0 / (1.0 / DP_0 + k0_aging * V_DP * t_eval)

        return DailyMeanResult(theta_TO=theta_TO, gases=gases, DP=DP,
                               theta_HS_mean=theta_HS_mean, V_arr_at_mean=V)


def resolved_reference(theta_HS_grid: np.ndarray, K_s: np.ndarray,
                       x0_gas: np.ndarray, t_eval: np.ndarray, T: float = TW,
                       DP_0: float = DP0):
    """The same construction with Arrhenius integrated along the trajectory.

    The only difference from `DailyMeanArrhenius.predict` is where Arrhenius is
    evaluated, so the pair isolates convexity. Used by the O-8 measurement to
    express the gap in ppm and in DP units as well as a dimensionless ratio.
    """
    s = np.linspace(0.0, T, len(theta_HS_grid))
    V = arrhenius(theta_HS_grid, E_act)                       # (ns, 5)
    pd = 1.0 + PD_gain * np.clip(K_s - K_PD_onset, 0.0, None) ** 2
    V = V.copy()
    V[:, 1] = V[:, 1] * pd
    gen = k_gen[None, :] * V
    ds = T / (len(s) - 1)
    trap = 0.5 * (gen[:-1] + gen[1:]) * ds
    cum = np.vstack([np.zeros((1, N_GAS)), np.cumsum(trap, axis=0)])

    t_eval = np.asarray(t_eval, dtype=float)
    F = np.stack([np.interp(t_eval, s, cum[:, g]) for g in range(N_GAS)], axis=1)
    gases = x0_gas[None, :] + F - (k_dis * x0_gas)[None, :] * t_eval[:, None]

    V_DP = arrhenius(theta_HS_grid, np.array([E_ACT_DP]))[:, 0]
    trap_dp = 0.5 * (V_DP[:-1] + V_DP[1:]) * ds
    cum_dp = np.concatenate([[0.0], np.cumsum(trap_dp)])
    F_dp = np.interp(t_eval, s, cum_dp)
    DP = 1.0 / (1.0 / DP_0 + k0_aging * F_dp)
    return gases, DP


__all__ = ["E_ACT_DP", "activation_energies_kJ", "arrhenius",
           "jensen_gap_from_trajectory", "jensen_gap_sinusoidal",
           "DailyMeanArrhenius", "DailyMeanResult", "resolved_reference"]
