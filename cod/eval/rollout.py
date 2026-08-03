"""Multi-window CHI / DP lifetime rollout.

Ported from n12 cell 3: `plot_chi_trajectory`'s inner `predict_chi_lifetime`
(L1705) and `plot_chi_model_dp` (L1811). Data only — plotting stays out of the
package, and the notebook is not allowed to hold science.

The two source functions differ in one respect, and it is the interesting one:

    dp_source="reference"  DP is advanced from `theta_TO_ss_ETC(K_w, Ta_w)`, i.e.
                           reference physics. EOL then does not depend on the
                           model at all, so it cannot reflect model quality.
    dp_source="model"      DP is advanced from the model's predicted theta_TO, so
                           a thermal bias shows up as an EOL shift: a negative
                           bias gives a conservatively long life, a positive bias
                           an optimistically short one.

PHASE 2 FIX 1 lands here as `steady_state`, which now defaults to the true fixed
point. Under v57 this rollout used formula B while the model's own attractor was
formula C and every IC came from formula A — three different steady states in one
experiment.

That matters for audit M-5. The manuscript explains a -3 degC rollout bias by a
staircase in the ETC term at K = 1, but B and C are *identical* at K = 1 (the load
factor is 1, so every exponent gives 1) and the only non-smooth element, the Rf
clamp, is inactive there. Over the rollout's K_base range of 0.85-1.10 the C-B
disagreement runs +0.67 to -1.03 degC and flips sign, so it can account for about
1 degC at most, not 3. With all three sites unified there is no formula mismatch
left to appeal to, and the bias must be re-diagnosed against the retrained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cod.data.physics import (
    B_aging,
    DP0,
    DP_EOL,
    N_SENSORS,
    T_ref,
    TW,
    fast_rhs_np,
    hot_spot_ETC_np,
    k0_aging,
)
from scipy.integrate import solve_ivp

from cod.data.steady_state import gas_ic_from_ss, true_fixed_point_np
from cod.training.losses import compute_chi


def cyclic_endpoint_theta(K_s: np.ndarray, Ta_s: np.ndarray,
                          theta_seed: float, T: float = TW,
                          rtol: float = 1e-8, tol: float = 1e-6,
                          max_cycles: int = 20) -> float:
    """Top-oil at the end of a window, for a unit already cycling on that window.

    Repeats the window's own forcing until the endpoint stops moving. This is the
    reference that has the intra-window ripple's phase lag **already in it**, which
    is what separates "lagged by the ripple" — not model error — from "the model
    got the temperature wrong" (O-9, `audit_port/BIAS_DIAGNOSIS.md`).

    Integrates `fast_rhs_np`'s thermal component with RK45, i.e. the same reference
    physics every other metric is scored against. The closed-form recurrence in
    `DailyMeanArrhenius` is *not* usable here and the reason is worth recording:
    it drives theta_TO toward `true_fixed_point_np(K, Ta)` as a fixed target, but
    the real ODE's `fac_n` depends on theta_TO itself through the copper-resistance
    correction `Rf = 1 + alpha_Cu (theta_HS - T_HS_ref)`. The two agree at
    equilibrium and drift apart during the transient — measured at 0.11 degC at
    K = 0.85 and 0.30 degC at K = 1.10, growing with load. Against a metric whose
    job is to resolve model error of order 0.5 degC, a reference carrying its own
    0.3 degC load-dependent error is the same class of defect O-9 just removed.

    Only the thermal state is integrated. The gas coupling is one-way, so
    `d(theta_TO)/dt` is a closed scalar ODE and there is no reason to carry five
    more states through the burn-in.

    This reference is physical rather than formulaic: it is where the real unit
    would be at the window end, which is not a function of which steady-state
    approximation the experiment was scored with. Passing `formula_B` as the
    rollout's `steady_state` to reproduce v57 does not change it.

    `tol` is the fixed-point tolerance and is deliberately looser than `rtol`.
    Setting it below the integrator's own reproducibility makes the iteration
    never converge and run to `max_cycles` every call — measured at 40 cycles and
    10 s per window instead of 6 cycles and 0.5 s, for a difference of 1e-5 degC.
    Each cycle contracts the mismatch by `exp(T / tau_oil)` = 122x, so six is
    already far past what a 0.5 degC metric can resolve.
    """
    tau = np.linspace(0.0, T, len(K_s))
    x = np.zeros(6)

    def d_theta(t, y):
        x[0] = y[0]
        return [fast_rhs_np(x, float(np.interp(t, tau, K_s)),
                            float(np.interp(t, tau, Ta_s)))[0]]

    theta = float(theta_seed)
    for _ in range(max_cycles):
        sol = solve_ivp(d_theta, [0.0, T], [theta], method="RK45",
                        t_eval=[T], rtol=rtol, atol=rtol * 1e-2)
        nxt = float(sol.y[0, -1])
        if abs(nxt - theta) < tol:
            return nxt
        theta = nxt
    return theta


def window_forcing(w: int, K_base: float, T: float = TW,
                   n_sensors: int = N_SENSORS):
    """The forcing of rollout window `w`: `(K_w, Ta_w, K_s, Ta_s)`.

    One definition, used by `chi_lifetime_rollout` and by `reference_rollout`.
    Restating it in the reference would let the two drift apart, and a reference
    driven by even slightly different forcing measures forcing difference rather
    than model error — the same class of defect O-9 found in `theta_bias`.

    Seasonal: `K_w` swings +-0.05 over the year and `Ta_w` from 15 to 39 degC.
    Intra-window: a full sine period of +-0.05 in K and +-2 degC in ambient.
    """
    tau = np.linspace(0.0, T, n_sensors)
    t_day = w * T / 1440
    K_w = K_base + 0.05 * np.sin(2 * np.pi * t_day / 365)
    Ta_w = 27.0 - 12.0 * np.cos(2 * np.pi * t_day / 365)
    Ta_s = (Ta_w + 2.0 * np.sin(2 * np.pi * tau / T)).astype(np.float32)
    K_s = np.clip(K_w + 0.05 * np.sin(2 * np.pi * tau / T),
                  0.3, 1.5).astype(np.float32)
    return float(K_w), float(Ta_w), K_s, Ta_s


def _advance_dp(DP_cur: float, theta_pts: np.ndarray, K_pts: np.ndarray,
                t_pts: np.ndarray) -> float:
    """One window of DP depolymerisation from a resolved thermal trajectory.

    Factored out so the model rollout and the reference rollout advance DP by the
    *same* quadrature. If they differed, the EOL gap between them would mix
    thermal error with integration error and could not be attributed.
    """
    V = np.array([
        np.exp(B_aging * (1.0 / T_ref
                          - 1.0 / (hot_spot_ETC_np(float(theta_pts[i]),
                                                   float(K_pts[i])) + 273.15)))
        for i in range(len(theta_pts))
    ])
    return max(1.0 / (1.0 / DP_cur + k0_aging * np.trapz(V, t_pts)), DP_EOL)


@dataclass
class ReferenceRollout:
    """The same rollout under reference physics, for scoring a model against.

    O-7 asks for a reference integrated continuously across the horizon rather
    than restarted per window, which is what this is: `x` carries over from one
    window to the next with no reset, so it accumulates exactly as the model's own
    state does and the difference between them is model error alone.
    """

    K_base: float
    years: np.ndarray
    theta_TO_end: np.ndarray     # reference top-oil at each window end
    theta_pts: np.ndarray        # (n_windows, n_eval) resolved trajectory
    dp: np.ndarray
    reached_eol: bool

    @property
    def eol_years(self) -> float:
        return float(self.years[-1])


def reference_rollout(K_base: float, max_years: int = 50, n_eval: int = 20,
                      T: float = TW, steady_state=None,
                      max_windows: int | None = None,
                      rtol: float = 1e-9, atol: float = 1e-11
                      ) -> ReferenceRollout:
    """`chi_lifetime_rollout` with the reference ODE in place of the model.

    Identical initial condition, identical forcing (`window_forcing`), identical
    DP quadrature (`_advance_dp`), identical window-to-window carry-over. The only
    thing replaced is what produces the trajectory: RK45 on `fast_rhs_np` instead
    of `model(x0, u, t)`.

    That is what makes the pair a measurement. Everything a rollout does other
    than predict the temperature is shared, so `model - reference` at a window end
    is the model's thermal error at that point in the horizon, and the gap between
    the two EOL years is the lifetime consequence of exactly that error.
    """
    if steady_state is None:
        steady_state = true_fixed_point_np
    windows_per_year = int(365 * 1440 / T)
    if max_windows is None:
        max_windows = max_years * windows_per_year

    t_eval_np = np.linspace(T * 0.01, T, n_eval)
    tau_np = np.linspace(0, T, N_SENSORS)

    Ta_init = 27.0
    x = np.concatenate([[float(steady_state(K_base, Ta_init))],
                        gas_ic_from_ss(K_base, Ta_init,
                                       steady_state=steady_state)]).astype(float)
    DP_cur = DP0
    yr_l, th_l, dp_l, pts_l = [], [], [], []
    reached_eol = False

    for w in range(max_windows):
        _, _, K_s, Ta_s = window_forcing(w, K_base, T)
        K_sf, Ta_sf = K_s.astype(float), Ta_s.astype(float)

        sol = solve_ivp(
            lambda t, y: fast_rhs_np(y, float(np.interp(t, tau_np, K_sf)),
                                     float(np.interp(t, tau_np, Ta_sf))),
            [0.0, T], x, method="RK45", t_eval=t_eval_np, rtol=rtol, atol=atol)
        theta_pts = sol.y[0]
        K_interp = np.interp(t_eval_np, tau_np, K_sf)
        DP_cur = _advance_dp(DP_cur, theta_pts, K_interp, t_eval_np)

        yr_l.append(w * T / 1440 / 365)
        th_l.append(float(theta_pts[-1]))
        dp_l.append(DP_cur)
        pts_l.append(theta_pts.copy())
        x = sol.y[:, -1].copy()

        if DP_cur <= DP_EOL + 1.0:
            reached_eol = True
            break

    return ReferenceRollout(K_base=K_base, years=np.array(yr_l),
                            theta_TO_end=np.array(th_l),
                            theta_pts=np.stack(pts_l), dp=np.array(dp_l),
                            reached_eol=reached_eol)


@dataclass
class RolloutResult:
    """One K_base scenario."""

    K_base: float
    years: np.ndarray
    chi: np.ndarray
    dp: np.ndarray
    theta_TO_end: np.ndarray      # model prediction at the end of each window
    theta_ss_ref: np.ndarray      # steady state of the window-MEAN forcing
    theta_cyc_ref: np.ndarray     # cyclic endpoint of the window's OWN forcing
    reached_eol: bool
    theta_pts: np.ndarray | None = None   # (n_windows, n_eval) resolved trajectory

    @property
    def eol_years(self) -> float:
        return float(self.years[-1])

    @property
    def chi_violations(self) -> int:
        """Windows where CHI rose. Should be zero if the model is monotone."""
        return int(np.sum(np.diff(self.chi) > 1e-4))

    @property
    def theta_bias(self) -> np.ndarray:
        """Model error at the window end, against the cyclic endpoint.

        FIXED (O-9, `audit_port/BIAS_DIAGNOSIS.md`). This used to return
        `theta_TO_end - theta_ss_ref`, which subtracts the steady state of the
        window's *mean* forcing from a value that is a *lagged* response to that
        forcing's ripple. With `tau_oil = 150` min against a 720 min window
        carrying a full sine period, the response lags 52.6 deg, so at the window
        end it is still below the mean it is returning to. A model with **zero
        error** scored -2.86 to -3.88 degC on that definition, which is where the
        manuscript's unexplained "-3 degC bias" came from.

        `theta_cyc_ref` is the endpoint a unit already cycling on this window's
        forcing would be at, so it carries the same lag and the difference is
        model error. An exact RK45 integration scores -0.002 degC against it.

        The old quantity is still available as `theta_ss_offset`; it is a real
        diagnostic of how far the operating point sits from equilibrium, just not
        a model error.
        """
        return self.theta_TO_end - self.theta_cyc_ref

    @property
    def theta_ss_offset(self) -> np.ndarray:
        """Distance from the equilibrium of the window-mean forcing.

        What `theta_bias` returned before O-9. Nonzero for a perfect model; see
        `theta_bias`. Kept because "how far from equilibrium is this operating
        point" is worth knowing, under a name that does not invite reading it as
        error.
        """
        return self.theta_TO_end - self.theta_ss_ref


@torch.no_grad()
def chi_lifetime_rollout(model, K_base: float, max_years: int = 50,
                         n_eval: int = 20, T: float = TW,
                         dp_source: str = "model", steady_state=None,
                         device=None, max_windows: int | None = None,
                         teacher_states: np.ndarray | None = None,
                         cyc_cache: dict | None = None
                         ) -> RolloutResult:
    """Roll the model forward window by window until DP reaches end of life.

    Each window: K_w = K_base + 0.05 sin(2 pi day / 365),
    Ta_w = 27 - 12 cos(2 pi day / 365), with intra-window K and Ta ripples of
    +/-0.05 and +/-2 degC. The next window's IC is the model's own prediction at
    the end of the previous one — theta_TO is continuous across windows and is
    deliberately not reset (the v55-B change).

    `steady_state` defaults to `true_fixed_point_np` (Phase 2 fix 1). Pass
    `formula_B` to reproduce v57.

    `max_windows` caps the horizon directly, for diagnostics that need many
    scenarios rather than long ones. `None` leaves the `max_years` behaviour
    untouched, so nothing that does not pass it can change.

    `teacher_states` (n_windows, 6) replaces the carried-over state at the start
    of each window with a supplied one — in practice `reference_rollout`'s
    trajectory. The free-running rollout above measures single-window error *plus*
    everything accumulated before it; teacher forcing measures single-window error
    alone, and the difference between the two is the accumulation. Without both,
    a rollout error cannot be attributed and O-7's "separate systematic bias from
    random error" has nothing to work with. `None` leaves the free-running
    behaviour exactly as it was.

    `cyc_cache` lets a caller share the cyclic-endpoint burn-ins across rollouts.
    The cyclic endpoint is a property of the **forcing**, not of the model, so
    scoring a free-running and a teacher-forced rollout over the same scenario
    recomputes an identical set of burn-ins twice when each builds its own cache.
    That is the dominant cost of a long rollout: each entry is several RK45 cycles
    and, over a year, every window has a distinct seasonal `(K_w, Ta_w)`. Passing
    one dict across both halves of the comparison halves it. `None` keeps the
    previous self-contained behaviour.
    """
    if steady_state is None:
        steady_state = true_fixed_point_np
    if dp_source not in ("model", "reference"):
        raise ValueError("dp_source must be 'model' or 'reference'")
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    windows_per_year = int(365 * 1440 / T)
    if max_windows is None:
        max_windows = max_years * windows_per_year
    t_q = torch.linspace(T * 0.01, T, n_eval, device=device).unsqueeze(-1)
    t_eval_np = t_q.squeeze(-1).cpu().numpy()
    sensor_grid = np.linspace(0, T, N_SENSORS)

    Ta_init = 27.0
    theta_ss0 = float(steady_state(K_base, Ta_init))
    gas_ic = gas_ic_from_ss(K_base, Ta_init, steady_state=steady_state)
    x_cur = torch.tensor(np.concatenate([[theta_ss0], gas_ic]).astype(np.float32),
                         dtype=torch.float32, device=device).unsqueeze(0)
    DP_cur = DP0

    chi_l, dp_l, yr_l, th_l, ss_l, cy_l, pts_l = [], [], [], [], [], [], []
    if cyc_cache is None:
        cyc_cache = {}
    reached_eol = False

    for w in range(max_windows):
        if teacher_states is not None:
            if w >= len(teacher_states):
                break
            x_cur = torch.tensor(np.asarray(teacher_states[w], np.float32),
                                 device=device).unsqueeze(0)

        K_w, Ta_w, K_s, Ta_s = window_forcing(w, K_base, T, N_SENSORS)

        s_t = torch.tensor(np.concatenate([K_s, Ta_s]), dtype=torch.float32,
                           device=device).unsqueeze(0)
        xp = model(x_cur.expand(n_eval, -1).contiguous(),
                   s_t.expand(n_eval, -1).contiguous(), t_q)

        K_interp = np.interp(t_eval_np, sensor_grid, K_s)
        if dp_source == "reference":
            theta_for_dp = np.full(n_eval, float(steady_state(K_w, Ta_w)))
        else:
            theta_for_dp = xp[:, 0].cpu().numpy().astype(float)

        DP_cur = _advance_dp(DP_cur, theta_for_dp, K_interp, t_eval_np)
        pts_l.append(np.asarray(xp[:, 0].cpu().numpy(), float))

        # CHI is computed AFTER the DP update, so CHI = 0 exactly when DP = EOL.
        chi_l.append(float(compute_chi(xp[-1:], DP_cur).item()))
        dp_l.append(DP_cur)
        yr_l.append(w * T / 1440 / 365)
        th_l.append(float(xp[-1, 0].item()))
        ss_l.append(float(steady_state(K_w, Ta_w)))
        # Seeded from the window-mean equilibrium rather than from the model, so
        # the reference stays independent of what is being scored against it.
        # (K_w, Ta_w) is a function of day-of-year alone, so the burn-in repeats
        # every 730 windows; a 50-year rollout would otherwise redo each one 50
        # times. Keyed on the values rather than the index so it stays correct if
        # the seasonal forcing is ever changed.
        ck = (round(K_w, 9), round(Ta_w, 9))
        if ck not in cyc_cache:
            # Seeded from the previous window's answer where there is one: the
            # seasonal forcing moves by ~0.01 degC per window, so the iteration
            # starts inside tolerance and costs one cycle instead of six.
            seed = cy_l[-1] if cy_l else ss_l[-1]
            cyc_cache[ck] = cyclic_endpoint_theta(
                K_s.astype(float), Ta_s.astype(float), seed, T=T)
        cy_l.append(cyc_cache[ck])

        x_cur = xp[-1:].clone()

        if DP_cur <= DP_EOL + 1.0:
            reached_eol = True
            break

    return RolloutResult(
        K_base=K_base, years=np.array(yr_l), chi=np.array(chi_l),
        dp=np.array(dp_l), theta_TO_end=np.array(th_l),
        theta_ss_ref=np.array(ss_l), theta_cyc_ref=np.array(cy_l),
        reached_eol=reached_eol, theta_pts=np.stack(pts_l) if pts_l else None,
    )


def chi_lifetime_sweep(model, K_scenarios=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
                       **kwargs) -> dict[float, RolloutResult]:
    """The six-scenario sweep the source plots. Returns data, not figures."""
    return {K: chi_lifetime_rollout(model, K, **kwargs) for K in K_scenarios}


def bias_table(results: dict[float, RolloutResult]) -> str:
    """theta_TO bias per scenario — the quantity audit M-5 disputes."""
    lines = ["| K_base | EOL (yr) | reached EOL | mean bias degC | "
             "CHI violations |", "|---|---|---|---|---|"]
    for K, r in sorted(results.items()):
        lines.append(f"| {K:.2f} | {r.eol_years:.1f} | {r.reached_eol} | "
                     f"{r.theta_bias.mean():+.2f} | {r.chi_violations} |")
    return "\n".join(lines)


__all__ = ["RolloutResult", "ReferenceRollout", "chi_lifetime_rollout",
           "chi_lifetime_sweep", "reference_rollout", "window_forcing",
           "bias_table", "cyclic_endpoint_theta"]
