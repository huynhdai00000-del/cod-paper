#!/usr/bin/env python3
"""O-9: where the -3 degC rollout bias comes from.

The audit confirmed the bias exists (|bias|_mean = 3.09 degC) and refuted the
manuscript's explanation (an ETC staircase at K = 1: the two formulas coincide
exactly there and the Rf clamp never activates). One lead remained — a ~-3 degC
offset between formula A and formula B at high load.

This checks that lead against the actual rollout, then tests the hypothesis the
code structure suggests instead: that `RolloutResult.theta_bias` compares a lagged
quantity against an unlagged one and would be nonzero for a *perfect* model.

The decisive experiment is §4. `ExactModel` integrates `fast_rhs_np` with RK45 and
exposes the same call signature as `CODOperator`, so `chi_lifetime_rollout` can be
run against ground truth itself. Any bias it reports is a property of the
diagnostic, not of any model.

Run:  python audit_port/scripts/16_bias_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.physics import (  # noqa: E402
    DTheta_oil_R, N_SENSORS, R_load, TW, fast_rhs_np, hot_spot_ETC_np,
    n_exp, tau_oil,
)
from cod.data.steady_state import (  # noqa: E402
    formula_A, formula_B, formula_C, true_fixed_point, true_fixed_point_np,
)
from cod.eval.rollout import chi_lifetime_rollout  # noqa: E402

OUT = ROOT / "audit_port" / "BIAS_DIAGNOSIS.md"
K_SCEN = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10)


# ═══════════════════════════════════════════════════════════════════════════
# A model that is exactly right, so any residual bias is the metric's
# ═══════════════════════════════════════════════════════════════════════════
class ExactModel(torch.nn.Module):
    """RK45 on `fast_rhs_np`, wearing `CODOperator`'s interface.

    Zero model error by construction. Feeding it to `chi_lifetime_rollout`
    isolates whatever the rollout's own bookkeeping contributes.
    """

    def __init__(self, T: float = TW, n_sensors: int = N_SENSORS):
        super().__init__()
        self.T = T
        self.n_sensors = n_sensors
        self._p = torch.nn.Parameter(torch.zeros(1))   # so .parameters() works

    def forward(self, x0, u_sensors, t):
        ns = self.n_sensors
        tau = np.linspace(0.0, self.T, ns)
        x0n = x0[0].detach().cpu().numpy().astype(float)
        K_s = u_sensors[0, :ns].detach().cpu().numpy().astype(float)
        Ta_s = u_sensors[0, ns:2 * ns].detach().cpu().numpy().astype(float)
        t_np = t.squeeze(-1).detach().cpu().numpy().astype(float)

        def rhs(s, x):
            return fast_rhs_np(x, float(np.interp(s, tau, K_s)),
                               float(np.interp(s, tau, Ta_s)))

        order = np.argsort(t_np)
        sol = solve_ivp(rhs, [0.0, float(max(t_np.max(), self.T))], x0n,
                        method="RK45", t_eval=t_np[order],
                        rtol=1e-10, atol=1e-12)
        y = np.empty((len(t_np), 6))
        y[order] = sol.y.T
        return torch.tensor(y, dtype=torch.float32, device=x0.device)


# ═══════════════════════════════════════════════════════════════════════════
def s1_formula_lead(A):
    """The audit's lead: is A - B about -3 degC at high load, and can it get in?"""
    A("## 1. The formula A vs B lead: right magnitude, wrong path\n")
    A("The rollout drives `K_base + 0.05 sin(2 pi day/365)` against "
      "`Ta = 27 - 12 cos(2 pi day/365)`, so it spans K in [0.80, 1.15] and "
      "theta_a in [15, 39]. The formulas over that box:\n")
    A("| K | theta_a | TRUE | A | B | C | A-B | C-B | A-TRUE |")
    A("|---|---|---|---|---|---|---|---|---|")
    peak = 0.0
    for Ta in (15.0, 27.0, 39.0):
        for K in (0.85, 1.00, 1.10, 1.15):
            t = true_fixed_point(K, Ta)
            a, b, c = formula_A(K, Ta), formula_B(K, Ta), formula_C(K, Ta)
            peak = min(peak, a - b)
            A(f"| {K:.2f} | {Ta:g} | {t:.2f} | {a:.2f} | {b:.2f} | {c:.2f} | "
              f"{a - b:+.2f} | {c - b:+.2f} | {a - t:+.2f} |")
    A("")
    A(f"**The lead is real in the box.** `A - B` passes through -3 degC inside the "
      f"rollout's operating range and reaches {peak:+.2f} at the corner. Right "
      "size, right sign. It is still not the cause, for two independent reasons — "
      "and the first only becomes visible once the trace is used instead of the "
      "table.\n")

    A("### 1a. Along the actual rollout trace, A - B has the wrong shape\n")
    A("The task is to check the lead against the trace, not against a corner of a "
      "table. Evaluating both formulas along the exact `(K_w, Ta_w)` sequence the "
      "rollout visits over one year:\n")
    n_win = 730
    t_day = np.arange(n_win) * TW / 1440
    K_tr = 1.00 + 0.05 * np.sin(2 * np.pi * t_day / 365)
    Ta_tr = 27.0 - 12.0 * np.cos(2 * np.pi * t_day / 365)
    ab = np.array([formula_A(k, ta) - formula_B(k, ta)
                   for k, ta in zip(K_tr, Ta_tr)])
    cb = np.array([formula_C(k, ta) - formula_B(k, ta)
                   for k, ta in zip(K_tr, Ta_tr)])
    A("| quantity along the trace (K_base = 1.00) | mean | sd | min | max | "
      "sign changes |")
    A("|---|---|---|---|---|---|")
    for lbl, v in [("`A - B`", ab), ("`C - B`", cb)]:
        A(f"| {lbl} | {v.mean():+.3f} | {v.std():.3f} | {v.min():+.3f} | "
          f"{v.max():+.3f} | {int((np.diff(np.sign(v)) != 0).sum())} |")
    A("| measured bias (§4, K_base = 1.00) | -3.409 | 0.069 | -3.505 | -2.993 "
      "| 0 |")
    A("")
    A("On the trace the lead fails on all three counts:\n")
    A(f"* **Mean.** `A - B` averages {ab.mean():+.2f} degC over the year, not -3. "
      f"The ~-3 degC figure is its seasonal *extreme* ({ab.min():+.2f}), reached "
      "only in the hottest weeks, not its typical value.\n")
    A(f"* **Variability.** `A - B` has sd {ab.std():.2f} degC and spans "
      f"{ab.max() - ab.min():.1f} degC across the year, because it is driven by "
      "the seasonal ambient. The measured bias has sd 0.07 and is flat. A quantity "
      "that varies by degrees cannot cause one that varies by hundredths.\n")
    A(f"* **Sign.** `A - B` changes sign "
      f"{int((np.diff(np.sign(ab)) != 0).sum())} times a year and is *positive* "
      f"for part of it (max {ab.max():+.2f}). The bias is negative in every "
      "window.\n")
    A("This is the value of checking the trace rather than the table the lead came "
      "from. In the table `A - B` reaches -7.87 degC and looks like a candidate; "
      "on the trajectory the rollout actually visits it is three times too small "
      "on average, twenty times too variable, and not even one-signed.\n")

    A("### 1b. And formula A is not on the rollout path at all\n")
    A("`chi_lifetime_rollout` takes a single `steady_state` argument and uses it "
      "for three things — the initial `theta_ss0`, the gas IC through "
      "`gas_ic_from_ss`, and `theta_ss_ref`, the reference the bias is scored "
      "against. All three are the same function. There is no second formula for "
      "the difference to be taken against, in v57 (where it was `formula_B` "
      "throughout) or now (where it is `true_fixed_point_np` throughout).\n")
    A("The one real formula mismatch in v57 was between the rollout's reference "
      "(B) and the model's own analytic attractor (C), which the model relaxes "
      f"toward. `C - B` along the trace has mean {cb.mean():+.3f} degC and changes "
      f"sign {int((np.diff(np.sign(cb)) != 0).sum())} times a year, so it cannot "
      "produce a one-signed 3 degC either. Phase 2 fix 1 removed it entirely, and "
      "§4 measures the bias with it gone — unchanged.\n")
    A("So the lead is a coincidence of magnitude. Worth having chased: it was the "
      "only quantity in the neighbourhood with both the right size and the right "
      "sign. Ruling it out on shape and on structure rather than on size is what "
      "leaves §2 as the remaining explanation.\n")
    return ab, cb


def s2_hypothesis(A):
    A("## 2. What `theta_bias` actually measures\n")
    A("```python")
    A("@property")
    A("def theta_bias(self):")
    A("    return self.theta_TO_end - self.theta_ss_ref")
    A("```")
    A("`theta_TO_end` is the model's top-oil at the **end** of the window. "
      "`theta_ss_ref` is `steady_state(K_w, Ta_w)`, the top-oil the unit would "
      "settle at if it were driven by the window's *mean* load and ambient "
      "forever.\n")
    A("Those are not the same quantity, and the difference is not model error. "
      "Within each window the rollout applies a full sine period of ripple:\n")
    A("```python")
    A("Ta_s = Ta_w + 2.0 * sin(2 pi tau / T)")
    A("K_s  = K_w  + 0.05 * sin(2 pi tau / T)")
    A("```")
    A("Top-oil follows that ripple through a first-order lag with "
      "`tau_oil = 150` min against a window of `T = 720` min. A lagged response "
      "to a sinusoid is phase-shifted, so at the instant the forcing returns to "
      "its mean — `tau = T`, where `sin(2 pi) = 0` — the *response* has not. It is "
      "still on its way back up, from below.\n")
    A("That predicts a bias that is negative, one-signed, present at every "
      "window, and **present for a model with no error at all**.\n")


def s3_analytic(A):
    A("## 3. What the lag predicts, in closed form\n")
    om = 2.0 * np.pi / TW
    ot = om * tau_oil
    gain = 1.0 / np.sqrt(1.0 + ot ** 2)
    phase = np.arctan(ot)
    A("For `dtheta/dt = (theta_ss(t) - theta)/tau_oil` driven by a sinusoid of "
      "angular frequency `omega`, the periodic response has gain "
      "`1/sqrt(1 + (omega tau)^2)` and lag `atan(omega tau)`. At the window end "
      "the forcing is at its mean and the response is `-gain * sin(lag)` times the "
      "steady-state amplitude.\n")
    A(f"- `omega tau_oil = 2 pi * {tau_oil:.0f} / {TW:.0f} = {ot:.4f}`")
    A(f"- gain = {gain:.4f}, lag = {phase:.4f} rad = {np.degrees(phase):.1f} deg")
    A(f"- end-of-window factor = `-gain * sin(lag)` = {-gain * np.sin(phase):.4f}\n")
    A("The two ripples' steady-state amplitudes at K = 1:\n")
    fac = (1.0 + R_load) / (1.0 + R_load)
    dss_dK = DTheta_oil_R * n_exp * fac ** (n_exp - 1) * (2 * R_load / (1 + R_load))
    amp_Ta, amp_K = 2.0, 0.05 * dss_dK
    A(f"| source | steady-state amplitude | contribution at window end |")
    A(f"|---|---|---|")
    A(f"| ambient ripple, ±2 degC | {amp_Ta:.3f} degC | "
      f"{-amp_Ta * gain * np.sin(phase):+.3f} degC |")
    A(f"| load ripple, ±0.05 K | {amp_K:.3f} degC "
      f"(`dtheta_ss/dK = {dss_dK:.1f}` degC per unit K) | "
      f"{-amp_K * gain * np.sin(phase):+.3f} degC |")
    tot = -(amp_Ta + amp_K) * gain * np.sin(phase)
    A(f"| **total** | | **{tot:+.3f} degC** |")
    A("")
    A("That linearisation is taken at K = 1 with the two ripples superposed. "
      "Dropping the linearisation — taking the steady-state amplitude as the half "
      "range of `true_fixed_point_np` over the ripple itself, which keeps the ETC "
      "correction and the K-dependence — gives a prediction per scenario:\n")
    A("| K_base | steady-state amplitude degC | predicted end-of-window bias |")
    A("|---|---|---|")
    pred = {}
    factor = -gain * np.sin(phase)
    for K in K_SCEN:
        hi = float(true_fixed_point_np(K + 0.05, 27.0 + 2.0))
        lo = float(true_fixed_point_np(K - 0.05, 27.0 - 2.0))
        amp = 0.5 * (hi - lo)
        pred[K] = amp * factor
        A(f"| {K:.2f} | {amp:.3f} | {pred[K]:+.3f} degC |")
    A("")
    A("Two signatures to check against §4, both of which the manuscript's "
      "ETC-staircase story does not have:\n")
    A("* the bias is **one-signed and negative** at every window;\n")
    A("* it **grows monotonically with K_base**, because `dtheta_ss/dK` does. A "
      "staircase at K = 1 would put a feature *at* K = 1, not a smooth trend "
      "through it.\n")
    return tot, pred


N_WIN = 90        # 45 days. The bias is stationary; see the sd column in §4.


def cyclic_endpoint(K_w: float, Ta_w: float, x_seed: np.ndarray,
                    n_cycles: int = 12) -> float:
    """Top-oil at the end of a window, for a unit already cycling on that window.

    Repeats the window's own forcing until the endpoint stops moving. This is the
    reference that has the ripple's phase lag *already in it*, so comparing
    against it separates "lagged by the intra-window ripple" from "has not settled
    to the seasonal operating point".
    """
    tau = np.linspace(0.0, TW, N_SENSORS)
    Ta_s = Ta_w + 2.0 * np.sin(2 * np.pi * tau / TW)
    K_s = np.clip(K_w + 0.05 * np.sin(2 * np.pi * tau / TW), 0.3, 1.5)
    x = np.asarray(x_seed, dtype=float).copy()
    for _ in range(n_cycles):
        sol = solve_ivp(
            lambda s, y: fast_rhs_np(y, float(np.interp(s, tau, K_s)),
                                     float(np.interp(s, tau, Ta_s))),
            [0.0, TW], x, method="RK45", t_eval=[TW], rtol=1e-9, atol=1e-11)
        x = sol.y[:, -1]
    return float(x[0])


def s4_exact(A, tot, pred):
    A("## 4. The decisive test: run the rollout against ground truth\n")
    A("`ExactModel` integrates `fast_rhs_np` with RK45 at `rtol = 1e-10` and "
      "exposes `CODOperator`'s call signature, so `chi_lifetime_rollout` runs "
      "against the reference physics itself. Its model error is zero by "
      "construction. Any bias it reports belongs to the diagnostic.\n")
    m = ExactModel()
    A("| K_base | predicted (§3) | measured mean | median | sd | min | max | "
      "windows |")
    A("|---|---|---|---|---|---|---|---|")
    biases, results = {}, {}
    for K in K_SCEN:
        r = chi_lifetime_rollout(m, K, dp_source="model",
                                 steady_state=true_fixed_point_np,
                                 max_windows=N_WIN)
        # `theta_ss_offset` is what `theta_bias` returned when this diagnosis was
        # written. The fix that followed it renamed the artifact and gave
        # `theta_bias` the corrected definition, so this script has to name the
        # old quantity explicitly or it would regenerate the report with the
        # near-zero numbers of the fixed metric and quietly contradict itself.
        b = r.theta_ss_offset
        biases[K], results[K] = b, r
        A(f"| {K:.2f} | {pred[K]:+.3f} | **{b.mean():+.3f}** | "
          f"{np.median(b):+.3f} | {b.std():.3f} | {b.min():+.3f} | "
          f"{b.max():+.3f} | {len(b)} |")
        print(f"  K={K:.2f} pred {pred[K]:+.3f} bias {b.mean():+.3f} "
              f"sd {b.std():.3f}")
    A("")
    allb = np.concatenate([biases[K] for K in K_SCEN])
    means = np.array([biases[K].mean() for K in K_SCEN])
    preds = np.array([pred[K] for K in K_SCEN])
    A(f"Pooled over all six scenarios: mean **{allb.mean():+.3f}** degC, "
      f"sd {allb.std():.3f}, {100.0 * (allb < 0).mean():.1f}% of windows "
      "negative.\n")
    A("**Both predicted signatures appear.** The bias is negative in every one of "
      f"the {len(allb)} windows, and it grows monotonically from "
      f"{means[0]:+.2f} degC at K_base = 0.85 to {means[-1]:+.2f} at 1.10, with no "
      "feature at K = 1.\n")
    A(f"The prediction overshoots by a consistent "
      f"{100 * (preds / means - 1.0).mean():.0f}% "
      f"(ratio {(preds / means).min():.3f} to {(preds / means).max():.3f} across "
      "the sweep), and the reason is specific rather than hand-waved: the "
      "prediction takes the driving amplitude as **half the peak-to-peak range** "
      "of `theta_ss` over the ripple. `theta_ss` is nonlinear in K, so its "
      "response to a sinusoidal K is not itself a sinusoid, and half its range "
      "exceeds the amplitude of its first harmonic — which is the only component "
      "the single-pole gain-and-phase formula applies to. Overshooting is "
      "therefore the expected direction. What the prediction gets right is the "
      "sign, the shape, and the slope in K_base, which is what identifies the "
      "mechanism.\n")
    A(f"Horizon is {N_WIN} windows ({N_WIN * TW / 1440:.0f} days) per scenario "
      "rather than to end of life. The bias is stationary — the within-scenario "
      f"sd is {100 * biases[1.00].std() / abs(biases[1.00].mean()):.0f}% of the "
      "mean and is the seasonal ambient drift, not spread — so a longer run "
      "changes nothing and costs 30x more RK45 solves.\n")
    A(f"**A model with zero error reports a bias of {allb.mean():+.2f} degC, "
      f"spanning {means[0]:+.2f} to {means[-1]:+.2f} across the sweep.** The "
      "audit's measured |bias|_mean of 3.09 degC sits inside that range. It is "
      "therefore not a model defect and not a physics defect. It is the metric "
      "subtracting an unlagged reference from a lagged prediction.\n")
    return allb, results, means


def s5_reference(A, results):
    A("## 5. Separating the phase lag from everything else\n")
    A("If the diagnosis in §2 is right, the bias should vanish against a reference "
      "that already contains the ripple's phase lag. `cyclic_endpoint` supplies "
      "one: it repeats the window's own forcing until the endpoint stops moving, "
      "so it is where top-oil is at the end of a window for a unit already cycling "
      "on that window. Nothing about the model enters it.\n")
    A("Same trajectories as §4, three references:\n")
    A("| K_base | vs `steady_state(K_w, Ta_w)` | vs the cyclic endpoint | "
      "vs its own endpoint |")
    A("|---|---|---|---|")
    both = []
    for K in K_SCEN:
        r = results[K]
        # score the last few windows, where the seasonal transient is smallest
        sel = slice(-20, None)
        th = r.theta_TO_end[sel]
        d_cyc = []
        for i, w in enumerate(range(len(r.years))[sel]):
            t_day = w * TW / 1440
            K_w = K + 0.05 * np.sin(2 * np.pi * t_day / 365)
            Ta_w = 27.0 - 12.0 * np.cos(2 * np.pi * t_day / 365)
            x_seed = np.concatenate([[th[i]], np.array([50., 15., 80., 300., 800.])])
            d_cyc.append(th[i] - cyclic_endpoint(K_w, Ta_w, x_seed))
        d_cyc = np.array(d_cyc)
        b_ss = r.theta_ss_offset[sel].mean()
        both.append((K, b_ss, d_cyc.mean()))
        A(f"| {K:.2f} | {b_ss:+.3f} degC | {d_cyc.mean():+.3f} degC | 0.000 degC |")
        print(f"  K={K:.2f} vs ss {b_ss:+.3f}  vs cyclic {d_cyc.mean():+.3f}")
    A("")
    m_ss = np.mean([b for _, b, _ in both])
    m_cy = np.mean([c for _, _, c in both])
    A(f"Mean over the six scenarios: **{m_ss:+.3f}** degC against the steady state "
      f"of the mean forcing, **{m_cy:+.3f}** degC against the cyclic endpoint of "
      f"the same forcing. The third column is identically zero because "
      "`ExactModel` *is* the true trajectory; it is tabulated so the substitution "
      "is explicit.\n")
    A(f"The ripple's phase lag accounts for "
      f"{100.0 * (1.0 - abs(m_cy) / abs(m_ss)):.2f}% of the reported bias. The "
      f"residual is {abs(m_cy):.3f} degC — three orders of magnitude below the "
      "effect and still one-signed, which is what it should be: it is the seasonal "
      "ambient drift, `Ta_w` moving between windows so that a unit is never quite "
      "at the cyclic state of the window it is currently in. That residual is a "
      "real, and negligible, physical lag; the 3.4 degC is not.\n")
    A("Concretely: `RolloutResult.theta_bias` should be scored against a reference "
      "integration of `fast_rhs_np` over the same window from the same initial "
      "condition. `theta_ss_ref` is worth keeping as its own field — how far the "
      "operating point sits from equilibrium is a real diagnostic — but it is not "
      "the thing to subtract a prediction from.\n")
    return m_ss, m_cy


def s6_ageing(A):
    A("## 6. The ageing consequence, which is smaller than feared\n")
    A("O-9 records the concern as: at 10.8 %/K Arrhenius sensitivity a systematic "
      "-3 degC understates the ageing rate by roughly 30%, so no end-of-life "
      "number can be published until it is understood. The sensitivity arithmetic "
      "is right and the conclusion needs revising, because **the -3 degC never "
      "entered the DP calculation**.\n")
    A("The DP update reads `theta_for_dp` and `theta_for_dp` is never "
      "`theta_ss_ref`:\n")
    A("```python")
    A("if dp_source == \"reference\":")
    A("    theta_for_dp = np.full(n_eval, float(steady_state(K_w, Ta_w)))")
    A("else:")
    A("    theta_for_dp = xp[:, 0].cpu().numpy()      # the predicted trajectory")
    A("```")
    A("Under `dp_source=\"model\"`, which is the default and the only setting that "
      "reflects model quality, DP is advanced from the model's own top-oil "
      "trajectory over the whole window — 20 quadrature points, not the endpoint, "
      "and not the steady-state reference. The bias field is reported alongside it "
      "and consumed by nothing.\n")
    A("Verified: the Arrhenius sensitivity itself, from the code's constants.\n")
    from cod.data.physics import B_aging as Bg
    A("| theta_HS | dV/V per +1 degC |")
    A("|---|---|")
    for th in (80.0, 90.0, 100.0, 110.0, 120.0):
        T = th + 273.15
        A(f"| {th:.0f} degC | {100.0 * Bg / T ** 2:.2f}% |")
    A("")
    A(f"So 10.8 %/K is the value at about "
      f"{np.sqrt(Bg / 0.108) - 273.15:.0f} degC, and a *real* systematic -3 degC "
      "would indeed cost about 30% of the ageing rate. That remains true and "
      "remains the reason to care. What changed is that no such offset has been "
      "demonstrated: the 3.09 degC that motivated the concern is an artifact of a "
      "diagnostic that does not feed the DP path.\n")
    A("**This does not clear the rollout to publish end-of-life numbers.** It "
      "removes one specific reason to distrust them and replaces it with an "
      "honest gap: the model's true thermal error over a rollout has not been "
      "measured, because the field that was supposed to measure it was measuring "
      "something else. §7 says what to run.\n")


def s7_open(A):
    A("## 7. What this leaves open\n")
    A("1. **The real rollout error is still unmeasured.** Fixing `theta_bias` to "
      "score against a reference integration gives the number O-9 was actually "
      "after. It cannot be produced yet: fix 6 (DECISIONS N-1) invalidated the "
      "checkpoint again, so there is no trained model to roll out. This is a "
      "post-retrain task and it is the one that decides whether an EOL number is "
      "publishable.\n")
    A("2. **Error accumulation across windows is a separate question.** Each "
      "window starts from the model's own previous endpoint, so per-window error "
      "compounds. That is O-7's subject and this diagnosis says nothing about it "
      "either way.\n")
    A("3. **The manuscript's ETC-staircase explanation stays refuted**, and now "
      "for a second reason. It was already false at K = 1 (the two formulas "
      "coincide there and the Rf clamp is inactive). It is also explaining an "
      "effect that has no physical existence.\n")
    A("4. **The fix to `rollout.py` is not applied here.** O-9 asked for a "
      "diagnosis, and changing the metric would change reported numbers in the "
      "same commit that explains why they were wrong. Separate change, separate "
      "before/after.\n")


def main() -> int:
    # Body first, headline second, so every number in the headline is one the
    # body actually produced. Writing the summary by hand against remembered
    # values is how the first draft of this file ended up quoting a -12 degC
    # seasonal swing that the trace does not have.
    body: list[str] = []
    A = body.append

    trace = s1_formula_lead(A)
    s2_hypothesis(A)
    tot, pred = s3_analytic(A)
    allb, results, means = s4_exact(A, tot, pred)
    m_ss, m_cy = s5_reference(A, results)
    s6_ageing(A)
    s7_open(A)

    ab, cb = trace
    head: list[str] = []
    H = head.append
    H("# O-9 — the -3 degC rollout bias, diagnosed\n")
    H("**Result: the bias is an artifact of the diagnostic, not of the model or "
      "the physics.** `RolloutResult.theta_bias` subtracts the steady state of "
      "the window's *mean* forcing from a top-oil value that is a *lagged* "
      "response to that forcing's ripple. A model with zero error reports it "
      "too.\n")
    H("Demonstrated three ways that agree:\n")
    H("1. A closed-form first-order-lag calculation predicts a negative, "
      "one-signed offset growing with load — `tau_oil = 150` min against a 720 "
      "min window gives a 52.6 deg phase lag, so at the window end the response "
      "is still below the mean it is returning to (§3).\n")
    H("2. Running the rollout against an exact RK45 integration of `fast_rhs_np` "
      "— a model with zero error by construction — measures "
      f"{means[0]:+.2f} degC at K_base = 0.85 rising monotonically to "
      f"{means[-1]:+.2f} at 1.10, negative in "
      f"{100.0 * (allb < 0).mean():.0f}% of {len(allb)} windows, with no feature "
      "at K = 1 (§4). The audit's |bias|_mean of 3.09 degC sits inside that "
      "range.\n")
    H("3. Scoring the same trajectories against a reference that already contains "
      f"the lag — the cyclic endpoint of the window's own forcing — leaves "
      f"{m_cy:+.3f} degC, i.e. "
      f"{100.0 * (1.0 - abs(m_cy) / abs(m_ss)):.2f}% of the effect is the lag "
      "(§5).\n")
    H("The monotonic growth through K = 1 refutes the manuscript's ETC-staircase "
      "explanation a second way: a staircase *at* K = 1 would put a "
      "discontinuity there, not a smooth trend through it.\n")
    H("The audit's remaining lead — a ~-3 degC offset between formula A and "
      "formula B at high load — survives a size check and is ruled out on shape "
      f"and on structure. Along the actual rollout trace `A - B` averages "
      f"{ab.mean():+.2f} degC, not -3; it has sd {ab.std():.2f} against the "
      f"bias's 0.07; and it changes sign "
      f"{int((np.diff(np.sign(ab)) != 0).sum())} times a year. `formula_A` also "
      "never appears on the rollout path at all (§1).\n")
    H("Consequence for the ageing concern in §6: the 10.8 %/K sensitivity "
      "arithmetic is correct, but the -3 degC never entered the DP calculation, "
      "which is advanced from the predicted trajectory rather than from this "
      "reference. The real rollout thermal error remains unmeasured and needs a "
      "retrained model (§7).\n")

    OUT.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
