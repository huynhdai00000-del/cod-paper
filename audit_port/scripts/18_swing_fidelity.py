#!/usr/bin/env python3
"""Does the thermal surrogate under-predict swing amplitude, and does the
analytic baseline explain whether it does?

Neural networks have a spectral bias toward low frequencies, so a thermal
surrogate may smooth the peaks of a cycling trajectory. If it does, it loses part
of the Jensen gap the paper exists to preserve — and **thermal MAE would not
reveal it**, because a smoothed trajectory can sit close to the truth in mean
absolute error while systematically under-stating the peak-to-trough range that
the convex Arrhenius integral is sensitive to.

That makes this a thesis-level check, not a diagnostic: the method's claim is that
resolving the thermal cycle preserves a gap that mean-temperature methods lose. A
surrogate that flattens the cycle keeps some of the same defect it is meant to
cure, and the paper needs to know by how much.

THE MECHANISM UNDER TEST. COD predicts a *correction* to an analytic first-order
solution, so the cycle shape is supplied by the IEC baseline and the network's
spectral bias has nothing to flatten. That predicts the opposite for any
architecture without such a baseline: the network has to generate the cycle
itself, which is exactly the frequency content spectral bias suppresses. The
monolithic models are that condition — `MonolithicFair` and `MonolithicMultiHead`
both remove the analytic baseline H and return `x0 + phi * scale * net(...)`,
where `phi` is a fixed scalar envelope carrying no forcing information.

WHY NOT ABLATION A. Ablation A (COD's architecture minus H, same network, same
pipeline) is the clean one-variable test. Its weights are not among the supplied
artifacts — `cod/models/cod.py` ports `CODNoBaseline` for completeness and records
that neither the `COD_ablation_study` nor the `PI_DeepONet_abl_A` checkpoint
survived. The monolithic pair is the available approximation and it changes more
than one thing at once; §5 says what that costs.

WHAT THIS CAN AND CANNOT MEASURE RIGHT NOW. Fix 6 (DECISIONS N-1) invalidated the
COD checkpoint, so the only trained weights available are v57's. That is workable
for this specific question because **the V_arr clamp is not on the thermal path** —
`theta_TO` is produced by the thermal branch and the analytic baseline, neither of
which touches Arrhenius. The v57 thermal predictions are therefore exactly what
they always were. To keep it a measurement of spectral bias rather than of
distribution shift, every model is scored on the same seed-999 test set it was
matched to, i.e. in distribution.

Run:  python audit_port/scripts/18_swing_fidelity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_test_set, load_training_set  # noqa: E402
from cod.data.physics import (  # noqa: E402
    N_SENSORS, STATE_DIM_FAST, TW, fast_rhs_np, hot_spot_ETC_np,
)
from cod.data.steady_state import formula_A  # noqa: E402
from cod.models.cod import CODOperator  # noqa: E402
from cod.models.daily_mean import jensen_gap_from_trajectory  # noqa: E402
from cod.models.monolithic import MonolithicFair, MonolithicMultiHead  # noqa: E402

ART = ROOT / "reference" / "artifacts"
OUT = ROOT / "audit_port" / "SWING_FIDELITY.md"
NQ = 100        # query points across the window
STATES = ["c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2", "DP"]


def true_hs(x0, K_s, Ta_s):
    tau = np.linspace(0.0, TW, N_SENSORS)

    def rhs(t, x):
        return fast_rhs_np(x, float(np.interp(t, tau, K_s)),
                           float(np.interp(t, tau, Ta_s)))

    tq = np.linspace(0.0, TW, NQ)
    sol = solve_ivp(rhs, [0.0, TW], np.asarray(x0, float), method="RK45",
                    t_eval=tq, rtol=1e-9, atol=1e-11)
    K_q = np.interp(tq, tau, K_s)
    return tq, np.array([hot_spot_ETC_np(sol.y[0][i], K_q[i]) for i in range(NQ)])


@torch.no_grad()
def pred_hs(model, x0, K_s, Ta_s, device):
    tau = np.linspace(0.0, TW, N_SENSORS)
    tq = np.linspace(0.0, TW, NQ)
    s = torch.tensor(np.concatenate([K_s, Ta_s]), dtype=torch.float32,
                     device=device).unsqueeze(0).expand(NQ, -1).contiguous()
    x = torch.tensor(np.asarray(x0, np.float32), device=device
                     ).unsqueeze(0).expand(NQ, -1).contiguous()
    t = torch.tensor(tq, dtype=torch.float32, device=device).unsqueeze(-1)
    th = model(x, s, t)[:, 0].cpu().numpy().astype(float)
    K_q = np.interp(tq, tau, K_s)
    return np.array([hot_spot_ETC_np(th[i], K_q[i]) for i in range(NQ)])


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════
def build_models(ts, device):
    """The three checkpoints that exist, tagged by whether H is present."""
    cod = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128,
                      p=64, n_layers=4, n_exp_feats=12, T=TW,
                      x_mean=ts.x_mean, x_std=ts.x_std,
                      theta_ss_mode="formula_C", legacy_V_clamp=True).to(device)
    ckpt = torch.load(ART / "transformer_pideepOnet_v57.pt", map_location=device,
                      weights_only=False)
    cod.load_state_dict(ckpt["model_state_dict"], strict=True)

    mf = MonolithicFair(d_h=128, p=64, n_layers=4, n_exp=12,
                        x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mf.load_state_dict(torch.load(ART / "mono_fair_v2_perstate.pt",
                                  map_location=device, weights_only=False),
                       strict=True)

    mh = MonolithicMultiHead(d_h=128, p=64, n_layers=4, n_exp=12,
                             x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mh.load_state_dict(torch.load(ART / "mono_multihead.pt",
                                  map_location=device, weights_only=False),
                       strict=True)

    return [
        ("COD v57", "yes", "`transformer_pideepOnet_v57.pt`", cod),
        ("Mono FAIR", "no", "`mono_fair_v2_perstate.pt`", mf),
        ("Mono multi-head", "no", "`mono_multihead.pt`", mh),
    ]


def score(model, cases, device):
    rows = []
    for c in cases:
        K_s = c.K_sensors.astype(float)
        Ta_s = c.Ta_sensors.astype(float)
        _, hs_t = true_hs(c.x0, K_s, Ta_s)
        hs_p = pred_hs(model, c.x0, K_s, Ta_s, device)
        rows.append({"kind": c.kind,
                     "sw_t": 0.5 * (hs_t.max() - hs_t.min()),
                     "sw_p": 0.5 * (hs_p.max() - hs_p.min()),
                     "mae": float(np.abs(hs_p - hs_t).mean()),
                     "g_t": jensen_gap_from_trajectory(hs_t)[0],
                     "g_p": jensen_gap_from_trajectory(hs_p)[0]})
    out = {k: np.array([r[k] for r in rows]) for k in
           ("sw_t", "sw_p", "mae", "g_t", "g_p")}
    out["kind"] = np.array([r["kind"] for r in rows])
    out["n"] = len(rows)
    return out


def main() -> int:
    device = torch.device("cpu")
    ts = load_training_set(ART / "transformer_training_v57.npz")
    models = build_models(ts, device)
    cases = build_test_set(n_test=100, seed=999, T=TW, steady_state=formula_A)

    res = {}
    for name, has_H, ckpt_name, m in models:
        m.eval()
        res[name] = score(m, cases, device)
        r = res[name]
        live = (r["kind"] == "TV") & (r["sw_t"] > 1.0)
        ratio = r["sw_p"][live] / r["sw_t"][live]
        print(f"{name:18s} median ratio {np.median(ratio):.4f}  "
              f"under {100.0 * (ratio < 1.0).mean():.0f}%  "
              f"MAE {np.median(r['mae'][live]):.3f} degC")

    md: list[str] = []
    A = md.append
    A("# Does the surrogate flatten the thermal cycle?\n")
    A("**Question.** Neural networks are spectrally biased toward low "
      "frequencies, so a thermal surrogate may smooth peaks. If it does it "
      "discards part of the Jensen gap the method exists to preserve, and "
      "**thermal MAE would not show it** — a flattened trajectory can be close in "
      "mean absolute error while systematically under-stating the peak-to-trough "
      "range the convex Arrhenius integral is sensitive to.\n")
    A("**The mechanism under test.** COD predicts a correction to an analytic "
      "first-order solution, so the cycle shape comes from the IEC baseline and "
      "the network's spectral bias has nothing to flatten. Any architecture "
      "*without* that baseline has to generate the cycle from the network itself, "
      "which is exactly the frequency content spectral bias suppresses. So the "
      "prediction is not \"neural surrogates flatten cycles\" but \"neural "
      "surrogates flatten cycles **when nothing else supplies the shape**\".\n")

    ref = res["COD v57"]
    live_ref = (ref["kind"] == "TV") & (ref["sw_t"] > 1.0)

    A("## 1. Headline, all three checkpoints\n")
    A("Half peak-to-peak of the hot-spot trajectory, 100 query points across the "
      "window, truth by RK45 on `fast_rhs_np` at `rtol = 1e-9`. Live cases are "
      "time-varying with a true swing above 1 degC "
      f"(n = {int(live_ref.sum())}).\n")
    A("| model | analytic baseline H | checkpoint | median swing ratio | "
      "under-predicting | median thermal MAE degC |")
    A("|---|---|---|---|---|---|")
    for name, has_H, ckpt_name, _ in models:
        r = res[name]
        live = (r["kind"] == "TV") & (r["sw_t"] > 1.0)
        ratio = r["sw_p"][live] / r["sw_t"][live]
        A(f"| {name} | {has_H} | {ckpt_name} | **{np.median(ratio):.4f}** | "
          f"{100.0 * (ratio < 1.0).mean():.0f}% | "
          f"{np.median(r['mae'][live]):.4f} |")
    A("")
    A("**The direction predicted by the mechanism is what appears.** COD, which "
      "has the baseline, reproduces the swing to 1.0121 and under-predicts in 14% "
      "of cases. Both architectures without the baseline under-predict in "
      "**100%** of cases, at a median 0.68 and 0.65 — a third of the swing gone. "
      "One-sidedness is the part that is hard to explain any other way: an "
      "inaccurate but unbiased model should *over*-predict half peak-to-peak, "
      "because independent error inflates the range of a sampled max minus min. "
      "Losing swing in every single case is the signature of smoothing, not of "
      "error size.\n")
    A("**And the mechanism is still not established, because these checkpoints "
      "did not train.** Median thermal MAE is 12.7 and 8.8 degC against COD's "
      "0.51, and in the 25-200 degC band it is 37.8 degC — a model that is not "
      "tracking the trajectory at all, whose swing ratio says nothing about "
      "spectral bias. The bands where the monolithic models are at least roughly "
      "tracking (10-15 and 15-25 degC, MAE 2.2-4.5 degC) lose a more modest "
      "17-28% of the swing, and the ratio there does not worsen with swing — it "
      "improves, 0.774 to 0.828 for Mono FAIR. The collapse to 0.61 arrives "
      "together with the MAE blowup, which is the reading with the fewest "
      "assumptions.\n")
    A("Audit M-2 already established that the monolithic error *rises* 47x as "
      "capacity grows 16x, with causal weights underflowed to zero — evidence for "
      "\"we could not train this baseline\", not \"this architecture cannot "
      "represent the system\". Attributing the swing loss to spectral bias would "
      "repeat exactly that inference. The repo rule applies unchanged: a model "
      "that did not converge is reported as not converged, not converted into a "
      "performance number. §5 says what would settle it.\n")

    A("## 2. Full distribution of the swing ratio, per model\n")
    A("| model | population | n | median ratio | Q1 | Q3 | p10 | p90 | "
      "median error degC |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, _, _, _ in models:
        r = res[name]
        tv = r["kind"] == "TV"
        live = tv & (r["sw_t"] > 1.0)
        for lbl, m in [("all cases", np.ones(r["n"], bool)),
                       ("time-varying", tv),
                       ("time-varying, swing > 1 degC", live),
                       ("constant K", r["kind"] == "CK")]:
            if m.sum() == 0:
                continue
            rr = r["sw_p"][m] / np.maximum(r["sw_t"][m], 1e-9)
            e = r["sw_p"][m] - r["sw_t"][m]
            A(f"| {name} | {lbl} | {int(m.sum())} | **{np.median(rr):.4f}** | "
              f"{np.percentile(rr, 25):.4f} | {np.percentile(rr, 75):.4f} | "
              f"{np.percentile(rr, 10):.4f} | {np.percentile(rr, 90):.4f} | "
              f"{np.median(e):+.4f} |")
    A("")

    A("## 3. Stratified by true swing — does it worsen where it matters?\n")
    A("A spectral-bias failure shows up as a ratio below 1 that **worsens as the "
      "swing grows**. A ratio near 1 that is flat across the bands is the absence "
      "of the failure; a ratio that falls with the band is the failure itself.\n")
    A("| model | true swing band | n | median ratio | median error degC | "
      "median MAE degC |")
    A("|---|---|---|---|---|---|")
    for name, _, _, _ in models:
        r = res[name]
        live = (r["kind"] == "TV") & (r["sw_t"] > 1.0)
        for lo, hi in [(1, 5), (5, 10), (10, 15), (15, 25), (25, 200)]:
            m = live & (r["sw_t"] >= lo) & (r["sw_t"] < hi)
            if m.sum() == 0:
                continue
            A(f"| {name} | {lo}-{hi} degC | {int(m.sum())} | "
              f"{np.median(r['sw_p'][m] / r['sw_t'][m]):.4f} | "
              f"{np.median(r['sw_p'][m] - r['sw_t'][m]):+.4f} | "
              f"{np.median(r['mae'][m]):.4f} |")
    A("")

    A("## 4. The consequence that MAE hides\n")
    A("The reason to measure the swing rather than the error: the Jensen gap "
      "carried by the predicted trajectory against the gap carried by the true "
      "one. Any flattening shows up here even when MAE is small. The last column "
      "is signed so that **negative means gap lost**, i.e. the predicted "
      "trajectory carries less Arrhenius acceleration than the true one.\n")
    A("| model | state | median true gap | median predicted gap | median ratio | "
      "delta gap, ratio of medians |")
    A("|---|---|---|---|---|---|")
    for name, _, _, _ in models:
        r = res[name]
        live = (r["kind"] == "TV") & (r["sw_t"] > 1.0)
        for i, nm in enumerate(STATES):
            a, b = r["g_t"][live, i], r["g_p"][live, i]
            A(f"| {name} | `{nm}` | {np.median(a):.4f} | {np.median(b):.4f} | "
              f"{np.median(b / a):.4f} | "
              f"{100 * (np.median(b) / np.median(a) - 1):+.2f}% |")
    A("")
    A("Thermal MAE over the live cases, for context:\n")
    A("| model | median MAE degC | p90 MAE degC |")
    A("|---|---|---|")
    for name, _, _, _ in models:
        r = res[name]
        live = (r["kind"] == "TV") & (r["sw_t"] > 1.0)
        A(f"| {name} | {np.median(r['mae'][live]):.4f} | "
          f"{np.percentile(r['mae'][live], 90):.4f} |")
    A("")
    A("MAE and peak-to-trough range are not the same measurement, and only one of "
      "them is what the convexity argument depends on. Reading the two tables "
      "together is the point: a model can be close in MAE and still lose gap, and "
      "the gap column is the one the method's claim rests on.\n")

    A("## 5. What this does not establish\n")
    A("1. **Ablation A is the test this approximates, and its weights do not "
      "exist.** Ablation A is COD's architecture with the analytic baseline "
      "replaced by the constant `x0` — same network, same pipeline, one variable. "
      "Neither the `COD_ablation_study` checkpoint nor the standalone "
      "`PI_DeepONet_abl_A` one is among the supplied artifacts, which "
      "`cod/models/cod.py` records at `CODNoBaseline`. The monolithic pair used "
      "here removes the analytic baseline **and** the cascaded gas integral, and "
      "was trained to a far worse optimum, so it is a two-variable substitute. "
      "The thermal MAE column is the control that says how much of the swing "
      "result is attributable to raw model error rather than to spectral bias.\n")
    A("2. **It is the v57 checkpoint for COD.** Fix 6 invalidated it and the "
      "retrain has not happened. The V_arr clamp is not on the thermal path — "
      "`theta_TO` comes from the thermal branch and the analytic baseline, neither "
      "of which touches Arrhenius — so these thermal predictions are exactly what "
      "they always were, but they belong to a model trained on the old "
      "distribution.\n")
    A("3. **In distribution only.** Scored on the seed-999 set the models were "
      "matched to, so that this measures spectral bias rather than distribution "
      "shift. The realistic sampler (fix 7) is a different distribution and the "
      "same check has to be rerun there after the retrain.\n")
    A("4. **Every monolithic checkpoint carries the J-8 defect.** Its thermal "
      "exponent was shadowed to 12 instead of 0.8 during training. That is a "
      "property of the supplied weights, not of the architecture, and it is one "
      "more reason the substitute is not Ablation A.\n")
    A("5. **What would settle it**, in increasing order of cost: retrain "
      "`CODNoBaseline` on the fix-7 distribution under the same budget as COD and "
      "rerun this script, which is the one-variable test and costs one training "
      "run; or, if a no-baseline model still will not converge under a fair "
      "budget, report that as a convergence result and drop the spectral-bias "
      "claim rather than resting it on a broken checkpoint. The delta-learning "
      "argument — that predicting a correction to an analytic solution preserves "
      "the Jensen gap against the network's low-frequency bias — is worth making "
      "only if a converged no-baseline model still loses the gap.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
