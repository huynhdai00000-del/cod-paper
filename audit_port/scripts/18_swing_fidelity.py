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

**This is also what decides whether a given thermal MAE is acceptable.** MAE alone
cannot say. Arrhenius errors are signed and `V_arr` is convex, so a positive
temperature error costs more gap than an equal negative one returns, and an error
that is unbiased in degrees is not unbiased in aging rate. §3 and §4 are the two
tables DECISIONS C-11 requires every model in the matrix to report alongside MAE.

THE MECHANISM UNDER TEST. COD predicts a *correction* to an analytic first-order
solution, so the cycle shape is supplied by the IEC baseline and the network's
spectral bias has nothing to flatten. That predicts the opposite for any
architecture without such a baseline: the network has to generate the cycle
itself, which is exactly the frequency content spectral bias suppresses. See N-8,
N-9 and O-12 — the clean one-variable test needs a converged `CODNoBaseline`,
which does not exist yet.

WHICH DISTRIBUTION. The default evaluation set is `build_realistic_test_set`: the
frozen fix-7 sampler at a held-out seed, i.e. genuinely in-distribution for a
model trained off `configs/example_cod_seed1.yaml`. The previous version of this
script scored on `build_test_set` (the v57 benchmark), which was correct then
because the only checkpoints were v57's, and is wrong for a fix-7 model —
`audit_port/TEST_SET_PROVENANCE.md` measures seven of nine distributional axes
outside training support. `--tier v57` still selects the legacy benchmark, now
labelled out-of-family rather than T1.

Run:
    python audit_port/scripts/18_swing_fidelity.py --checkpoint PATH/model.pt
    python audit_port/scripts/18_swing_fidelity.py --v57-checkpoints   # legacy
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import (  # noqa: E402
    build_realistic_test_set, build_test_set, load_training_set,
)
from cod.data.physics import (  # noqa: E402
    N_SENSORS, STATE_DIM_FAST, TW, fast_rhs_np, hot_spot_ETC_np,
)
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.data.steady_state import formula_A  # noqa: E402
from cod.models.cod import CODNoBaseline, CODOperator  # noqa: E402
from cod.models.daily_mean import jensen_gap_from_trajectory  # noqa: E402
from cod.models.monolithic import MonolithicFair, MonolithicMultiHead  # noqa: E402

ART = ROOT / "reference" / "artifacts"
CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
OUT = ROOT / "audit_port" / "SWING_FIDELITY.md"
NQ = 100        # query points across the window
STATES = ["c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2", "DP"]
BANDS = [(1, 5), (5, 10), (10, 15), (15, 25), (25, 200)]

# Gates. A swing ratio below this in any band the model actually tracks means the
# cycle is being flattened where the Jensen argument needs it kept.
RATIO_FLOOR = 0.95
# Bands whose thermal MAE exceeds this are not evidence about spectral bias at
# all: the model is not following the trajectory there (N-9).
TRACKING_MAE = 5.0


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
def load_cod_checkpoint(path: Path, device):
    """A COD checkpoint written by `scripts/run.py`, with its own provenance.

    `x_mean_TO` / `x_std_TO` are registered buffers, so `strict=True` restores the
    normalisation from the file and the constructor values are placeholders. The
    load is strict on purpose: a checkpoint that needs `strict=False` to fit the
    current architecture is not a checkpoint of the current architecture.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ck:
        raise ValueError(
            f"{path} has no `model_state_dict`. Checkpoints written before "
            "run.py persisted weights do not exist — the model was discarded on "
            "exit — so there is nothing to score and the run has to be repeated.")
    kind = ck.get("model_kind", "cod")
    if kind not in ("cod", "cod_no_baseline"):
        raise ValueError(f"{path} holds a {kind!r} model; this loader handles "
                         "the COD family. The other C-11 architectures are "
                         "loaded through run.py's builder.")
    # Ablation A differs from COD only in `_ode_baseline`, so it loads through
    # the same constructor with the same state dict.
    cls = CODNoBaseline if kind == "cod_no_baseline" else CODOperator
    model = cls(
        state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
        n_layers=4, n_exp_feats=12, T=TW, x_mean=np.zeros(6), x_std=np.ones(6),
        theta_ss_mode=ck.get("theta_ss_mode", "true_fixed_point"),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    meta = {
        "model_kind": ck.get("model_kind", "cod"),
        "converged": ck.get("converged"),
        "stop_reason": ck.get("stop_reason"),
        "distribution_hash": ck.get("distribution_hash"),
        "config_hash": ck.get("config_hash"),
        "seed": ck.get("seed"),
    }
    return model, meta


def load_exact(smooth: float | None = None):
    """`ExactModel` from `16_bias_diagnosis.py`, optionally flattened.

    Loaded by path because `16_bias_diagnosis` is not an importable name; reused
    rather than restated so RK45-wearing-a-model-signature has one definition.

    `smooth` pulls the trajectory a fixed fraction toward its own mean, which is
    precisely the failure mode this script tests for — a trajectory that stays
    close in MAE while losing peak-to-trough range. It exists so the gate can be
    shown to fire, not just to pass.
    """
    spec = importlib.util.spec_from_file_location(
        "bias_diag", Path(__file__).with_name("16_bias_diagnosis.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exact = mod.ExactModel()
    if smooth is None:
        return exact

    class Smoothed(torch.nn.Module):
        def __init__(self, inner, frac):
            super().__init__()
            self.inner, self.frac = inner, float(frac)

        def forward(self, x0, u_sensors, t):
            y = self.inner(x0, u_sensors, t).clone()
            th = y[:, 0]
            y[:, 0] = th.mean() + (1.0 - self.frac) * (th - th.mean())
            return y

    return Smoothed(exact, smooth)


def build_v57_models(device):
    """The three v57-era checkpoints, tagged by whether the baseline H is present."""
    ts = load_training_set(ART / "transformer_training_v57.npz")
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

    return [("COD v57", "yes", "`transformer_pideepOnet_v57.pt`", cod, {}),
            ("Mono FAIR", "no", "`mono_fair_v2_perstate.pt`", mf, {}),
            ("Mono multi-head", "no", "`mono_multihead.pt`", mh, {})]


def score(model, cases, device):
    rows = []
    for c in cases:
        K_s = c.K_sensors.astype(float)
        Ta_s = c.Ta_sensors.astype(float)
        _, hs_t = true_hs(c.x0, K_s, Ta_s)
        hs_p = pred_hs(model, c.x0, K_s, Ta_s, device)
        rows.append({"kind": c.kind, "family": c.family or "-",
                     "sw_t": 0.5 * (hs_t.max() - hs_t.min()),
                     "sw_p": 0.5 * (hs_p.max() - hs_p.min()),
                     "mae": float(np.abs(hs_p - hs_t).mean()),
                     "bias": float((hs_p - hs_t).mean()),
                     "g_t": jensen_gap_from_trajectory(hs_t)[0],
                     "g_p": jensen_gap_from_trajectory(hs_p)[0]})
    out = {k: np.array([r[k] for r in rows]) for k in
           ("sw_t", "sw_p", "mae", "bias", "g_t", "g_p")}
    out["kind"] = np.array([r["kind"] for r in rows])
    out["family"] = np.array([r["family"] for r in rows])
    out["n"] = len(rows)
    return out


def live_mask(r):
    """Cases where a swing ratio is a meaningful quantity at all.

    Time-varying, and with a true swing above 1 degC. Below that the ratio is a
    small number over a small number and says nothing; the constant-load cases
    have no cycle to preserve and no Jensen gap by definition (N-4).
    """
    return (r["kind"] == "TV") & (r["sw_t"] > 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, nargs="+",
                    help="One or more model.pt from scripts/run.py. Several are "
                         "scored into the same tables, which is what the O-12 "
                         "comparison needs: Ablation A is only interpretable "
                         "beside COD on the same evaluation set, and building "
                         "two reports would invite comparing rows that were "
                         "scored on different draws.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Display names, one per checkpoint.")
    ap.add_argument("--v57-checkpoints", action="store_true",
                    help="Also score the three v57-era checkpoints.")
    ap.add_argument("--exact", action="store_true",
                    help="Score ExactModel (zero error by construction) as a "
                         "self-test: the swing ratio must come out at 1.0000 "
                         "and the gap ratio at 1.0000.")
    ap.add_argument("--smooth-test", type=float, default=None,
                    help="With --exact, low-pass the exact trajectory by this "
                         "fraction toward its own mean. Proves the gate fires "
                         "on a flattened cycle, which is the failure this "
                         "script exists to detect.")
    ap.add_argument("--tier", default="realistic",
                    choices=["realistic", "v57"],
                    help="realistic = frozen sampler, held-out seed (T1). "
                         "v57 = the legacy seed-999 benchmark (out of family).")
    ap.add_argument("--n-cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="Where to write the report. Defaults to a FIXED path, "
                         "which is correct for a one-off but silently destroys "
                         "results when this is looped: 15 cells x 7 seeds would "
                         "overwrite one file 105 times and keep the last. Pass a "
                         "per-run path in a sweep.")
    args = ap.parse_args()
    out_path = args.out

    if not args.checkpoint and not args.v57_checkpoints and not args.exact:
        ap.error("give --checkpoint, --exact for the self-test, or "
                 "--v57-checkpoints to reproduce N-9.")
    device = torch.device(args.device)

    # ── Evaluation set ─────────────────────────────────────────────────────
    if args.tier == "realistic":
        params = RealisticParams.from_config(
            yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            ["distribution"]["sampler"]["params"])
        cases = build_realistic_test_set(n_test=args.n_cases, seed=args.seed,
                                         params=params, T=TW)
        tier_label = (f"T1 in-distribution — frozen fix-7 sampler "
                      f"(`fc4cb76c3b32ec17`), held-out seed {args.seed}")
    else:
        cases = build_test_set(n_test=args.n_cases, seed=args.seed, T=TW,
                               steady_state=formula_A)
        tier_label = ("out of family — the v57-era seed-999 benchmark "
                      "(`TEST_SET_PROVENANCE.md`)")
    print(f"[eval] {tier_label}, n = {len(cases)}")

    # ── Models ─────────────────────────────────────────────────────────────
    models = []
    if args.exact:
        lbl = ("ExactModel (zero error)" if args.smooth_test is None
               else f"ExactModel smoothed {args.smooth_test:g}")
        models.append((lbl, "n/a", "`16_bias_diagnosis.ExactModel`",
                       load_exact(args.smooth_test), {}))
    if args.checkpoint:
        for i, ckpt in enumerate(args.checkpoint):
            m, meta = load_cod_checkpoint(ckpt, device)
            if meta.get("converged") is False:
                print(f"[warn] {ckpt.name} did NOT converge "
                      f"(stop_reason={meta.get('stop_reason')!r}). Per the repo "
                      "rule its swing numbers describe a non-converged model.")
            kind = (meta.get("model_kind") or "cod")
            # Whether the analytic baseline H is present is the one variable
            # O-12 changes, so it is read from the checkpoint rather than typed
            # into a label that could disagree with the weights.
            has_H = "no" if kind == "cod_no_baseline" else "yes"
            if args.labels and i < len(args.labels):
                label = args.labels[i]
            else:
                label = f"{kind} (seed {meta.get('seed')})"
            models.append((label, has_H, f"`{ckpt.parent.name}/{ckpt.name}`",
                           m, meta))
    if args.v57_checkpoints:
        models.extend(build_v57_models(device))

    res = {}
    for name, _, _, m, _ in models:
        m.eval()
        res[name] = score(m, cases, device)
        r = res[name]
        live = live_mask(r)
        ratio = r["sw_p"][live] / r["sw_t"][live]
        print(f"{name:26s} median ratio {np.median(ratio):.4f}  "
              f"under {100.0 * (ratio < 1.0).mean():.0f}%  "
              f"MAE {np.median(r['mae'][live]):.3f} degC  (n_live={live.sum()})")

    # ── Gate ───────────────────────────────────────────────────────────────
    # Only the primary checkpoint is gated; the v57 models are reproduced for
    # comparison and are already known not to have converged (N-9).
    failures = []
    n_tracked = 0
    gated = args.checkpoint or args.exact
    if gated:
        r = res[list(res)[0]]
        live = live_mask(r)
        for lo, hi in BANDS:
            m = live & (r["sw_t"] >= lo) & (r["sw_t"] < hi)
            if m.sum() < 3:
                continue
            band_mae = float(np.median(r["mae"][m]))
            band_ratio = float(np.median(r["sw_p"][m] / r["sw_t"][m]))
            if band_mae > TRACKING_MAE:
                continue        # not tracking: says nothing about spectral bias
            n_tracked += 1
            if band_ratio < RATIO_FLOOR:
                failures.append(
                    f"{lo}-{hi} degC: median swing ratio {band_ratio:.4f} < "
                    f"{RATIO_FLOOR} at a tracked MAE of {band_mae:.2f} degC")

    # ── Report ─────────────────────────────────────────────────────────────
    md: list[str] = []
    A = md.append
    A("# Does the surrogate flatten the thermal cycle?\n")
    A(f"Evaluation set: **{tier_label}**, n = {len(cases)}. Truth by RK45 on "
      "`fast_rhs_np` at `rtol = 1e-9`, 100 query points across the window. "
      "Generated by `audit_port/scripts/18_swing_fidelity.py`.\n")
    A("**Question.** Neural networks are spectrally biased toward low "
      "frequencies, so a thermal surrogate may smooth peaks. If it does it "
      "discards part of the Jensen gap the method exists to preserve, and "
      "**thermal MAE would not show it** — a flattened trajectory can be close in "
      "mean absolute error while systematically under-stating the peak-to-trough "
      "range the convex Arrhenius integral is sensitive to. Because `V_arr` is "
      "convex, the error is also signed: a positive temperature excursion adds "
      "more gap than an equal negative one removes, so a model unbiased in "
      "degrees is not unbiased in aging rate. §4 is where that shows up.\n")
    A("**The mechanism under test.** COD predicts a correction to an analytic "
      "first-order solution, so the cycle shape comes from the IEC baseline and "
      "the network's spectral bias has nothing to flatten. Any architecture "
      "*without* that baseline has to generate the cycle from the network itself, "
      "which is exactly the frequency content spectral bias suppresses. So the "
      "prediction is not \"neural surrogates flatten cycles\" but \"neural "
      "surrogates flatten cycles **when nothing else supplies the shape**\".\n")

    A("## 1. Headline\n")
    A("Half peak-to-peak of the hot-spot trajectory. Live cases are time-varying "
      "with a true swing above 1 degC; constant-load windows have no cycle to "
      "preserve and no Jensen gap by definition (N-4).\n")
    A("| model | analytic baseline H | checkpoint | n live | median swing ratio | "
      "under-predicting | median thermal MAE degC | median thermal bias degC |")
    A("|---|---|---|---|---|---|---|---|")
    for name, has_H, ckpt_name, _, _ in models:
        r = res[name]
        live = live_mask(r)
        ratio = r["sw_p"][live] / r["sw_t"][live]
        A(f"| {name} | {has_H} | {ckpt_name} | {int(live.sum())} | "
          f"**{np.median(ratio):.4f}** | {100.0 * (ratio < 1.0).mean():.0f}% | "
          f"{np.median(r['mae'][live]):.4f} | "
          f"{np.median(r['bias'][live]):+.4f} |")
    A("")

    A("## 2. Full distribution of the swing ratio, per model\n")
    A("| model | population | n | median ratio | Q1 | Q3 | p10 | p90 | "
      "median error degC |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, _, _, _, _ in models:
        r = res[name]
        tv = r["kind"] == "TV"
        live = live_mask(r)
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
    A("**This is the table that decides whether a given thermal MAE is "
      "acceptable.** A spectral-bias failure shows up as a ratio below 1 that "
      "**worsens as the swing grows**. A ratio near 1 that is flat across the "
      "bands is the absence of the failure; a ratio that falls with the band is "
      "the failure itself. A band whose MAE is above "
      f"{TRACKING_MAE:.0f} degC is not evidence either way — the model is not "
      "tracking the trajectory there, and its swing ratio says nothing about "
      "spectral bias (N-9).\n")
    A("| model | true swing band | n | median ratio | median error degC | "
      "median MAE degC | tracking |")
    A("|---|---|---|---|---|---|---|")
    for name, _, _, _, _ in models:
        r = res[name]
        live = live_mask(r)
        for lo, hi in BANDS:
            m = live & (r["sw_t"] >= lo) & (r["sw_t"] < hi)
            if m.sum() == 0:
                continue
            bm = float(np.median(r["mae"][m]))
            A(f"| {name} | {lo}-{hi} degC | {int(m.sum())} | "
              f"{np.median(r['sw_p'][m] / r['sw_t'][m]):.4f} | "
              f"{np.median(r['sw_p'][m] - r['sw_t'][m]):+.4f} | "
              f"{bm:.4f} | {'yes' if bm <= TRACKING_MAE else '**no**'} |")
    A("")

    if args.tier == "realistic":
        A("### 3b. By sampler family\n")
        A("Which load families the result is actually about. `family` is the "
          "day-pattern the case was drawn from; `kind` is the *realised* "
          "in-window variation, so a family that varies over the day can still "
          "appear as a constant window (fix 7 made that a real part of the "
          "population).\n")
        A("| model | family | n | n live | median true swing degC | "
          "median ratio | median MAE degC |")
        A("|---|---|---|---|---|---|---|")
        for name, _, _, _, _ in models:
            r = res[name]
            live = live_mask(r)
            for fam in sorted(set(r["family"].tolist())):
                m = r["family"] == fam
                ml = m & live
                if m.sum() == 0:
                    continue
                cells = (f"{np.median(r['sw_t'][ml]):.2f} | "
                         f"{np.median(r['sw_p'][ml] / r['sw_t'][ml]):.4f} | "
                         f"{np.median(r['mae'][ml]):.4f}"
                         if ml.sum() else "- | - | -")
                A(f"| {name} | `{fam}` | {int(m.sum())} | {int(ml.sum())} | "
                  f"{cells} |")
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
    for name, _, _, _, _ in models:
        r = res[name]
        live = live_mask(r)
        for i, nm in enumerate(STATES):
            a, b = r["g_t"][live, i], r["g_p"][live, i]
            A(f"| {name} | `{nm}` | {np.median(a):.4f} | {np.median(b):.4f} | "
              f"{np.median(b / a):.4f} | "
              f"{100 * (np.median(b) / np.median(a) - 1):+.2f}% |")
    A("")
    A("Thermal MAE over the live cases, for context:\n")
    A("| model | median MAE degC | p90 MAE degC | median bias degC |")
    A("|---|---|---|---|")
    for name, _, _, _, _ in models:
        r = res[name]
        live = live_mask(r)
        A(f"| {name} | {np.median(r['mae'][live]):.4f} | "
          f"{np.percentile(r['mae'][live], 90):.4f} | "
          f"{np.median(r['bias'][live]):+.4f} |")
    A("")
    A("MAE and peak-to-trough range are not the same measurement, and only one of "
      "them is what the convexity argument depends on. Reading the two tables "
      "together is the point: a model can be close in MAE and still lose gap, and "
      "the gap column is the one the method's claim rests on.\n")

    A("## 5. Gate\n")
    A(f"Fails if any band the model tracks (median MAE <= {TRACKING_MAE:.0f} "
      f"degC) has a median swing ratio below {RATIO_FLOOR}.\n")
    if not gated:
        A("Not evaluated: no fix-7 checkpoint given, only the v57 models.\n")
    elif failures:
        A("**FAIL**\n")
        for f in failures:
            A(f"* {f}")
        A("")
    elif n_tracked == 0:
        A("**NOT EVALUATED** — the model tracks no swing band at a median MAE "
          f"of {TRACKING_MAE:.0f} degC or better, so there is no band in which "
          "the swing ratio carries information. A model that far off the "
          "trajectory is a convergence result, not a spectral-bias result "
          "(N-9).\n")
    else:
        A(f"**PASS** — the cycle is preserved in all {n_tracked} band(s) the "
          "model tracks.\n")

    A("## 6. What this does not establish\n")
    A("1. **Ablation A is still the test this approximates.** Ablation A is "
      "COD's architecture with the analytic baseline replaced by the constant "
      "`x0` — same network, same pipeline, one variable. Its weights are not "
      "among the supplied artifacts (`cod/models/cod.py`, `CODNoBaseline`). The "
      "monolithic pair removes the analytic baseline **and** the cascaded gas "
      "integral and was trained to a far worse optimum, so it is a two-variable "
      "substitute. DECISIONS O-12 is the one training run that would settle it.\n")
    A("2. **Every monolithic checkpoint carries the J-8 defect** — its thermal "
      "exponent was shadowed to 12 instead of 0.8 during training — and none of "
      "them converged (N-9). Their swing ratios are reported for continuity with "
      "the earlier version of this document, not as evidence about "
      "architecture.\n")
    A("3. **The v57 and fix-7 rows are not comparable.** They are different "
      "models scored on different distributions; only rows sharing an evaluation "
      "set can be read against each other.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    if not gated:
        print("\n(no gate: v57 checkpoints only)")
    elif n_tracked == 0:
        print(f"\nNOT EVALUATED: no swing band tracked at MAE <= "
              f"{TRACKING_MAE:.0f} degC. This model's swing ratio carries no "
              "information about spectral bias.")
    else:
        print(f"\nPASS ({n_tracked} tracked band(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
