#!/usr/bin/env python3
"""The model's real thermal error through a lifetime rollout — the number O-9
left open, and the one that decides whether an end-of-life figure is publishable.

WHAT O-9 ESTABLISHED, AND WHAT IT LEFT. The manuscript's unexplained -3 degC
rollout bias was an artifact of the metric: `theta_bias` subtracted the steady
state of the window's *mean* forcing from a value that is a *lagged* response to
that forcing's ripple, so a model with zero error scored -2.86 to -3.88 degC
(`audit_port/BIAS_DIAGNOSIS.md`). `rollout.py` now references the cyclic endpoint
of the window's own forcing instead, and `19_verify_bias_fix.py` shows an exact
integrator scoring -0.002 degC on it.

That removed a false number and put an honest hole in its place: **the model's
actual thermal error through a rollout has never been measured**, because the
field meant to measure it was measuring something else. This script measures it.

WHY THE ROLLOUT AND NOT THE 12 H TEST SET. A single-window MAE says nothing about
a 30-year horizon. The rollout feeds the model its own prediction back as the next
window's initial condition, so a small per-window bias compounds, and DP advances
through `V_arr` — exponential in temperature, ~10.8 %/K at 100 degC. A thermal
error that is negligible against the 2 degC engineering threshold can still move
end of life by years. That conversion is the whole question.

THE DESIGN (DECISIONS O-7). Three rollouts over the same forcing:

  reference    RK45 on `fast_rhs_np`, integrated continuously across windows with
               no reset (`cod.eval.rollout.reference_rollout`). This is the truth.
  free-running the model carrying its own state forward, exactly as deployed.
               `free - reference` is the total error at each point in the horizon.
  teacher-forced
               the model restarted from the reference state at every window.
               `teacher - reference` is single-window error alone.

`free - teacher` is therefore the accumulation: how much of the error is the model
compounding its own mistakes rather than making fresh ones. O-7 asks for systematic
bias to be separated from random error and this is the separation.

Everything except what produces the trajectory is shared between the three —
identical forcing via `window_forcing`, identical DP quadrature via `_advance_dp`,
identical initial condition. A reference that differed in any of those would
measure the difference rather than the model.

VERIFICATION WITHOUT A CHECKPOINT. `--exact` scores `ExactModel` (RK45 on
`fast_rhs_np` wearing `CODOperator`'s call signature, zero error by construction)
instead of a checkpoint. Every error column must come out at ~0 and the EOL gap at
~0 months. That is what makes this script trustworthy before any model exists: if
it reports a bias for a model that cannot have one, the instrument is broken, which
is precisely the failure O-9 spent its time on.

Run:
    python audit_port/scripts/24_rollout_thermal_error.py --exact
    python audit_port/scripts/24_rollout_thermal_error.py --checkpoint PATH/model.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.physics import DP0, DP_EOL, N_SENSORS, STATE_DIM_FAST, TW  # noqa: E402
from cod.data.steady_state import true_fixed_point_np  # noqa: E402
from cod.eval.rollout import (  # noqa: E402
    chi_lifetime_rollout, reference_rollout,
)
from cod.models.cod import CODOperator  # noqa: E402

OUT = ROOT / "audit_port" / "ROLLOUT_THERMAL_ERROR.md"
K_SCENARIOS = (0.85, 0.95, 1.00, 1.10)

# Gates for the `--exact` self-test. A zero-error model must score zero.
EXACT_BIAS_TOL = 0.05        # degC
EXACT_EOL_TOL = 0.5          # months

# Aging sensitivity, for converting degrees into lifetime. B_aging / T^2 at
# 100 degC; `16_bias_diagnosis.py` checks this against the constants in the code.
PCT_PER_K = 10.77


def load_exact():
    """`ExactModel` from the diagnosis script, loaded by path.

    `16_bias_diagnosis` is not an importable module name. Reusing it rather than
    restating RK45-wearing-a-model-signature keeps one definition of the thing.
    """
    spec = importlib.util.spec_from_file_location(
        "bias_diag", Path(__file__).with_name("16_bias_diagnosis.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExactModel()


class BiasedModel(torch.nn.Module):
    """`ExactModel` plus a constant offset on theta_TO, for testing the gate.

    A gate that has only ever passed is not a verified gate. Injecting a known
    bias must make §5 fail, and must make it fail by about the injected amount —
    which also checks the sign convention, since a metric that reports the right
    magnitude with the wrong sign would misattribute a conservative model as an
    optimistic one and vice versa.
    """

    def __init__(self, inner, offset: float):
        super().__init__()
        self.inner = inner
        self.offset = float(offset)

    def forward(self, x0, u_sensors, t):
        y = self.inner(x0, u_sensors, t)
        y = y.clone()
        y[:, 0] = y[:, 0] + self.offset
        return y


def load_checkpoint(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ck:
        raise SystemExit(
            f"{path} carries no `model_state_dict`. Runs from before run.py "
            "persisted weights discarded the model on exit; there is nothing to "
            "roll and the run has to be repeated.")
    model = CODOperator(
        state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
        n_layers=4, n_exp_feats=12, T=TW, x_mean=np.zeros(6), x_std=np.ones(6),
        theta_ss_mode=ck.get("theta_ss_mode", "true_fixed_point"),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, ck


def eol_months(dp: np.ndarray, years: np.ndarray, reached: bool) -> float | None:
    """End of life in months, linearly interpolated in 1/DP.

    Depolymerisation is linear in `1/DP`, not in `DP`, so interpolating the
    reciprocal is the interpolation the physics implies rather than a convenience.
    Returns None if the horizon ended before EOL — a censored rollout is reported
    as censored, not as its last window.
    """
    if not reached:
        return None
    inv = 1.0 / dp
    target = 1.0 / DP_EOL
    j = int(np.argmax(inv >= target))
    if j == 0:
        return float(years[0] * 12.0)
    y0, y1 = years[j - 1], years[j]
    a, b = inv[j - 1], inv[j]
    f = 0.0 if b == a else (target - a) / (b - a)
    return float((y0 + f * (y1 - y0)) * 12.0)


def run_scenario(model, K_base, max_windows, device):
    """reference / free-running / teacher-forced, over identical forcing."""
    ref = reference_rollout(K_base, max_windows=max_windows,
                            steady_state=true_fixed_point_np)
    n = len(ref.years)

    # One cache across both model rollouts. The cyclic endpoint depends only on
    # the forcing, so recomputing it for the teacher pass is pure waste — and it
    # is the dominant cost, several RK45 cycles per distinct (K_w, Ta_w).
    cyc_cache: dict = {}
    free = chi_lifetime_rollout(model, K_base, dp_source="model",
                                steady_state=true_fixed_point_np,
                                max_windows=n, device=device,
                                cyc_cache=cyc_cache)

    # The reference state at the START of each window is the previous window's
    # end state. Window 0 starts from the shared initial condition, so the teacher
    # series is the reference end-states shifted by one, with the IC prepended.
    x_ic = np.concatenate([[float(true_fixed_point_np(K_base, 27.0))],
                           _gas_ic(K_base)])
    # Only theta_TO is restored from the reference. The gas states are what the
    # cascade produces downstream of it, so forcing those too would score the
    # thermal branch through a channel it does not drive.
    teacher_states = np.tile(x_ic, (n, 1))
    teacher_states[1:, 0] = ref.theta_TO_end[:-1]

    teach = chi_lifetime_rollout(model, K_base, dp_source="model",
                                 steady_state=true_fixed_point_np,
                                 max_windows=n, device=device,
                                 teacher_states=teacher_states,
                                 cyc_cache=cyc_cache)
    return ref, free, teach


def _gas_ic(K_base):
    from cod.data.steady_state import gas_ic_from_ss
    return gas_ic_from_ss(K_base, 27.0, steady_state=true_fixed_point_np)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--exact", action="store_true",
                    help="Score ExactModel (zero error by construction) as a "
                         "self-test of the instrument.")
    ap.add_argument("--inject-bias", type=float, default=None,
                    help="Add this many degC to every theta_TO prediction. Use "
                         "with --exact to prove the gate can fail and that the "
                         "reported bias has the right sign and size.")
    ap.add_argument("--max-windows", type=int, default=400,
                    help="Windows per scenario. 730 = one year.")
    ap.add_argument("--k-scenarios", type=float, nargs="+", default=None,
                    help="Load scenarios to run. Measured cost is ~3.2 s per "
                         "scenario-window, so a full year over all four is "
                         "~2.6 h — long enough to be killed as one job. Run "
                         "them separately and combine with --render-from.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Write this run's scenario rows here instead of "
                         "rendering a partial report.")
    ap.add_argument("--render-from", type=Path, nargs="+", default=None,
                    help="Render the report from previously written JSON rows. "
                         "Combining must not require re-running the rollouts.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.render_from:
        rows = []
        for p in args.render_from:
            rows.extend(json.loads(p.read_text(encoding="utf-8"))["rows"])
        rows.sort(key=lambda r: r["K"])
        for r in rows:            # arrays do not survive JSON; rebuild
            r["e_free"] = np.asarray(r["e_free"], float)
            r["e_teach"] = np.asarray(r["e_teach"], float)
        print(f"[render] {len(rows)} scenario(s) from "
              f"{len(args.render_from)} file(s), no rollout")
        return _render(rows, args, label=rows[0].get("label", "COD"),
                       failures=[])
    if not args.checkpoint and not args.exact:
        ap.error("give --checkpoint, or --exact for the self-test")

    device = torch.device(args.device)
    if args.exact:
        model, ck, label = load_exact(), {}, "ExactModel (zero error)"
        if args.inject_bias is not None:
            model = BiasedModel(model, args.inject_bias)
            label = f"ExactModel + {args.inject_bias:+g} degC injected"
    else:
        model, ck = load_checkpoint(args.checkpoint, device)
        label = f"COD `{args.checkpoint.name}`"
        if ck.get("converged") is False:
            print(f"[warn] this checkpoint did NOT converge "
                  f"(stop_reason={ck.get('stop_reason')!r}); its rollout error "
                  "describes a non-converged model.")

    rows = []
    scenarios = args.k_scenarios if args.k_scenarios else list(K_SCENARIOS)
    print(f"[roll] {label}, {args.max_windows} windows per scenario "
          f"({args.max_windows * TW / 1440 / 365:.2f} yr), "
          f"K = {scenarios}")
    for K in scenarios:
        ref, free, teach = run_scenario(model, K, args.max_windows, device)
        e_free = free.theta_TO_end - ref.theta_TO_end
        e_teach = teach.theta_TO_end - ref.theta_TO_end
        rows.append({
            "K": K, "n": len(ref.years),
            "ref_dp_end": float(ref.dp[-1]), "free_dp_end": float(free.dp[-1]),
            "bias_free": float(e_free.mean()), "sd_free": float(e_free.std()),
            "max_free": float(np.abs(e_free).max()),
            "bias_teach": float(e_teach.mean()), "sd_teach": float(e_teach.std()),
            "drift": float(e_free.mean() - e_teach.mean()),
            "e_free": e_free, "e_teach": e_teach,
            "eol_ref": eol_months(ref.dp, ref.years, ref.reached_eol),
            "eol_free": eol_months(free.dp, free.years, free.reached_eol),
            "dp_rel": float(free.dp[-1] / ref.dp[-1] - 1.0),
            "years": ref.years, "label": label,
        })
        r = rows[-1]
        print(f"  K={K:.2f}  bias {r['bias_free']:+7.4f} degC  sd "
              f"{r['sd_free']:6.4f}  single-window bias {r['bias_teach']:+7.4f}  "
              f"accumulation {r['drift']:+7.4f}  DP end "
              f"{r['free_dp_end']:.1f} vs {r['ref_dp_end']:.1f}")

    # ── Gate ───────────────────────────────────────────────────────────────
    failures = []
    if args.exact:
        for r in rows:
            if abs(r["bias_free"]) > EXACT_BIAS_TOL:
                failures.append(
                    f"K={r['K']:.2f}: ExactModel scored a free-running bias of "
                    f"{r['bias_free']:+.4f} degC, above {EXACT_BIAS_TOL} — a "
                    "model with zero error by construction must score zero, so "
                    "the instrument is measuring something other than model "
                    "error (this is the O-9 failure mode).")
            if abs(r["bias_teach"]) > EXACT_BIAS_TOL:
                failures.append(
                    f"K={r['K']:.2f}: ExactModel single-window bias "
                    f"{r['bias_teach']:+.4f} degC, above {EXACT_BIAS_TOL}.")
            if (r["eol_ref"] is not None and r["eol_free"] is not None
                    and abs(r["eol_free"] - r["eol_ref"]) > EXACT_EOL_TOL):
                failures.append(
                    f"K={r['K']:.2f}: ExactModel moved EOL by "
                    f"{r['eol_free'] - r['eol_ref']:+.2f} months.")

    if args.json_out:
        # Rows only, so scenarios run as separate jobs can be combined later
        # without re-running any rollout. Measured cost is ~3.2 s per
        # scenario-window; a year over four scenarios does not fit in one job.
        args.json_out.write_text(json.dumps({"rows": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items()} for r in rows]}), encoding="utf-8")
        print(f"[json] wrote {args.json_out} ({len(rows)} scenario(s))")
        return 1 if failures else 0

    return _render(rows, args, label, failures)


def _render(rows, args, label, failures) -> int:
    # Taken from the rows, not from `args`: in --render-from mode the horizon
    # belongs to the runs being combined, and `n_win` would be
    # whatever default this invocation happened to carry.
    n_win = max(r["n"] for r in rows) if rows else 0
    # ── Report ─────────────────────────────────────────────────────────────
    md: list[str] = []
    A = md.append
    A("# Thermal error through a lifetime rollout\n")
    A(f"Model: **{label}**. {n_win} windows per scenario "
      f"({n_win * TW / 1440 / 365:.2f} yr at a {TW:g} min window). "
      "Generated by `audit_port/scripts/24_rollout_thermal_error.py`.\n")
    A("This is the measurement DECISIONS O-9 left open. O-9 showed the "
      "manuscript's -3 degC rollout bias was an artifact of the metric — "
      "`theta_bias` compared a lagged response against the equilibrium of the "
      "window-mean forcing, and an exact integrator scored -2.86 to -3.88 degC on "
      "it. Fixing the reference removed a false number and left an honest gap: "
      "the model's real thermal error through a rollout had never been measured, "
      "because the field meant to measure it was measuring something else.\n")

    A("## 1. Design\n")
    A("Three rollouts over **identical** forcing (`window_forcing`), identical DP "
      "quadrature (`_advance_dp`) and an identical initial condition. The only "
      "thing that differs is what produces the trajectory.\n")
    A("| rollout | trajectory from | what its difference from the reference means |")
    A("|---|---|---|")
    A("| reference | RK45 on `fast_rhs_np`, continuous across windows (O-7) | "
      "— it is the truth |")
    A("| free-running | the model, carrying its own prediction forward | total "
      "error as deployed |")
    A("| teacher-forced | the model, restarted from the reference each window | "
      "single-window error alone |")
    A("")
    A("`free - teacher` is the accumulation: the part of the error that comes "
      "from the model compounding its own output rather than from fresh mistakes. "
      "Separating those is what O-7 asks for.\n")

    A("## 2. Thermal error\n")
    A("| K_base | windows | free-running bias degC | sd | max abs | "
      "single-window bias degC | accumulation degC |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['K']:.2f} | {r['n']} | **{r['bias_free']:+.4f}** | "
          f"{r['sd_free']:.4f} | {r['max_free']:.4f} | {r['bias_teach']:+.4f} | "
          f"{r['drift']:+.4f} |")
    A("")
    A("Bias is the mean signed error and sd the scatter about it, which is O-7's "
      "systematic/random split. A signed bias matters far more than its magnitude "
      "suggests: `V_arr` is convex and exponential, so a persistent offset does "
      "not average out of the aging integral the way a zero-mean scatter of the "
      "same size does.\n")

    A("## 3. Error against horizon\n")
    A("Does the error grow as the rollout proceeds? A flat series is a model with "
      "a stable offset; a growing one is compounding, and only the second "
      "threatens a long-horizon claim.\n")
    A("| K_base | first 10 windows | first quarter | last quarter | "
      "last 10 windows |")
    A("|---|---|---|---|---|")
    for r in rows:
        e = r["e_free"]
        n = len(e)
        A(f"| {r['K']:.2f} | {e[:10].mean():+.4f} | {e[:n // 4].mean():+.4f} | "
          f"{e[-(n // 4):].mean():+.4f} | {e[-10:].mean():+.4f} |")
    A("")

    A("## 4. The lifetime consequence\n")
    A(f"Aging sensitivity is ~{PCT_PER_K:.2f} %/K at 100 degC, so a thermal bias "
      "converts to a rate error before it converts to a lifetime error. `DP end` "
      "is the state at the horizon; the EOL columns are populated only where a "
      "scenario actually reached end of life inside the horizon — a censored "
      "rollout is reported as censored, not as its last window.\n")
    A("| K_base | DP end, reference | DP end, model | relative | "
      "implied rate error | EOL reference (months) | EOL model (months) | "
      "EOL gap (months) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        eol_ref, eol_free = r["eol_ref"], r["eol_free"]
        cell_ref = "-" if eol_ref is None else f"{eol_ref:.1f}"
        cell_free = "-" if eol_free is None else f"{eol_free:.1f}"
        cell_gap = ("censored" if eol_ref is None or eol_free is None
                    else f"**{eol_free - eol_ref:+.2f}**")
        A(f"| {r['K']:.2f} | {r['ref_dp_end']:.2f} | {r['free_dp_end']:.2f} | "
          f"{100 * r['dp_rel']:+.3f}% | {PCT_PER_K * r['bias_free']:+.2f}% | "
          f"{cell_ref} | {cell_free} | {cell_gap} |")
    A("")
    A(f"Horizon is {n_win} windows "
      f"({n_win * TW / 1440 / 365:.2f} yr) and DP starts at {DP0:g} "
      f"against an EOL of {DP_EOL:g}, so reaching EOL requires a much longer run "
      "than the default. Pass `--max-windows` large enough for the EOL columns to "
      "populate before quoting an EOL figure.\n")

    A("## 5. Gate\n")
    if args.exact:
        A(f"`--exact` self-test: `ExactModel` has zero error by construction, so "
          f"every bias must be below {EXACT_BIAS_TOL} degC and any EOL gap below "
          f"{EXACT_EOL_TOL} months. This is what makes the instrument usable "
          "before a checkpoint exists — O-9 was a case of a metric reporting a "
          "confident number for a model that could not have produced it.\n")
        if failures:
            A("**FAIL**\n")
            for f in failures:
                A(f"* {f}")
            A("")
        else:
            A("**PASS** — the instrument reports zero for a zero-error model.\n")
    else:
        A("No gate: gating a real model's rollout bias against a fixed threshold "
          "would be inventing an acceptance criterion that no decision in "
          "DECISIONS has taken. Run `--exact` to check the instrument; read §2 "
          "and §4 to decide whether the model's number is publishable.\n")

    A("## 6. What this does not establish\n")
    A("1. **Only `dp_source=\"model\"`.** Under `dp_source=\"reference\"` DP "
      "advances from reference physics and the model is absent from the aging "
      "path by construction, so its EOL cannot reflect model quality (O-9).\n")
    A("2. **One seasonal forcing.** The rollout's `(K_w, Ta_w)` is a fixed "
      "function of day-of-year, not a sample from the training distribution. It "
      "is a scenario, not a population, and the error here is the error on that "
      "scenario.\n")
    A("3. **Teacher forcing restores theta_TO only.** The gas states are "
      "downstream of the thermal branch through the cascade; resetting them too "
      "would score the thermal prediction through a channel it does not drive.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS" if args.exact else "\n(no gate for a real checkpoint; read §2 and §4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
