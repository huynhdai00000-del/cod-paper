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

#: The five gases, in the order `fast_rhs_np` carries them, with the IEC 60599
#: attention level each is read against. ANALYSIS_PLAN Amendment 1 scores
#: end-of-rollout concentration in absolute ppm against these; no percentage of
#: any of them is reportable (C-9).
GAS_NAMES = ("c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2")
GAS_IEC_PPM = (100.0, 1.0, 50.0, 350.0, 2500.0)

# Gates for the `--exact` self-test. A zero-error model must score zero.
EXACT_BIAS_TOL = 0.05        # degC
EXACT_EOL_TOL = 0.5          # months
#: Gas gate for `--exact`, in ppm, per gas, in `GAS_NAMES` order.
#:
#: Set from measurement. `--exact --max-windows 730 --k-scenarios 0.95 1.10`
#: (one year, both Amendment 1 scenarios) scores `ExactModel`, which has zero
#: error by construction, at:
#:
#:     gas       worst |err| over both scenarios      tolerance      margin
#:     c_H2      1.29e-04 ppm                         5e-3            39x
#:     c_C2H2    2.33e-05 ppm                         1e-3            43x
#:     c_C2H4    2.75e-06 ppm                         1e-3           364x
#:     c_CO      1.21e-04 ppm                         5e-3            41x
#:     c_CO2     3.96e-04 ppm                         2e-2            51x
#:
#: The residual is float32 round-trip of the state carried between windows, not
#: physics: within a single window the two integrators agree to far more digits.
#: It grows **sub-linearly** with horizon — c_CO2 at K=0.95 went 1.2e-05 at 60
#: windows to 1.0e-04 at 730, a factor 8.6 for a factor 12 in windows — which is
#: what accumulated round-off looks like and not what a systematic drift looks
#: like, so the margins above hold at the two-year horizon too.
#:
#: Two bars they have to clear, and both are checked rather than asserted.
#: Against **IEC 60599 attention levels** they are 4e-4 to 1e-3 of the level, so
#: a residual this size cannot be mistaken for a finding. Against **ANALYSIS_PLAN
#: §3's reporting floors** — the smallest difference the analysis will report at
#: all — they are 0.5% to 2.5%, so an instrument residual cannot mask a
#: difference the plan would have reported. The second bar is the binding one:
#: an earlier draft set c_CO2 at 1.0 ppm, which is 100% of its floor, i.e. an
#: instrument allowed to be as large as the smallest thing it must resolve.
EXACT_GAS_TOL = (0.005, 0.001, 0.001, 0.005, 0.02)

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
    """`ExactModel` plus a constant offset, for testing the gate.

    A gate that has only ever passed is not a verified gate. Injecting a known
    bias must make §6 fail, and must make it fail by about the injected amount —
    which also checks the sign convention, since a metric that reports the right
    magnitude with the wrong sign would misattribute a conservative model as an
    optimistic one and vice versa.

    `gas_offset` is the same injection on the five gas channels, in ppm, and it
    exists for the same reason: Amendment 1's primary metric needs a gate that
    has been shown to fail, not one that has only ever been passed by a model
    that cannot fail it.

    A thermal-only injection does perturb the gases too — the offset theta is
    carried into the next window's initial condition and drives generation —
    but that is an indirect, integrated effect and it does not test the gas
    comparison's own arithmetic. The direct injection does.
    """

    def __init__(self, inner, offset: float = 0.0, gas_offset: float = 0.0):
        super().__init__()
        self.inner = inner
        self.offset = float(offset)
        self.gas_offset = float(gas_offset)

    def forward(self, x0, u_sensors, t):
        y = self.inner(x0, u_sensors, t)
        y = y.clone()
        y[:, 0] = y[:, 0] + self.offset
        if self.gas_offset:
            y[:, 1:] = y[:, 1:] + self.gas_offset
        return y


def load_checkpoint(path: Path, device):
    """Rebuild whatever architecture wrote this checkpoint.

    This used to construct a `CODOperator` unconditionally, which made the
    script usable on exactly one of the fifteen matrix cells — and the runbook
    asks for it on every cell. Since Amendment 1 puts the **primary** metric
    here, a loader that only speaks COD would leave H1 undecidable for every
    baseline no matter how many runs finished.

    `run.py` saves `cfg.raw` into the checkpoint, so the model is rebuilt by the
    same `build_model` that trained it rather than by a second constructor kept
    in step by hand. `strict=True` is what checks the rebuild was right.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ck:
        raise SystemExit(
            f"{path} carries no `model_state_dict`. Runs from before run.py "
            "persisted weights discarded the model on exit; there is nothing to "
            "roll and the run has to be repeated.")

    cfg = ck.get("config")
    if cfg is None:
        raise SystemExit(
            f"{path} carries no `config`. Without it the architecture cannot be "
            "rebuilt from the file alone, and guessing COD would silently "
            "produce a rollout of a different model than the one trained.")

    sys.path.insert(0, str(ROOT / "scripts"))
    from run import build_model                                  # noqa: E402
    x_mean = np.asarray(ck.get("x_mean", np.zeros(6)), float)
    x_std = np.asarray(ck.get("x_std", np.ones(6)), float)
    model, _predict = build_model(cfg, x_mean, x_std, device)
    model = model.to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    # Every architecture's `predict` is `model(x0, u, t)` and `cod_predict` adds
    # only an optional `theta_ss_grid` that defaults to None, so the rollout's
    # direct call is the same call for all of them. Asserted rather than assumed,
    # because a model whose forward needed an extra argument would raise inside
    # the rollout after the burn-ins had already been paid for.
    with torch.no_grad():
        probe = model(torch.zeros(1, STATE_DIM_FAST, device=device),
                      torch.zeros(1, 2 * N_SENSORS, device=device),
                      torch.zeros(1, 1, device=device))
    if probe.shape != (1, STATE_DIM_FAST):
        raise SystemExit(
            f"{path}: forward returned {tuple(probe.shape)}, expected "
            f"(1, {STATE_DIM_FAST}). The rollout carries all six states across "
            "windows and cannot use a model that does not return them.")
    return model, ck


#: `chi_lifetime_rollout` stops at `DP <= DP_EOL + 1.0`, so a rollout can halt
#: just short of DP_EOL itself -- 200.91 in the K=1.10 reference. EOL is therefore
#: read at this threshold, matching the stopping rule. One DP unit out of the 800
#: consumed is under 0.13% of life and far inside the model error being measured;
#: what is not acceptable is the two rules disagreeing, which is what reported an
#: end of life of 0.0 months.
DP_EOL_REPORT = DP_EOL + 1.0


def eol_months(dp: np.ndarray, years: np.ndarray, reached: bool) -> float | None:
    """End of life in months, linearly interpolated in 1/DP.

    Depolymerisation is linear in `1/DP`, not in `DP`, so interpolating the
    reciprocal is the interpolation the physics implies rather than a convenience.
    Returns None if the horizon ended before EOL — a censored rollout is reported
    as censored, not as its last window.

    The crossing is located explicitly rather than with `argmax`, which returns 0
    on an all-False array and is therefore indistinguishable from "crossed at the
    first window". That is not hypothetical: it reported EOL at month 0 for a
    reference that ended at DP 200.91 against a target of 200.
    """
    crossed = np.flatnonzero(dp <= DP_EOL_REPORT)
    if not reached or crossed.size == 0:
        return None
    j = int(crossed[0])
    if j == 0:
        return float(years[0] * 12.0)
    inv = 1.0 / dp
    target = 1.0 / DP_EOL_REPORT
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
    # The model runs to `max_windows`, not to the reference's length. Truncating
    # it at `n` would censor the model's own end of life whenever the reference
    # reaches EOL first — which a cold-biased model guarantees, since it reads DP
    # high and therefore reaches EOL later. The EOL gap would then be unmeasurable
    # by construction rather than by horizon, and would report as `censored` in
    # exactly the case it is most interesting.
    free = chi_lifetime_rollout(model, K_base, dp_source="model",
                                steady_state=true_fixed_point_np,
                                max_windows=max_windows, device=device,
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
    ap.add_argument("--inject-gas-bias", type=float, default=None,
                    help="Add this many ppm to every gas prediction. The same "
                         "proof for Amendment 1's primary metric: its gate must "
                         "be shown to fail, not merely to have passed.")
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
    ap.add_argument("--out", type=Path, default=OUT,
                    help="Where to write the report. Defaults to a FIXED path, "
                         "correct for a one-off but destructive in a loop: 15 "
                         "cells x 7 seeds would overwrite one file 105 times and "
                         "keep only the last. Pass a per-run path in a sweep. "
                         "--json-out already varies per run; this is the markdown "
                         "half that did not.")
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
        if args.inject_bias is not None or args.inject_gas_bias is not None:
            model = BiasedModel(model, args.inject_bias or 0.0,
                                args.inject_gas_bias or 0.0)
            inj = []
            if args.inject_bias:
                inj.append(f"{args.inject_bias:+g} degC")
            if args.inject_gas_bias:
                inj.append(f"{args.inject_gas_bias:+g} ppm on every gas")
            label = "ExactModel + " + " and ".join(inj) + " injected"
    else:
        model, ck = load_checkpoint(args.checkpoint, device)
        # The cell, not "COD". This script now loads any of the fifteen
        # architectures, and a report headed "COD" over a MIONet rollout is a
        # mislabelled artifact of exactly the kind J-92 exists to prevent.
        variant = (ck.get("config", {}).get("experiment", {}).get("variant")
                   or ck.get("model_kind") or "unknown")
        label = f"`{variant}` from `{args.checkpoint.parent.name}/model.pt`"
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
        # The model may outlive the reference now that it is not truncated to it,
        # so the error series are compared over the horizon both cover. The EOL
        # columns below deliberately use each rollout's own length instead.
        m = min(len(ref.theta_TO_end), len(free.theta_TO_end),
                len(teach.theta_TO_end))
        e_free = free.theta_TO_end[:m] - ref.theta_TO_end[:m]
        e_teach = teach.theta_TO_end[:m] - ref.theta_TO_end[:m]

        # ── Amendment 1's primary metric ────────────────────────────────────
        # Gas concentration in ppm at the end of the rollout, model against
        # reference. Read at the last window **both** cover: at K = 1.10 the
        # reference reaches EOL and stops before the horizon, and comparing a
        # model still running at window 1460 against a reference that stopped at
        # 1020 would score two different points in the unit's life against each
        # other. Same rule as the thermal series above, and it is the O-7 fix
        # applied to a second quantity rather than a new convention.
        #
        # Free-running only. Teacher forcing restores theta_TO alone by design
        # (§7.3), so a "teacher-forced" gas series is still free-running in the
        # gas channels and would not mean what its name says.
        #
        # The horizon is over the two rollouts this comparison actually
        # involves. `m` above also takes the teacher-forced rollout into
        # account, and the teacher pass is not part of the gas comparison at
        # all: letting it shorten this window would censor the metric by a
        # third party that does not appear in it (J-89's Form B — a window
        # derived from something other than the quantity being measured).
        if ref.gas_end is None or free.gas_end is None:
            raise SystemExit(
                "the rollout produced no gas state; `gas_end` is None, which "
                "means no window completed")
        j = min(len(ref.gas_end), len(free.gas_end)) - 1
        gas_ref = np.asarray(ref.gas_end[j], float)
        gas_free = np.asarray(free.gas_end[j], float)
        rows.append({
            "K": K, "n": len(ref.years),
            "gas_window": int(j), "gas_years": float(ref.years[j]),
            # Names travel with the values. A consumer that zipped its own gas
            # list against this one would be relying on two orderings in two
            # files staying in step, which is the shape of defect this project
            # keeps finding (J-89).
            "gas_names": list(GAS_NAMES),
            "gas_ref_end": gas_ref.tolist(),
            "gas_free_end": gas_free.tolist(),
            "gas_err": (gas_free - gas_ref).tolist(),
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
        print("          gas ppm at window "
              f"{r['gas_window']} ({r['gas_years']:.2f} yr): "
              + "  ".join(f"{g} {e:+.4g}" for g, e
                          in zip(GAS_NAMES, r["gas_err"])))

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
            # The same self-test on Amendment 1's primary metric. The
            # tolerances are per gas and are set from the measured `--exact`
            # residual, which is float32 round-trip of the carried state
            # compounded over the horizon, not physics — the two integrators
            # agree to far more digits than this within a single window. They
            # are two to three orders of magnitude below the IEC attention level
            # of each gas, so a residual this size cannot be mistaken for a
            # finding, and they are tight enough that the injection below trips
            # them by a wide margin.
            for i, (g, tol) in enumerate(zip(GAS_NAMES, EXACT_GAS_TOL)):
                e = r["gas_err"][i]
                if abs(e) > tol:
                    failures.append(
                        f"K={r['K']:.2f}: ExactModel scored an end-of-rollout "
                        f"{g} error of {e:+.4g} ppm, above {tol} ppm — a model "
                        "with zero error by construction must score zero, so "
                        "the primary metric is measuring something other than "
                        "model error.")

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

    A("## 5. End-of-rollout gas concentration — ANALYSIS_PLAN Amendment 1's "
      "primary metric\n")
    A("Gases obey `dc/dt = k_gen V_arr(theta) - k_dis c`, so they relax toward "
      "an equilibrium set by temperature. An in-cascade model reaches the "
      "correct one **by construction**, because it integrates the right "
      "quadrature; a monolithic model has no such constraint, and a wrong "
      "equilibrium persists indefinitely. That is a structural claim about "
      "long-horizon behaviour, and a 12 h window cannot see it because nothing "
      "has had time to accumulate.\n")
    A("Absolute ppm against IEC 60599 attention levels. **No percentage of any "
      "gas is reportable from this benchmark** (C-9): the median 12 h variation "
      "is three to five orders of magnitude below what a DGA lab resolves, so a "
      "gas ratio is a ratio to a physically empty quantity.\n")
    for r in rows:
        A(f"**K_base = {r['K']:.2f}**, read at window {r['gas_window']} "
          f"({r['gas_years']:.2f} yr) — the last window the model and the "
          "reference both cover.\n")
        A("| gas | reference ppm | model ppm | error ppm | IEC 60599 level | "
          "error as fraction of level |")
        A("|---|---|---|---|---|---|")
        for i, g in enumerate(GAS_NAMES):
            ref_c, mod_c = r["gas_ref_end"][i], r["gas_free_end"][i]
            err, lvl = r["gas_err"][i], GAS_IEC_PPM[i]
            A(f"| {g} | {ref_c:.4g} | {mod_c:.4g} | **{err:+.4g}** | "
              f"{lvl:g} | {abs(err) / lvl:.2e} |")
        A("")
    A("The last column is a fraction of an **engineering threshold**, not of "
      "the signal, which is what makes it a legitimate ratio where a gas NMAE "
      "is not: the denominator is a fixed quantity an instrument resolves, not "
      "a ground-truth variation that may be arbitrarily small.\n")

    A("## 6. Gate\n")
    if args.exact:
        A(f"`--exact` self-test: `ExactModel` has zero error by construction, so "
          f"every bias must be below {EXACT_BIAS_TOL} degC, any EOL gap below "
          f"{EXACT_EOL_TOL} months, and every end-of-rollout gas error below its "
          "per-gas tolerance ("
          + ", ".join(f"{g} {t:g} ppm"
                      for g, t in zip(GAS_NAMES, EXACT_GAS_TOL))
          + "). This is what makes the instrument usable before a checkpoint "
          "exists — O-9 was a case of a metric reporting a confident number for "
          "a model that could not have produced it.\n")
        A("`--inject-bias` and `--inject-gas-bias` are the other direction: a "
          "gate that has only ever passed is not a verified gate. Both are "
          "shown to fail on demand in PORT_LOG J-95.\n")
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
          "DECISIONS has taken. Run `--exact` to check the instrument; read §2, "
          "§4 and §5 to decide whether the model's numbers are publishable.\n")

    A("## 7. What this does not establish\n")
    A("1. **Only `dp_source=\"model\"`.** Under `dp_source=\"reference\"` DP "
      "advances from reference physics and the model is absent from the aging "
      "path by construction, so its EOL cannot reflect model quality (O-9).\n")
    A("2. **One seasonal forcing.** The rollout's `(K_w, Ta_w)` is a fixed "
      "function of day-of-year, not a sample from the training distribution. It "
      "is a scenario, not a population, and the error here is the error on that "
      "scenario.\n")
    A("3. **Teacher forcing restores theta_TO only.** The gas states are "
      "downstream of the thermal branch through the cascade; resetting them too "
      "would score the thermal prediction through a channel it does not drive. "
      "The consequence for §5 is that its gas figures are **free-running "
      "only**: there is no single-window / accumulation split for them, because "
      "the teacher-forced rollout is still free-running in the gas channels.\n")
    A("4. **§5 is one point of one scenario, not a distribution.** It is the "
      "concentration at one window of one seasonal forcing. ANALYSIS_PLAN §4 "
      "supplies the distribution by running seven seeds and reporting median "
      "and full min-max; `scripts/aggregate_results.py` assembles that, and one "
      "run of this script is one of its inputs rather than an answer on its "
      "own.\n")

    # `args.out`, not a module-level name. `out_path` was assigned as a local in
    # `main` and read here as a global, so every markdown render raised
    # NameError — the whole non-`--json-out` path. It survived because the only
    # exercise this script had since `--out` was added went through
    # `--json-out`, which returns before this function.
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS" if args.exact
          else "\n(no gate for a real checkpoint; read §2, §4 and §5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
