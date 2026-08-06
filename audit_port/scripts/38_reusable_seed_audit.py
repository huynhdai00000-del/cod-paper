#!/usr/bin/env python3
"""Which existing runs count as matrix seeds, and which have to be repeated?

Before launching ~110 training runs it is worth knowing exactly which of the
finished ones can be kept. Getting this wrong is expensive in both directions:
re-running something valid wastes GPU hours, and **keeping something invalid
poisons a cell**, because a seed trained under different code is a sweep over
the code as well as over the seed and its min-max range is not a seed range.

Five conditions, and a run has to meet all five.

  1. **Its config hash matches a config that exists now.** A hash match means
     the identical config; anything else means the cell it belongs to is a
     guess.
  2. **It is on the frozen distribution** `fc4cb76c3b32ec17`.
  3. **It converged**, `stop_reason = converged_plateau`, and the wall-clock
     budget did not bind. A budget-bound run is reported as non-converged and
     is not a performance figure (README rule 5).
  4. **It has `model.pt`.** ANALYSIS_PLAN Amendment 1 makes end-of-rollout gas
     ppm the primary metric, and that is computed by rolling the weights
     forward. A run whose `run.json` survived but whose weights did not cannot
     produce the primary metric at all, so it cannot be a seed of the matrix no
     matter how good its 12 h numbers look.
  5. **Its training loop was not changed after it ran.** This is the one that
     cannot be measured from the artifact, so it is declared explicitly below,
     commit by commit, rather than assumed.

Condition 5's counterpart *is* measurable and is the expensive check here: the
checkpoint is reloaded under **current** code and rescored on the same tier,
and the recomputed metrics must reproduce what `run.json` recorded. That proves
the forward and evaluation paths have not moved under the artifact. It does not
prove the training path has not moved, which is what the declared table is for.

Run:  python audit_port/scripts/38_reusable_seed_audit.py
      python audit_port/scripts/38_reusable_seed_audit.py --artifacts artifacts
Exit: 0 if every run got a verdict and every reusable one reproduced its
      numbers, 1 if a run claimed to be reusable failed to reproduce.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.cells import cell_factors  # noqa: E402
from cod.config import load_config  # noqa: E402

FROZEN = "fc4cb76c3b32ec17"

#: ANALYSIS_PLAN §3's "worth reporting" floors, per state, in physical units.
#: The reproduction check is made **against these**, in absolute units, and not
#: as a relative tolerance. That is not a loosening; it is the correct
#: instrument, and using the wrong one manufactured a false failure here:
#:
#: A relative tolerance of 1e-6 reported that neither `o5` nor `o12` reproduced,
#: at 1.4e-3 and 2.0e-3 relative on the gases. The absolute differences were
#: 2.8e-7 and 1.5e-7 **ppm**. What moved was not the model: the
#: `denominator_median` of every state — a property of the RK45 ground truth
#: with no model in it at all — also differs by 1e-6 to 1e-5 relative, because
#: the recorded run was scored on Colab (Linux, numpy 2.0.2, scipy 1.16.3) and
#: the rescoring here runs on Windows with numpy 1.24.3 and scipy 1.14.1. A gas
#: MAE of 5e-4 ppm sitting on a ground truth that wobbles at 1e-5 relative
#: yields exactly the 1e-3 relative seen. `o12`'s theta_TO, whose magnitude is
#: four orders larger, reproduced at 1.4e-7 — and COD's gases are downstream of
#: theta, so a real forward-path change could not have moved them while leaving
#: theta bit-exact.
#:
#: This is C-9's lesson one level up: a **ratio to a quantity far below the
#: measurement floor** is arithmetically valid and physically empty. The
#: question that matters is whether the difference could change any verdict the
#: aggregator produces, and the smallest step it can take is the §3 floor.
REPRO_FLOOR = {"theta_TO": 0.2, "c_H2": 0.2, "c_C2H2": 0.1, "c_C2H4": 0.2,
               "c_CO": 0.5, "c_CO2": 1.0}
#: Fraction of the floor a reproduction difference may reach. 1% of the
#: smallest step the analysis can resolve is a difference that cannot affect a
#: verdict, and it is still three orders above what was measured.
REPRO_FRAC_OF_FLOOR = 0.01

#: Commits after which a run of the affected loop is superseded. Declared, with
#: what each one changed, because the artifact cannot show it: a checkpoint
#: records the commit it was trained at but not what happened afterwards.
#:
#: `train_physics` is `ode_physics_loss_shared`; `train_v34` is
#: `ode_physics_loss`. A commit that touches one does not touch the other.
LOSS_CHANGES = [
    {
        "commit": "14cd674",
        "date": "2026-08-04",
        "loops": ("train_physics",),
        "what": "the scalar-500 clamp in `ode_physics_loss_shared` became the "
                "per-state ceiling. It truncated states the model predicted "
                "correctly (ground-truth c_CO reaches 819 ppm, c_CO2 1456) and "
                "applied a tighter gas clamp to every baseline than to COD. "
                "The commit message states the consequence itself: 'the "
                "completed fno_in_cascade run was trained under the old clamp "
                "and its number is not attributable to the architecture. "
                "Ablation A used train_v34 and is unaffected.'",
    },
]


def _load_25():
    """`25_checkpoint_roundtrip.py`'s scorer, by path.

    Reused rather than restated: rebuilding the test set and recomputing the
    physical metrics is exactly what that script already does, and a second
    copy would be a second thing to keep in step with `run.py`.
    """
    spec = importlib.util.spec_from_file_location(
        "roundtrip", Path(__file__).with_name("25_checkpoint_roundtrip.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _registry() -> dict:
    reg = {}
    for p in sorted(ROOT.glob("configs/**/*.yaml")):
        try:
            cfg = load_config(p)
            reg[cfg.hash] = (p, cfg)
        except Exception:
            continue
    return reg


def _loop_of(cfg_raw: dict) -> str:
    kind = cfg_raw["model"]["kind"]
    return cfg_raw["training"].get(
        "loop", "train_v34" if kind.startswith("cod") else "train_physics")


def _commit_order() -> list:
    """Commit hashes oldest-first, so 'ran before X' is answerable."""
    import subprocess
    r = subprocess.run(["git", "log", "--format=%h", "--reverse"],
                       cwd=ROOT, capture_output=True, text=True)
    return [h.strip() for h in r.stdout.splitlines() if h.strip()]


def audit(run_dir: Path, reg: dict, order: list, rt) -> dict:
    """One run: a verdict plus the reason for it."""
    rj = run_dir / "run.json"
    j = json.loads(rj.read_text(encoding="utf-8"))
    cfgblk = j.get("config", {})
    out = j.get("outcome", {})
    crit = j.get("convergence_criterion", {})
    res = {"dir": run_dir.name, "seed": j.get("seed"),
           "commit": str(j.get("provenance", {}).get("git_commit", ""))[:7],
           "blockers": [], "notes": [], "variant": None, "repro": None}

    # 1. config hash resolves to a config that exists now
    h = cfgblk.get("config_hash")
    if h not in reg:
        res["blockers"].append(
            f"config hash {h} matches no config under configs/ — the cell it "
            "belongs to cannot be established")
        return res
    cfg_path, cfg = reg[h]
    factors = cell_factors(cfg.raw)
    res["variant"] = factors.variant
    res["config_path"] = str(cfg_path.relative_to(ROOT))

    # 2. frozen distribution
    if cfgblk.get("distribution_hash") != FROZEN:
        res["blockers"].append(
            f"distribution hash {cfgblk.get('distribution_hash')} is not the "
            f"frozen {FROZEN}")

    # 3. converged, and the budget did not bind
    if out.get("stop_reason") != "converged_plateau":
        res["blockers"].append(
            f"stop_reason is {out.get('stop_reason')!r}, not converged_plateau")
    budget = crit.get("max_wall_seconds")
    wall = out.get("wall_seconds")
    if budget and wall and wall >= budget * 0.999:
        res["blockers"].append(
            f"wall clock {wall:.0f} s reached the {budget:.0f} s budget")
    elif budget and wall:
        res["notes"].append(
            f"converged at {wall:.0f} s under a {budget:.0f} s cap "
            f"({100 * wall / budget:.0f}% of it), so the cap never bound and a "
            "larger cap would give the identical run")

    # 4. weights present
    if not (run_dir / "model.pt").is_file():
        res["blockers"].append(
            "no model.pt — Amendment 1's primary metric is computed by rolling "
            "the weights forward, so this run cannot produce it")

    # 5. was its loop changed after it ran
    loop = _loop_of(cfg.raw)
    res["loop"] = loop
    run_commit = res["commit"]
    idx = {c: i for i, c in enumerate(order)}
    for ch in LOSS_CHANGES:
        if loop not in ch["loops"]:
            continue
        i_run, i_ch = idx.get(run_commit), idx.get(ch["commit"])
        if i_run is None or i_ch is None:
            res["notes"].append(
                f"could not place commit {run_commit} against {ch['commit']} "
                "in history")
            continue
        if i_run < i_ch:
            res["blockers"].append(
                f"trained at {run_commit}, before {ch['commit']} "
                f"({ch['date']}) changed {loop}: {ch['what']}")

    # The measurable half of 5: does current code still reproduce it?
    if (run_dir / "model.pt").is_file():
        guard = float(cfg.raw["evaluation"].get("ground_truth", {})
                      .get("right_edge_guard", 0.9999))
        n_test = int(j.get("evaluation", {}).get("n_cases", 100))
        got, _ck = rt.score_from_checkpoint(run_dir / "model.pt", cfg.raw,
                                            n_test, guard)
        want = j.get("evaluation", {}).get("mae_physical_units", {})
        # Worst difference as a fraction of that state's own reporting floor,
        # so states in degC and in ppm are on one comparable scale.
        worst, worst_key, worst_abs = 0.0, None, 0.0
        for k, v in want.items():
            if k not in got or k not in REPRO_FLOOR:
                continue
            d = abs(float(v["mae"]) - float(got[k]["mae"]))
            frac = d / REPRO_FLOOR[k]
            if frac > worst:
                worst, worst_key, worst_abs = frac, k, d
        res["repro"] = (worst, worst_key, worst_abs)
        # The model-free control for that number. `denominator_median` is the
        # median ground-truth variation of each state: RK45 only, no model in
        # it. If it moves too, what moved is the platform's floating point and
        # not the code — which is what the docstring above asserts, so it is
        # measured here rather than argued.
        dm = 0.0
        for k, v in want.items():
            a = v.get("denominator_median")
            b = got.get(k, {}).get("denominator_median")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                dm = max(dm, 0.0 if a == b
                         else abs(a - b) / max(abs(float(a)), 1e-30))
        res["denom_rel"] = dm
        if worst > REPRO_FRAC_OF_FLOOR:
            res["blockers"].append(
                f"current code does NOT reproduce its recorded metrics: "
                f"{worst_key} differs by {worst_abs:.3g}, which is "
                f"{worst:.1%} of its {REPRO_FLOOR[worst_key]:g} reporting "
                "floor. A difference that large can change a verdict, so the "
                "forward or evaluation path has moved under this artifact.")

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "audit_port" / "REUSABLE_SEEDS.md")
    args = ap.parse_args()

    reg = _registry()
    order = _commit_order()
    rt = _load_25()
    dirs = sorted(p.parent for p in args.artifacts.rglob("run.json"))
    if not dirs:
        raise SystemExit(f"no run.json under {args.artifacts}")

    print(f"[audit] {len(dirs)} finished run(s) under {args.artifacts}")
    results = []
    for d in dirs:
        print(f"\n--- {d.name} ---")
        r = audit(d, reg, order, rt)
        results.append(r)
        if r["repro"] is not None:
            worst, key, wabs = r["repro"]
            print(f"  rescored under current code: worst difference "
                  f"{wabs:.3g} on {key}, i.e. {worst:.2%} of its reporting "
                  "floor")
            print(f"  model-free control: the ground-truth denominator_median "
                  f"moves {r['denom_rel']:.1e} relative between the recorded "
                  "run's platform and this one")
        for n in r["notes"]:
            print(f"  note: {n}")
        if r["blockers"]:
            print(f"  VERDICT: REPEAT ({len(r['blockers'])} blocker(s))")
            for b in r["blockers"]:
                print(f"    - {b}")
        else:
            print(f"  VERDICT: REUSABLE as {r['variant']} seed {r['seed']}")

    reusable = [r for r in results if not r["blockers"]]
    md = ["# Which finished runs count as matrix seeds", "",
          f"Generated by `audit_port/scripts/38_reusable_seed_audit.py` over "
          f"`{args.artifacts.name}/`.", "",
          "A run is reusable only if all five conditions hold: its config hash "
          "resolves to a config that exists now, it is on the frozen "
          "distribution, it converged without the wall budget binding, its "
          "`model.pt` is present (Amendment 1's primary metric is computed "
          "from the weights), and its training loop was not changed after it "
          "ran. The last is declared rather than measured; its measurable "
          "counterpart — that current code still reproduces the recorded "
          "metrics from the checkpoint — is checked here at "
          f"{REPRO_FRAC_OF_FLOOR:.0%} of each state's ANALYSIS_PLAN §3 "
          "reporting floor — an absolute bar in physical units, because a "
          "relative one on a 5e-4 ppm quantity measures the platform's RK45 "
          "round-off rather than the code (see the module docstring).", "",
          "| run | cell | seed | loop | rescored (worst abs diff) | verdict |",
          "|---|---|---|---|---|---|"]
    for r in results:
        rep = ("—" if r["repro"] is None
               else f"{r['repro'][2]:.2g} ({r['repro'][0]:.2%} of floor)")
        md.append(f"| `{r['dir']}` | {r['variant'] or '—'} | {r['seed']} | "
                  f"{r.get('loop', '—')} | {rep} | "
                  + ("**reusable**" if not r["blockers"] else "repeat") + " |")
    md.append("")
    for r in results:
        if not r["blockers"]:
            continue
        md.append(f"**`{r['dir']}` — repeat.**")
        md.append("")
        for b in r["blockers"]:
            md.append(f"- {b}")
        md.append("")
    md.append("## Loop changes that supersede an earlier run")
    md.append("")
    for ch in LOSS_CHANGES:
        md.append(f"- **{ch['commit']}** ({ch['date']}), affects "
                  f"{', '.join(ch['loops'])}: {ch['what']}")
    md.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Machine-readable, so `scripts/make_launch_list.py` subtracts what is
    # already done rather than being told. A launch list that carries a
    # hand-maintained "already done" list is one edit away from re-running 14
    # GPU-hours or, worse, leaving a cell at six seeds and calling it seven.
    js = args.out.with_name("reusable_seeds.json")
    js.write_text(json.dumps({
        "generated_from": str(args.artifacts.name),
        "criteria": {"frozen_distribution": FROZEN,
                     "repro_frac_of_floor": REPRO_FRAC_OF_FLOOR},
        "reusable": [{"variant": r["variant"], "seed": r["seed"],
                      "dir": r["dir"], "commit": r["commit"]}
                     for r in reusable],
        "repeat": [{"variant": r["variant"], "seed": r["seed"],
                    "dir": r["dir"], "blockers": r["blockers"]}
                   for r in results if r["blockers"]],
    }, indent=2), encoding="utf-8")
    print(f"[audit] wrote {js}")

    print(f"\n{'=' * 72}")
    print(f"[audit] {len(reusable)}/{len(results)} reusable:")
    for r in reusable:
        print(f"  {r['variant']} seed {r['seed']}  ({r['dir']})")
    print(f"[audit] wrote {args.out}")
    # A run that claimed to be reusable but failed to reproduce is the failure
    # this script exists to catch; a run correctly identified as "repeat" is a
    # finding, not an error.
    bad = [r for r in results
           if r["repro"] is not None and r["repro"][0] > REPRO_FRAC_OF_FLOOR]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
