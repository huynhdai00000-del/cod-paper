#!/usr/bin/env python3
"""The J-89 checklist applied to `scripts/aggregate_results.py`, step 6 included.

PORT_LOG J-89 requires a new metric to be checked against the silent-sentinel
pattern **before** it is allowed to produce a number. An aggregator is a metric:
it turns a directory of runs into medians, ranges and verdicts, and the same
failure applies — handed an input for which its answer is undefined, does it
return a finite, plausible value that nothing downstream can tell apart from a
real measurement?

Every check builds synthetic run directories with known contents and asserts on
the report. Synthetic because the degenerate inputs are the point: a cell with
one seed, a cell with zero variance, a cell that never converged, a metric
absent from `run.json`. Those cannot be produced on demand by training.

  1. **The null case.** A cell with one converged seed a side, and a pair of
     cells whose values are identical. Neither may come back as a separation.
  2. **Is a sentinel inside the output range?** A `run.json` with the metric
     absent must not read as 0.0 ppm — a perfect cell — and a run that resolves
     to no cell must not be silently pooled into one.
  3. **Does a window or normaliser depend on the quantity measured?** The
     converged-only filter does: convergence and error come from the same
     training. Asserted that the convergence rate is reported for every cell and
     that a cell with no converged seed yields no error distribution at all.
  4. **Do the stopping rule and the definition agree?** `converged` and
     `stop_reason` must not disagree; a run claiming `converged: true` with a
     `stop_reason` of `wall_clock_budget` is a contradiction the aggregator has
     to surface rather than average.
  5. **Can the gate fail?** A pair constructed to clear both bars must return
     "difference worth reporting", and one constructed to clear neither must
     not. Both directions, by injection.
  6. **Vary something the report should depend on, and confirm it moves.**
     Change one seed's MAE and the median, the range and the verdict must all
     change; change something the report must *not* depend on — the tag in the
     directory name — and the report must be byte-identical. This is the check
     that caught J-89 instance 4, where three architectures reported
     bit-identical clamp series because the diagnostic did not depend on the
     model at all.

Run:  python audit_port/scripts/37_aggregator_sentinel_check.py
Exit: 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cod.cells import cell_factors  # noqa: E402
from cod.config import load_config  # noqa: E402
import aggregate_results as agg  # noqa: E402

FAILURES: list = []

#: Two cells of one architecture, so a pair exists to compare.
CFG_CASCADE = ROOT / "configs" / "matrix" / "fno_in_cascade.yaml"
CFG_MONO = ROOT / "configs" / "matrix" / "fno_monolithic.yaml"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else '[FAIL]'} {name}" + (f" — {detail}" if detail
                                                      else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def _run_json(cfg_path: Path, seed: int, theta: float | None, gas: float | None,
              converged: bool = True, stop: str = "converged_plateau",
              status: str = "run", dist_hash: str | None = None) -> dict:
    """A minimal but structurally honest `run.json`."""
    cfg = load_config(cfg_path)
    ev_states = {}
    for m in agg.METRICS:
        v = theta if m.kind == "thermal" else gas
        if v is None:
            continue
        ev_states[m.key] = {"name": m.key, "unit": m.unit, "mae": v,
                            "n_cases": 100}
    return {
        "provenance": {"git_commit": "0" * 40},
        "config": {"config_path": str(cfg_path), "config_hash": cfg.hash,
                   "distribution_hash": dist_hash or cfg.distribution_hash,
                   "experiment_name": cfg.raw["experiment"]["name"],
                   "variant": cfg.raw["experiment"]["variant"]},
        "seed": seed,
        "status": status,
        "cell": cell_factors(cfg.raw).to_dict(),
        "model_kind": cfg.raw["model"]["kind"],
        "n_ic": 8000,
        "outcome": {"converged": converged, "stop_reason": stop,
                    "epochs_reached": 5000, "wall_seconds": 1000.0},
        "evaluation": {"tier": "T1_in_distribution",
                       "tier_source": "realistic_sampler", "tier_seed": 999,
                       "n_cases": 100, "mae_physical_units": ev_states},
    }


def _write(root: Path, tag: str, rec: dict) -> Path:
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return d


def _report(root: Path) -> tuple:
    reg = agg._registry()
    runs = [agg.read_run(p, reg) for p in sorted(root.rglob("run.json"))]
    return agg.build_report(runs, root)


def _cell(md_root: Path, cfg: Path, seeds: dict, **kw):
    """`seeds` maps seed -> (theta, gas)."""
    for s, (th, g) in seeds.items():
        _write(md_root, f"{cfg.stem}_s{s}", _run_json(cfg, s, th, g, **kw))


# ── 1. the null case ─────────────────────────────────────────────────────────
def check_null_case(tmp: Path) -> None:
    print("\n1. the null case — no variance, and one seed a side")

    root = tmp / "one_seed"
    _cell(root, CFG_CASCADE, {1: (1.0, 0.001)})
    _cell(root, CFG_MONO, {1: (9.0, 0.500)})
    md, _ = _report(root)
    check("one seed a side does not become a separation claim",
          "insufficient seeds for a separation claim" in md,
          "with n=1 the min-max range is a point and any two distinct points "
          "are disjoint")
    check("bar (b) is reported as not evaluable, not as passed",
          md.count("| yes | yes | difference worth reporting |") == 0)

    root = tmp / "zero_variance"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.010) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (2.0, 0.010) for s in range(1, 8)})
    md, _ = _report(root)
    check("two identical cells return 'no operationally meaningful difference'",
          "no operationally meaningful difference" in md
          and "difference worth reporting" not in md)


# ── 2. sentinels inside the output range ─────────────────────────────────────
def check_sentinels(tmp: Path) -> None:
    print("\n2. sentinels — a missing metric, and an unresolvable run")

    root = tmp / "missing_metric"
    _cell(root, CFG_CASCADE, {s: (2.0, None) for s in range(1, 8)})
    md, _ = _report(root)
    gas_rows = [ln for ln in md.splitlines() if ln.startswith("| c_H2 (ppm) |")]
    check("an absent metric does not become 0.0",
          all("0.0 |" not in r and "| 0 |" not in r.replace("| 0 |", "| 0 |", 1)
              or "—" in r for r in gas_rows),
          f"rows: {gas_rows[:1]}")
    check("an absent metric reads as em-dash",
          any("| — | — | — | 0 |" in r for r in gas_rows), f"{gas_rows[:1]}")

    root = tmp / "unresolvable"
    rec = _run_json(CFG_CASCADE, 1, 2.0, 0.01)
    rec.pop("cell")
    rec["config"]["config_hash"] = "deadbeefdeadbeef"
    _write(root, "mystery", rec)
    md, n = _report(root)
    check("a run matching no config is reported unresolved, not pooled",
          "Unresolved runs" in md and n >= 1)
    check("an unresolved run makes the aggregation fail", n >= 1, f"n={n}")

    # A truncated write must not take the rest of the sweep with it, and must
    # not pass silently either.
    root = tmp / "broken_json"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.01) for s in range(1, 8)})
    (root / "torn" ).mkdir(parents=True, exist_ok=True)
    (root / "torn" / "run.json").write_text('{"config": {"config_ha',
                                            encoding="utf-8")
    reg = agg._registry()
    good, bad = [], []
    for p in sorted(root.rglob("run.json")):
        try:
            good.append(agg.read_run(p, reg))
        except Exception as exc:
            bad.append(f"{p}: {exc}")
    check("an unreadable run.json does not lose the other runs",
          len(good) == 7 and len(bad) == 1, f"{len(good)} good, {len(bad)} bad")
    md, _ = agg.build_report(good, root)
    check("and the other seven still aggregate",
          "**Converged 7/7 seeds.**" in md)

    # A run on another distribution is not a cell of this matrix.
    root = tmp / "offdist"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.01) for s in range(1, 8)},
          dist_hash="ffffffffffffffff")
    md, _ = _report(root)
    check("a run on another distribution is excluded, and said so",
          "on another distribution, excluded: **7**" in md)


# ── 3. a filter derived from the quantity measured ───────────────────────────
def check_censoring(tmp: Path) -> None:
    print("\n3. the converged-only filter, which depends on what it measures")

    root = tmp / "none_converged"
    _cell(root, CFG_CASCADE, {s: (40.0, 5.0) for s in range(1, 8)},
          converged=False, stop="wall_clock_budget")
    md, _ = _report(root)
    check("a cell with no converged seed yields no error distribution",
          "No converged seed" in md)
    check("its numbers are still present, bracketed, never dropped",
          "[40.000]" in md, "§4: a non-converged seed is never dropped")
    check("convergence rate is stated before the numbers",
          md.index("**Converged 0/7 seeds.**") < md.index("[40.000]"))

    root = tmp / "partial"
    seeds = {s: (2.0, 0.01) for s in range(1, 6)}
    seeds.update({6: (40.0, 5.0), 7: (41.0, 5.1)})
    for s, (th, g) in seeds.items():
        _write(root, f"c_s{s}",
               _run_json(CFG_CASCADE, s, th, g, converged=s <= 5,
                         stop="converged_plateau" if s <= 5
                         else "wall_clock_budget"))
    md, _ = _report(root)
    check("a partly converged cell reports 5/7 and escalates to 21 seeds",
          "**Converged 5/7 seeds.**" in md and "Escalate to 21 seeds" in md)
    check("the non-converged seeds do not enter the median",
          "| theta_TO (degC) | 2.000 | 2.000 | 2.000 | 5 |" in md,
          "median over converged seeds only, n = 5")


# ── 4. stopping rule against definition ──────────────────────────────────────
def check_stop_agreement(tmp: Path) -> None:
    print("\n4. do `converged` and `stop_reason` agree")

    root = tmp / "contradiction"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.01) for s in range(1, 8)},
          converged=True, stop="wall_clock_budget")
    md, n = _report(root)
    check("a run claiming convergence with a budget stop_reason is flagged",
          n >= 1 and "stop_reason" in md.lower(),
          "`converged: true` and `stop_reason: wall_clock_budget` cannot both "
          "be right; averaging such a run would launder a budget-bound result "
          "into a converged one")


# ── 5. can the gate fail, both directions ────────────────────────────────────
def check_gate_fires(tmp: Path) -> None:
    print("\n5. can the verdict fire — by injection, both directions")

    # Separated by far more than the floor, ranges disjoint.
    root = tmp / "clear_both"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.010 + 0.0001 * s) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (2.4, 5.000 + 0.0001 * s) for s in range(1, 8)})
    md, _ = _report(root)
    check("a real, separated difference returns 'difference worth reporting'",
          "difference worth reporting" in md)

    # Separated cleanly but far below the floor: bar (b) yes, bar (a) no.
    root = tmp / "separable_negligible"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.0100 + 1e-6 * s) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (2.0, 0.0200 + 1e-6 * s) for s in range(1, 8)})
    md, _ = _report(root)
    check("clean separation below the floor is 'operationally negligible'",
          "statistically separable, operationally negligible" in md,
          "0.01 ppm against a 0.2 ppm floor")

    # Large difference, overlapping ranges: bar (a) yes, bar (b) no.
    root = tmp / "suggestive"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.10 + 0.30 * (s % 3)) for s in
                              range(1, 8)})
    _cell(root, CFG_MONO, {s: (2.0, 0.50 + 0.30 * (s % 3)) for s in range(1, 8)})
    md, _ = _report(root)
    check("a large difference with overlapping seeds is 'suggestive'",
          "suggestive, seeds overlap" in md)

    # The §1 confound control: thermal MAE more than 2x apart.
    root = tmp / "confounded"
    _cell(root, CFG_CASCADE, {s: (1.0, 0.01) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (9.0, 5.00) for s in range(1, 8)})
    md, _ = _report(root)
    check("a 9x thermal gap marks the gas comparison CONFOUNDED",
          "CONFOUNDED" in md,
          "gas error inherits thermal error through V_arr, exponential in T")

    root = tmp / "not_confounded"
    _cell(root, CFG_CASCADE, {s: (1.0, 0.01) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (1.5, 5.00) for s in range(1, 8)})
    md, _ = _report(root)
    check("a 1.5x thermal gap does not", "CONFOUNDED" not in md)


# ── 6. vary what it should depend on; hold what it should not ────────────────
def check_dependence(tmp: Path) -> None:
    print("\n6. vary something the report should depend on, and confirm it moves")

    base = tmp / "dep_base"
    _cell(base, CFG_CASCADE, {s: (2.0 + 0.01 * s, 0.010) for s in range(1, 8)})
    md_base, _ = _report(base)

    # (a) one seed's value changes -> median, range and verdict must move.
    moved = tmp / "dep_moved"
    shutil.copytree(base, moved)
    p = moved / f"{CFG_CASCADE.stem}_s4" / "run.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["evaluation"]["mae_physical_units"]["theta_TO"]["mae"] = 30.0
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    md_moved, _ = _report(moved)
    check("changing one seed's MAE changes the report", md_moved != md_base)
    check("it changes the max of the range", "30.00" in md_moved)
    check("and it trips the max/min escalation rule",
          "Escalate to 21 seeds" in md_moved
          and "Escalate to 21 seeds" not in md_base,
          "max/min thermal MAE above 2x, §4")

    # (b) something it must NOT depend on: the directory name.
    renamed = tmp / "dep_renamed"
    shutil.copytree(base, renamed)
    for d in sorted(renamed.iterdir()):
        d.rename(d.parent / (d.name + "_someothertag"))
    md_renamed, _ = _report(renamed)
    check("renaming the run directories does not change the report",
          md_renamed.replace(str(renamed), "X")
          == md_base.replace(str(base), "X"),
          "the cell comes from the config, not from the directory name")

    # (c) the dual of J-89 instance 4: two cells that differ must not report
    #     the same numbers. Built to differ in exactly one cell's values.
    two = tmp / "dep_two_cells"
    _cell(two, CFG_CASCADE, {s: (2.0, 0.01) for s in range(1, 8)})
    _cell(two, CFG_MONO, {s: (7.0, 0.90) for s in range(1, 8)})
    md_two, _ = _report(two)
    check("two cells with different inputs report different numbers",
          "| theta_TO (degC) | 2.000 | 2.000 | 2.000 | 7 |" in md_two
          and "| theta_TO (degC) | 7.000 | 7.000 | 7.000 | 7 |" in md_two)


# ── the factorial arithmetic, checked against hand-computed values ───────────
def check_factorial(tmp: Path) -> None:
    print("\n7. the factorial — main effects, interaction, and what is not "
          "identified")

    cfgs = {
        1: ROOT / "configs" / "matrix" / "fno_baseline_monolithic.yaml",
        2: ROOT / "configs" / "matrix" / "fno_monolithic.yaml",
        3: ROOT / "configs" / "matrix" / "fno_baseline_in_cascade.yaml",
        4: ROOT / "configs" / "matrix" / "fno_in_cascade.yaml",
    }
    #: cell -> thermal MAE. Chosen so every term is a distinct round number:
    #: main cascade = ((3+6)-(5+10))/2 = -3; main baseline = ((5+3)-(10+6))/2
    #: = -4; interaction = (3-6)-(5-10) = +2.
    vals = {1: 5.0, 2: 10.0, 3: 3.0, 4: 6.0}

    root = tmp / "factorial_full"
    for n, cfg in cfgs.items():
        _cell(root, cfg, {s: (vals[n], 0.01 * n) for s in range(1, 8)})
    md, _ = _report(root)
    row = [ln for ln in md.splitlines() if ln.startswith("| theta_TO (degC) |")
           and "n/i" not in ln and ln.count("|") == 9]
    check("a full quadrant yields main effects and an interaction",
          bool(row), f"rows found: {len(row)}")
    if row:
        cells = [c.strip() for c in row[0].strip("|").split("|")]
        check("main effect of cascade = -3.000", cells[1] == "-3.000", cells[1])
        check("main effect of baseline = -4.000", cells[2] == "-4.000", cells[2])
        check("interaction = 2.000", cells[3] == "2.000", cells[3])
        check("simple effect of baseline in-cascade = -3.000",
              cells[6] == "-3.000", cells[6])
        check("simple effect of baseline monolithic = -5.000",
              cells[7] == "-5.000", cells[7])

    # Drop cell 2: no main effect and no interaction may survive.
    root = tmp / "factorial_partial"
    for n, cfg in cfgs.items():
        if n == 2:
            continue
        _cell(root, cfg, {s: (vals[n], 0.01 * n) for s in range(1, 8)})
    md, _ = _report(root)
    check("a partial quadrant reports main effects as not identified",
          "so neither main effect nor the interaction is identified" in md)
    check("and does not print a number in their place",
          all(ln.startswith("| theta_TO (degC) | n/i | n/i | n/i |")
              for ln in md.splitlines()
              if ln.startswith("| theta_TO (degC) |") and ln.count("|") == 9),
          "J-89 Form A: a main effect computed from three cells would be a "
          "plausible number for an undefined quantity")


# ── C-9: no gas percentage may reach the output ──────────────────────────────
def check_no_gas_percent(tmp: Path) -> None:
    print("\n8. C-9 — no gas percentage anywhere in the output")
    root = tmp / "c9"
    _cell(root, CFG_CASCADE, {s: (2.0, 0.01) for s in range(1, 8)})
    _cell(root, CFG_MONO, {s: (7.0, 0.90) for s in range(1, 8)})
    md, _ = _report(root)
    bad = [ln for ln in md.splitlines()
           if any(g.key in ln for g in agg.GASES) and "%" in ln]
    check("no line carries both a gas name and a percent sign", not bad,
          f"{bad[:1]}")
    # Prose may name NMAE — §7 explains why it is absent — but no *table* may
    # carry it, because a table cell is what gets quoted.
    tbl = [ln for ln in md.splitlines()
           if ln.startswith("|") and "nmae" in ln.lower()]
    check("no table row or header mentions NMAE", not tbl, f"{tbl[:1]}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="agg_sentinel_"))
    print(f"[tmp] {tmp}")
    try:
        check_null_case(tmp)
        check_sentinels(tmp)
        check_censoring(tmp)
        check_stop_agreement(tmp)
        check_gate_fires(tmp)
        check_dependence(tmp)
        check_factorial(tmp)
        check_no_gas_percent(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[PASS] the aggregator survives the J-89 checklist, steps 1-6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
