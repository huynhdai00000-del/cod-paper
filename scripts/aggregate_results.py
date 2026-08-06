#!/usr/bin/env python3
"""Read every `run.json` under a results directory and build the ANALYSIS_PLAN tables.

    python scripts/aggregate_results.py --results /path/to/cod_matrix \
        --out audit_port/MATRIX_RESULTS.md

What it produces, and where each rule comes from:

  * **inventory and integrity** — one distribution hash across the whole matrix,
    every run resolved to a cell, smoke tests segregated from production runs;
  * **per cell: convergence rate first, error distribution second**
    (ANALYSIS_PLAN §4). Every seed's `stop_reason` in a table, and a
    non-converged seed reported as non-converged and never dropped;
  * **median and full min-max across seeds** — not mean +- sd, because M-2 and
    N-9 describe unstable training in the monolithic regime, so the
    distributions are expected skewed or bimodal and a mean describes neither
    mode (§4);
  * **the two reporting bars applied automatically** (§3) — physical relevance
    against the engineering floors, and seed separation by non-overlapping
    min-max — with the four pre-approved verdict phrasings and no upgrading;
  * **the mandatory 2x thermal confound control** (§1) applied before any gas
    comparison counts toward H1;
  * **H1 on Amendment 1's primary metric** — end-of-rollout gas ppm at K = 0.95
    and K = 1.10, read from `<run>/rollout.json` — with the 12 h gas MAE kept
    but reported separately as secondary, under Amendment 1's own reasoning;
  * **the 2x2 factorial** per architecture with both main effects and the
    interaction (J-92, Amendment 1).

Three things it deliberately refuses to do:

  * **No gas percentages, anywhere.** C-9 as tightened 2026-08-02: the median
    12 h gas variation is 0.001%-0.046% of each gas's engineering threshold, so
    a gas NMAE is a ratio to a physically empty quantity. Absolute ppm against
    IEC 60599 only. There is a self-check for this at the end of the run.
  * **No separation claim from fewer than `MIN_SEEDS_FOR_SEPARATION` converged
    seeds.** With one seed per side the min-max "ranges" are points and are
    disjoint unless exactly equal, so bar (b) would pass on nothing at all —
    a plausible verdict for an undefined input, i.e. J-89 Form A. Such a
    comparison is reported as insufficient, not as separable.
  * **No substitute for a metric a run has not been scored on.** ANALYSIS_PLAN
    Amendment 1 makes end-of-rollout gas ppm the primary metric; it comes from
    `<run>/rollout.json`, written by `24_rollout_thermal_error.py --json-out`.
    A run without that file contributes nothing to the primary tables rather
    than a zero, and the demoted 12 h gas MAE is never promoted in its place.

Exit: 0 if the aggregation is clean, 1 if any integrity problem was found —
a run that resolves to no cell, duplicate seeds inside a cell, a cell whose
runs disagree about the evaluation tier, two cells claiming one factorial
slot, or a gas percentage reaching the output. Those make the tables above
unsafe to read, so they fail rather than print a footnote.

Verified by `audit_port/scripts/37_aggregator_sentinel_check.py`, which runs
this module against synthetic run directories built to be degenerate: J-89
steps 1-6, including step 6's "vary something the metric should depend on and
confirm it moves".
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cod.cells import CellFactors, cell_factors  # noqa: E402
from cod.config import load_config  # noqa: E402

# ── The metrics, their units, and the §3 floors ───────────────────────────────
#: (key in mae_physical_units, unit, engineering threshold, "worth reporting"
#: floor). The floor is 10% of the threshold, and the thresholds are the ones
#: already in cod/eval/metrics.py — what a practitioner needs, not targets tuned
#: against. Gas NMAE is absent by construction: there is no percentage column.


@dataclass(frozen=True)
class Metric:
    key: str
    unit: str
    threshold: float
    floor: float
    kind: str          # "thermal" | "gas"


METRICS = [
    Metric("theta_TO", "degC", 2.0, 0.2, "thermal"),
    Metric("c_H2", "ppm", 2.0, 0.2, "gas"),
    Metric("c_C2H2", "ppm", 1.0, 0.1, "gas"),
    Metric("c_C2H4", "ppm", 2.0, 0.2, "gas"),
    Metric("c_CO", "ppm", 5.0, 0.5, "gas"),
    Metric("c_CO2", "ppm", 10.0, 1.0, "gas"),
]
THERMAL = METRICS[0]
GASES = [m for m in METRICS if m.kind == "gas"]

#: ANALYSIS_PLAN Amendment 1's primary metric: absolute gas concentration error
#: in ppm at the end of the rollout, at these two scenarios. Produced by
#: `24_rollout_thermal_error.py --json-out <run>/rollout.json`, which is the
#: only place it exists — nothing in `run.json` carries it.
#:
#: Same floors as the 12 h gas MAE, because Amendment 1 substitutes the metric
#: and leaves §3 otherwise unchanged: "the two bars in §3 ... applies unchanged
#: with the metric substituted".
ROLLOUT_SCENARIOS = (0.95, 1.10)


def rollout_key(K: float, gas: str) -> str:
    return f"rollout_K{K:.2f}_{gas}"


ROLLOUT_METRICS = [
    Metric(rollout_key(K, g.key), "ppm", g.threshold, g.floor, "rollout")
    for K in ROLLOUT_SCENARIOS for g in GASES
]

#: DISTRIBUTION_FREEZE.md §1. A run trained on a different distribution is not
#: a cell of this matrix and cannot be compared with one, so the frozen hash is
#: what defines membership — of a *config* in the declared cell set, and of a
#: *run* in any table below. Deriving membership this way rather than from a
#: list of names keeps the Phase-1 reproduction configs (`v57_faithful`, on the
#: v57 sampler) out without naming them.
FROZEN_DISTRIBUTION_HASH = "fc4cb76c3b32ec17"

#: §4 asks for 7 seeds per cell. Below this many *converged* seeds on a side,
#: the min-max range is not a range and bar (b) is not evaluable.
MIN_SEEDS_FOR_SEPARATION = 3
#: §4: escalate to 21 seeds for any cell with a non-converged seed or a
#: max/min thermal MAE ratio above this.
SEED_TARGET = 7
ESCALATION_TARGET = 21
UNSTABLE_RATIO = 2.0
#: `cod/training/harness.py` sets `converged = True` at exactly one place, and
#: this is the `stop_reason` it sets there. The two are the same fact recorded
#: twice, so a run where they disagree is a run whose convergence cannot be
#: read.
CONVERGED_STOP_REASON = "converged_plateau"
#: §1: if the median thermal MAE of two configurations differs by more than
#: this factor, their gas comparison is confounded and counts toward H1 in
#: neither direction.
CONFOUND_RATIO = 2.0


# ── One run ───────────────────────────────────────────────────────────────────
@dataclass
class Run:
    path: Path
    seed: int
    status: str
    converged: bool
    stop_reason: str
    epochs: int | None
    wall_seconds: float | None
    config_hash: str
    distribution_hash: str
    commit: str
    tier: str
    tier_source: str
    n_cases: int | None
    mae: dict            #: metric key -> float, or absent if run.json lacks it
    cell: CellFactors | None
    unresolved_reason: str | None = None

    @property
    def variant(self) -> str:
        return self.cell.variant if self.cell else "<unresolved>"


def _registry() -> dict:
    """config hash -> CellFactors, from the local config tree.

    Runs written before `run.json` carried its own `cell` block are resolved
    this way. It is exact — a hash match means the identical config — but it
    depends on the file still being here, which is why `run.py` now records the
    cell directly (cod/config.py `summary`).
    """
    reg = {}
    for p in sorted(ROOT.glob("configs/**/*.yaml")):
        try:
            cfg = load_config(p)
            reg[cfg.hash] = cell_factors(cfg.raw)
        except Exception:
            continue
    return reg


def _read_rollout(run_dir: Path) -> dict:
    """Amendment 1's primary metric, from `<run>/rollout.json` if it is there.

    Written by `24_rollout_thermal_error.py --json-out`. Absent for a run that
    has not been scored yet, and absence stays absence: an unscored run must not
    contribute a zero to a median, which is a perfect cell wearing a plausible
    number (J-89 step 2).

    The stored quantity is signed (`model - reference`). What the two bars are
    applied to is its **magnitude**: H1 is about which configuration ends the
    rollout closer to the right concentration, and a cell whose seeds straddle
    zero would otherwise get a median near zero for being inconsistent.
    """
    p = run_dir / "rollout.json"
    if not p.is_file():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    except Exception:
        return {}
    out = {}
    known = {m.key for m in GASES}
    for r in rows:
        K = r.get("K")
        err = r.get("gas_err")
        names = r.get("gas_names")
        # Zipped by NAME, from the names the producer wrote alongside the
        # values. Zipping against this file's own gas list would depend on two
        # orderings in two files staying in step for ever.
        if K is None or not isinstance(err, list) or not isinstance(names, list):
            continue
        if len(names) != len(err) or not any(abs(K - s) < 1e-9
                                             for s in ROLLOUT_SCENARIOS):
            continue
        for name, e in zip(names, err):
            if name in known:
                out[rollout_key(float(K), name)] = abs(float(e))
    return out


def read_run(path: Path, reg: dict) -> Run:
    j = json.loads(path.read_text(encoding="utf-8"))
    cfgblk = j.get("config", {})
    out = j.get("outcome", {})
    ev = j.get("evaluation", {})

    mae = {}
    for m in METRICS:
        d = ev.get("mae_physical_units", {}).get(m.key)
        # Missing stays missing. A metric absent from run.json must not become
        # 0.0, which would read as a perfect cell (J-89 step 2).
        if isinstance(d, dict) and isinstance(d.get("mae"), (int, float)):
            mae[m.key] = float(d["mae"])
    mae.update(_read_rollout(path.parent))

    cell, reason = None, None
    if isinstance(j.get("cell"), dict):
        c = j["cell"]
        try:
            cell = CellFactors(**c)
        except TypeError:
            reason = "run.json 'cell' block does not match cod/cells.CellFactors"
    if cell is None:
        h = cfgblk.get("config_hash")
        if h in reg:
            cell = reg[h]
        elif reason is None:
            reason = (f"config hash {h} matches no config under configs/, and "
                      "run.json carries no 'cell' block")

    return Run(
        path=path, seed=int(j.get("seed", -1)), status=str(j.get("status", "?")),
        converged=bool(out.get("converged", False)),
        stop_reason=str(out.get("stop_reason", "?")),
        epochs=out.get("epochs_reached"), wall_seconds=out.get("wall_seconds"),
        config_hash=str(cfgblk.get("config_hash", "?")),
        distribution_hash=str(cfgblk.get("distribution_hash", "?")),
        commit=str(j.get("provenance", {}).get("git_commit", "?"))[:8],
        tier=str(ev.get("tier", "?")), tier_source=str(ev.get("tier_source", "?")),
        n_cases=ev.get("n_cases"), mae=mae, cell=cell, unresolved_reason=reason,
    )


# ── A cell: every seed of one variant ─────────────────────────────────────────
@dataclass
class Cell:
    factors: CellFactors
    runs: list

    @property
    def converged(self) -> list:
        return [r for r in self.runs if r.converged]

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def conv_rate(self) -> str:
        return f"{len(self.converged)}/{self.n}"

    def values(self, key: str, converged_only: bool = True) -> list:
        src = self.converged if converged_only else self.runs
        return [r.mae[key] for r in src if key in r.mae]

    def stat(self, key: str) -> tuple | None:
        """(median, min, max, n) over converged seeds, or None if nothing."""
        v = self.values(key)
        if not v:
            return None
        return (median(v), min(v), max(v), len(v))

    @property
    def needs_escalation(self) -> tuple:
        """(bool, reason) — §4's rule for going from 7 seeds to 21."""
        if len(self.converged) < self.n:
            return True, (f"{self.n - len(self.converged)} non-converged seed(s); "
                          "for an unstable cell the quantity of interest is the "
                          "failure rate, and 7 seeds cannot separate 15% from 40%")
        s = self.stat(THERMAL.key)
        if s and s[1] > 0 and s[2] / s[1] > UNSTABLE_RATIO:
            return True, (f"thermal MAE max/min = {s[2] / s[1]:.2f}x, above "
                          f"{UNSTABLE_RATIO}x")
        return False, ""


def _fmt(x: float, unit: str) -> str:
    if x is None:
        return "—"
    return f"{x:.4g}" if unit == "ppm" else f"{x:.3f}"


def _cellstat(cell: Cell, m: Metric) -> str:
    s = cell.stat(m.key)
    if s is None:
        return "—"
    med, lo, hi, n = s
    if n == 1:
        return f"{_fmt(med, m.unit)} (n=1)"
    return f"{_fmt(med, m.unit)} [{_fmt(lo, m.unit)}–{_fmt(hi, m.unit)}]"


# ── §3: the two bars ──────────────────────────────────────────────────────────
@dataclass
class Comparison:
    label_a: str
    label_b: str
    metric: Metric
    stat_a: tuple | None
    stat_b: tuple | None
    delta: float | None
    bar_a: bool | None       #: physical relevance
    bar_b: bool | None       #: seed separation; None = not evaluable
    verdict: str
    note: str = ""


def compare(a: Cell, b: Cell, m: Metric) -> Comparison:
    """Apply §3's two bars to one metric of one pair of cells.

    Both bars, or the difference is not reported as a difference. The four
    phrasings below are pre-approved in ANALYSIS_PLAN §3 and none of them is
    upgraded later.
    """
    sa, sb = a.stat(m.key), b.stat(m.key)
    if sa is None or sb is None:
        which = a.factors.variant if sa is None else b.factors.variant
        return Comparison(a.factors.label, b.factors.label, m, sa, sb, None,
                          None, None, "no converged seed",
                          f"{which} has no converged seed carrying {m.key}")

    delta = sa[0] - sb[0]
    bar_a = abs(delta) >= m.floor

    # Bar (b) needs actual ranges. One seed a side gives two points, and two
    # points are disjoint unless exactly equal — a verdict of "separable"
    # derived from no variance at all.
    if sa[3] < MIN_SEEDS_FOR_SEPARATION or sb[3] < MIN_SEEDS_FOR_SEPARATION:
        return Comparison(
            a.factors.label, b.factors.label, m, sa, sb, delta, bar_a, None,
            "insufficient seeds for a separation claim",
            f"n = {sa[3]} and {sb[3]} converged seeds; bar (b) needs "
            f"{MIN_SEEDS_FOR_SEPARATION}+ a side, §4 asks for {SEED_TARGET}. "
            f"Bar (a) alone: |delta| = {abs(delta):.4g} {m.unit} against a "
            f"{m.floor} {m.unit} floor.")

    bar_b = sa[2] < sb[1] or sb[2] < sa[1]
    if bar_a and bar_b:
        verdict = "difference worth reporting"
    elif bar_a and not bar_b:
        verdict = "suggestive, seeds overlap"
    elif bar_b and not bar_a:
        verdict = "statistically separable, operationally negligible"
    else:
        verdict = "no operationally meaningful difference"
    return Comparison(a.factors.label, b.factors.label, m, sa, sb, delta,
                      bar_a, bar_b, verdict)


def confound_check(a: Cell, b: Cell) -> tuple:
    """§1's mandatory control: comparable thermal error, or the gas comparison
    is confounded and counts toward H1 in neither direction."""
    sa, sb = a.stat(THERMAL.key), b.stat(THERMAL.key)
    if sa is None or sb is None:
        return None, "thermal MAE missing on one side"
    lo, hi = sorted((sa[0], sb[0]))
    if lo <= 0:
        return None, "a median thermal MAE is zero or negative"
    ratio = hi / lo
    if ratio > CONFOUND_RATIO:
        return False, (f"median thermal MAE differs {ratio:.2f}x "
                       f"({sa[0]:.3f} vs {sb[0]:.3f} degC), above the "
                       f"{CONFOUND_RATIO}x control")
    return True, f"median thermal MAE within {ratio:.2f}x"


# ── report ────────────────────────────────────────────────────────────────────
def build_report(runs: list, results_dir: Path) -> tuple:
    """Returns (markdown, n_integrity_problems)."""
    L: list = []
    problems: list = []
    w = L.append

    w("# C-11 matrix — aggregated results")
    w("")
    w(f"Source directory: `{results_dir}`  ")
    w(f"Generated by `scripts/aggregate_results.py` from "
      f"{len(runs)} `run.json` file(s).")
    w("")
    w("Every rule applied here is from ANALYSIS_PLAN, which was committed "
      "before the first matrix run: §3's two bars, §4's median and full "
      "min-max with non-converged seeds never dropped, §1's 2x thermal "
      "confound control. Gas percentages appear nowhere (C-9).")
    w("")

    # ── inventory ────────────────────────────────────────────────────────────
    smoke = [r for r in runs if r.status != "run"]
    prod = [r for r in runs if r.status == "run"]
    # A run on another training distribution is not a cell of this matrix and
    # is segregated before anything is grouped, not footnoted afterwards.
    off = [r for r in prod
           if r.distribution_hash != FROZEN_DISTRIBUTION_HASH]
    on = [r for r in prod
          if r.distribution_hash == FROZEN_DISTRIBUTION_HASH]
    unresolved = [r for r in on if r.cell is None]
    resolved = [r for r in on if r.cell is not None]

    w("## 0. Inventory and integrity")
    w("")
    w(f"- production runs (`status: run`): **{len(prod)}**")
    w(f"- smoke tests (`status: smoke_test`), excluded from every table below: "
      f"**{len(smoke)}**")
    w(f"- on the frozen distribution `{FROZEN_DISTRIBUTION_HASH}`: "
      f"**{len(on)}**; on another distribution, excluded: **{len(off)}**")
    w(f"- unresolved (cell unknown): **{len(unresolved)}**")
    w("")
    if off:
        w("**Runs on another training distribution — excluded from every "
          "table.** A model trained on a different distribution is not a cell "
          "of this matrix; comparing it with one is the train/test mismatch "
          "N-10 records, one level up.")
        w("")
        for r in off:
            w(f"- `{r.path.parent.name}` — distribution hash "
              f"`{r.distribution_hash}`")
        w("")
    if smoke:
        w("A smoke run has budgets overridden on the command line and is not a "
          "result. They are counted, not pooled.")
        w("")
    if unresolved:
        problems.append(f"{len(unresolved)} production run(s) could not be "
                        "resolved to a cell")
        w("**Unresolved runs — these are not in any table:**")
        w("")
        for r in unresolved:
            w(f"- `{r.path.parent.name}` — {r.unresolved_reason}")
        w("")

    w(f"Every run in the tables below carries distribution hash "
      f"`{FROZEN_DISTRIBUTION_HASH}`, by construction.")
    w("")
    commits = sorted({r.commit for r in resolved})
    if len(commits) > 1:
        w(f"Code commits present: {', '.join(commits)}. §1 lists them per "
          "cell: a cell whose seeds span commits is a sweep over the code as "
          "well as over the seed.")
        w("")

    # ── group into cells ─────────────────────────────────────────────────────
    by_variant: dict = {}
    for r in resolved:
        by_variant.setdefault(r.cell.variant, []).append(r)
    cells = {v: Cell(rs[0].cell, sorted(rs, key=lambda r: r.seed))
             for v, rs in by_variant.items()}

    for c in cells.values():
        tiers = {(r.tier, r.tier_source, r.n_cases) for r in c.runs}
        if len(tiers) > 1:
            problems.append(f"cell {c.factors.variant} mixes evaluation tiers: "
                            f"{sorted(tiers)}")
        seeds = [r.seed for r in c.runs]
        if len(set(seeds)) != len(seeds):
            problems.append(f"cell {c.factors.variant} has duplicate seeds: "
                            f"{sorted(seeds)}")
        # `harness.train` sets `converged = True` at exactly one place, together
        # with `stop_reason = "converged_plateau"`, so the two are one fact
        # recorded twice. If they disagree, one of them is wrong and the
        # convergence filter — which decides what enters a median — is reading
        # the wrong one (J-89 step 4: do the stopping rule and the definition
        # agree?).
        for r in c.runs:
            if r.converged != (r.stop_reason == CONVERGED_STOP_REASON):
                problems.append(
                    f"`{r.path.parent.name}`: converged={r.converged} but "
                    f"stop_reason={r.stop_reason!r}. The harness sets both at "
                    "one place, so they cannot disagree; until it is resolved "
                    "this run may be entering a median it does not belong in.")

    # ── coverage ─────────────────────────────────────────────────────────────
    w("## 1. Cell coverage")
    w("")
    w(f"§4 asks for {SEED_TARGET} seeds per cell, escalating to "
      f"{ESCALATION_TARGET} for any cell with a non-converged seed or a "
      f"max/min thermal MAE above {UNSTABLE_RATIO}x.")
    w("")
    w("| cell | label | arch | cascade | baseline | seeds | converged | "
      "target | commits |")
    w("|---|---|---|---|---|---|---|---|---|")
    expected = _expected_cells()
    for variant in sorted(expected, key=lambda v: (expected[v].architecture,
                                                   expected[v].cell_number)):
        f = expected[variant]
        c = cells.get(variant)
        if c is None:
            w(f"| {variant} | {f.label} | {f.architecture} | "
              f"{'in_cascade' if f.cascade else 'monolithic'} | "
              f"{'yes' if f.baseline else 'no'} | **0** | — | "
              f"{SEED_TARGET} | — |")
            continue
        esc, _ = c.needs_escalation
        target = ESCALATION_TARGET if esc else SEED_TARGET
        cm = ",".join(sorted({r.commit for r in c.runs}))
        w(f"| {variant} | {f.label} | {f.architecture} | "
          f"{'in_cascade' if f.cascade else 'monolithic'} | "
          f"{'yes' if f.baseline else 'no'} | {c.n} | {c.conv_rate} | "
          f"{target} | {cm} |")
    for variant, c in sorted(cells.items()):
        if variant not in expected:
            w(f"| {variant} | {c.factors.label} | {c.factors.architecture} | "
              f"{'in_cascade' if c.factors.cascade else 'monolithic'} | "
              f"{'yes' if c.factors.baseline else 'no'} | {c.n} | "
              f"{c.conv_rate} | {SEED_TARGET} | (not a declared cell) |")
    w("")
    missing = [v for v in expected if v not in cells]
    if missing:
        w(f"**{len(missing)} declared cell(s) have no production run:** "
          + ", ".join(f"`{v}`" for v in sorted(missing)) + ".")
        w("")
    w(_declared_quadrant_note(expected))
    w("")

    # ── per cell ─────────────────────────────────────────────────────────────
    w("## 2. Per cell — convergence rate first, error distribution second")
    w("")
    w("§4: *a cell with non-converged seeds is summarised by its convergence "
      "rate first and its error distribution second.* The distribution is over "
      "**converged seeds only** — README rule 5 forbids converting a "
      "non-converged model into a performance figure — and every seed, "
      "including every non-converged one, appears in the per-seed table with "
      "its `stop_reason`. Nothing is dropped; what a non-converged seed does "
      "not do is enter a median.")
    w("")
    for variant in sorted(cells):
        c = cells[variant]
        esc, why = c.needs_escalation
        w(f"### {variant} — {c.factors.label}")
        w("")
        w(f"**Converged {c.conv_rate} seeds.**"
          + (f"  \n**Escalate to {ESCALATION_TARGET} seeds:** {why}." if esc
             else ""))
        w("")
        w("| seed | stop_reason | converged | epochs | wall (s) | "
          + " | ".join(f"{m.key} ({m.unit})" for m in METRICS) + " |")
        w("|---|---|---|---|---|" + "---|" * len(METRICS))
        for r in c.runs:
            vals = []
            for m in METRICS:
                v = r.mae.get(m.key)
                s = "—" if v is None else _fmt(v, m.unit)
                # A non-converged seed's numbers are shown in brackets: present
                # in the record, marked as not a performance figure.
                vals.append(s if r.converged else f"[{s}]")
            w(f"| {r.seed} | {r.stop_reason} | "
              f"{'yes' if r.converged else '**no**'} | {r.epochs} | "
              f"{'' if r.wall_seconds is None else f'{r.wall_seconds:.0f}'} | "
              + " | ".join(vals) + " |")
        w("")
        if len(c.converged) == 0:
            w("*No converged seed: this cell has no error distribution. It is "
              "reported as non-converged, not as a performance number.*")
            w("")
            continue
        w("| metric | median | min | max | n converged |")
        w("|---|---|---|---|---|")
        for m in METRICS:
            s = c.stat(m.key)
            if s is None:
                w(f"| {m.key} ({m.unit}) | — | — | — | 0 |")
                continue
            w(f"| {m.key} ({m.unit}) | {_fmt(s[0], m.unit)} | "
              f"{_fmt(s[1], m.unit)} | {_fmt(s[2], m.unit)} | {s[3]} |")
        w("")

    # ── H1, primary ──────────────────────────────────────────────────────────
    w("## 3. H1 (PRIMARY) — the cascade, on end-of-rollout gas ppm")
    w("")
    w("ANALYSIS_PLAN Amendment 1: the primary metric is **gas concentration "
      "error in ppm at the end of the rollout**, scenarios K = 0.95 and "
      "K = 1.10, scored against IEC 60599. Four tests, one per architecture, "
      "the pair being `in_cascade` against `monolithic` **at the same baseline "
      "level**.")
    w("")
    w("Why this and not the 12 h window: gases obey "
      "`dc/dt = k_gen V_arr(theta) - k_dis c`, so they relax toward an "
      "equilibrium set by temperature. An in-cascade model reaches the correct "
      "one by construction; a monolithic model has no such constraint and a "
      "wrong equilibrium persists indefinitely. That is a structural claim "
      "about long-horizon behaviour, and 12 hours is too short for anything to "
      "accumulate.")
    w("")
    w("The value per seed is `|model - reference|` in ppm at the last window "
      "both rollouts cover. It comes from `<run>/rollout.json`, written by "
      "`24_rollout_thermal_error.py --json-out`; a run without that file "
      "contributes nothing here rather than a zero.")
    w("")
    n_primary = sum(1 for c in cells.values()
                    for m in ROLLOUT_METRICS if c.stat(m.key))
    # Counted over ALL seeds, converged or not, so "nobody ran the rollout" and
    # "the rollout ran but on seeds that did not converge" are distinguishable.
    # They call for different actions and the same sentence would hide it.
    n_scored = sum(1 for c in cells.values() for r in c.runs
                   if any(m.key in r.mae for m in ROLLOUT_METRICS))
    if n_primary == 0:
        if n_scored == 0:
            w("**No run in this directory has been scored on the primary "
              "metric.** Every cell needs `24_rollout_thermal_error.py "
              "--json-out <run>/rollout.json --k-scenarios 0.95 1.10`.")
        else:
            w(f"**{n_scored} run(s) carry a `rollout.json`, but none of them "
              "converged**, so none contributes a number here. A "
              "non-converged model is reported as non-converged, not "
              "converted into a performance figure (README rule 5) — and that "
              "applies to the primary metric exactly as it does to the rest.")
        w("")
        w("§4 below is the *secondary* comparison and does not substitute for "
          "this one: Amendment 1 demoted the 12 h gas MAE for a stated "
          "reason, and reading it as the primary result would undo that "
          "decision silently. **H1 is undecided.**")
        w("")
    else:
        for arch in _architectures(cells):
            for baseline in (False, True):
                a = _find(cells, arch, True, baseline, problems)
                b = _find(cells, arch, False, baseline, problems)
                lvl = "with baseline" if baseline else "no baseline"
                w(f"### {arch}, {lvl}")
                w("")
                if a is None or b is None:
                    have = [x.factors.variant for x in (a, b) if x]
                    w(f"*Pair incomplete — present: "
                      f"{', '.join(have) if have else 'neither cell'}. "
                      "No test.*")
                    w("")
                    continue
                ok, why = confound_check(a, b)
                w(f"- in-cascade: `{a.factors.variant}` "
                  f"({a.conv_rate} converged)")
                w(f"- monolithic: `{b.factors.variant}` "
                  f"({b.conv_rate} converged)")
                w(f"- §1 control: {why} → "
                  + ("**CONFOUNDED, counts toward H1 in neither direction**"
                     if ok is False else
                     "not evaluable" if ok is None else "control passed"))
                w("")
                # No literal pipes in the header text: `|model-ref|` would be
                # read as two extra column separators and the table would
                # render with its columns out of step with the separator row.
                w("| K | gas | in-cascade abs err ppm | monolithic abs err "
                  "ppm | delta | bar (a) | bar (b) | verdict |")
                w("|---|---|---|---|---|---|---|---|")
                for K in ROLLOUT_SCENARIOS:
                    for g in GASES:
                        m = next(x for x in ROLLOUT_METRICS
                                 if x.key == rollout_key(K, g.key))
                        cp = compare(a, b, m)
                        w(f"| {K:.2f} | {g.key} | {_cellstat(a, m)} | "
                          f"{_cellstat(b, m)} | "
                          f"{'—' if cp.delta is None else _fmt(cp.delta, 'ppm')}"
                          f" | {_bar(cp.bar_a)} | {_bar(cp.bar_b)} | "
                          f"{cp.verdict} |")
                w("")

    # ── H1, secondary ────────────────────────────────────────────────────────
    w("## 4. H1 (SECONDARY) — the same test on 12 h gas MAE")
    w("")
    w("Kept, demoted, with its reason attached so its negligibility reads as "
      "expected rather than as a finding: it is what the training loss targets, "
      "so a model bad at it will be bad at rollout, and it is the honest place "
      "to show that the short-horizon differences are real but operationally "
      "irrelevant. Amendment 1's own figures — MIONet `c_C2H2` MAE 8.6e-05 ppm "
      "against a 35 ppm IEC attention level — are why every architecture is "
      "pre-destined to return \"statistically separable, operationally "
      "negligible\" here.")
    w("")
    for arch in _architectures(cells):
        for baseline in (False, True):
            a = _find(cells, arch, True, baseline, problems)
            b = _find(cells, arch, False, baseline, problems)
            lvl = "with baseline" if baseline else "no baseline"
            w(f"### {arch}, {lvl}")
            w("")
            if a is None or b is None:
                have = [x.factors.variant for x in (a, b) if x]
                w(f"*Pair incomplete — present: "
                  f"{', '.join(have) if have else 'neither cell'}. No test.*")
                w("")
                continue
            ok, why = confound_check(a, b)
            w(f"- in-cascade: `{a.factors.variant}` ({a.conv_rate} converged)")
            w(f"- monolithic: `{b.factors.variant}` ({b.conv_rate} converged)")
            w(f"- §1 control: {why} → "
              + ("**CONFOUNDED, counts toward H1 in neither direction**"
                 if ok is False else
                 "not evaluable" if ok is None else "control passed"))
            w("")
            w("| gas | in-cascade median [min–max] | monolithic median "
              "[min–max] | delta | bar (a) | bar (b) | verdict |")
            w("|---|---|---|---|---|---|---|")
            for m in GASES:
                cp = compare(a, b, m)
                w(f"| {m.key} | {_cellstat(a, m)} | {_cellstat(b, m)} | "
                  f"{'—' if cp.delta is None else _fmt(cp.delta, m.unit)} | "
                  f"{_bar(cp.bar_a)} | {_bar(cp.bar_b)} | {cp.verdict} |")
            w("")
            notes = {compare(a, b, m).note for m in GASES} - {""}
            for n in sorted(notes):
                w(f"> {n}")
                w("")

    # ── S1 thermal ───────────────────────────────────────────────────────────
    w("## 5. S1 — thermal MAE, in-cascade against monolithic")
    w("")
    w("The §1 control reported in its own right. Both configurations predict "
      "`theta_TO` with the same network capacity, so this is not where the "
      "cascade acts; it is here to be read alongside every gas table above.")
    w("")
    w("| arch | baseline | in-cascade | monolithic | delta (degC) | bar (a) | "
      "bar (b) | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    n_pairs = 0
    for arch in _architectures(cells):
        for baseline in (False, True):
            a = _find(cells, arch, True, baseline, problems)
            b = _find(cells, arch, False, baseline, problems)
            if a is None or b is None:
                continue
            n_pairs += 1
            cp = compare(a, b, THERMAL)
            w(f"| {arch} | {'yes' if baseline else 'no'} | "
              f"{_cellstat(a, THERMAL)} | {_cellstat(b, THERMAL)} | "
              f"{'—' if cp.delta is None else _fmt(cp.delta, 'degC')} | "
              f"{_bar(cp.bar_a)} | {_bar(cp.bar_b)} | {cp.verdict} |")
    if n_pairs == 0:
        w("| — | — | — | — | — | — | — | no complete in-cascade/monolithic "
          "pair |")
    w("")

    # ── factorial ────────────────────────────────────────────────────────────
    w("## 6. The 2x2 factorial — cascade x analytic baseline")
    w("")
    w("Cell numbering is J-92's: 1 = monolithic with baseline, 2 = monolithic "
      "without, 3 = in-cascade with baseline, 4 = in-cascade without. A "
      "with-baseline cell of FNO, MIONet or S-DeepONet is a **hybrid** and is "
      "labelled as one; reporting it under the source paper's name would "
      "attribute our modification to their published method.")
    w("")
    w("Main effects are on the cell medians, pooled across the other factor's "
      "levels (Amendment 1). The interaction is "
      "`(cell3 - cell4) - (cell1 - cell2)`: how much more the cascade is worth "
      "**when the analytic baseline is present** than when it is not. All "
      "three need the full quadrant, so with a cell missing they read `n/i`, "
      "and the **simple effects** — one factor at one fixed level of the other "
      "— carry what a partial quadrant does support. Sign convention "
      "throughout is `with - without`, so a negative entry means the factor "
      "lowers error.")
    w("")
    for arch in _architectures(cells):
        w(f"### {arch}")
        w("")
        quad = {n: _find_cell_number(cells, arch, n, problems)
                for n in (1, 2, 3, 4)}
        present = [n for n, c in quad.items() if c is not None]
        w("| | no baseline | with baseline |")
        w("|---|---|---|")
        for casc, (nb, wb) in (("monolithic", (2, 1)), ("in_cascade", (4, 3))):
            row = []
            for n in (nb, wb):
                c = quad[n]
                row.append("— (cell %d absent)" % n if c is None
                           else f"{_cellstat(c, THERMAL)} · {c.conv_rate} conv")
            w(f"| **{casc}** | {row[0]} | {row[1]} |")
        w("")
        w(f"*thermal MAE, degC, median [min–max]. Cells present: "
          f"{', '.join(f'cell {n}' for n in present) or 'none'}.*")
        w("")
        w("| metric | main: cascade | main: baseline | interaction | "
          "simple: cascade &#124; baseline | simple: cascade &#124; no baseline "
          "| simple: baseline &#124; in-cascade | simple: baseline &#124; "
          "monolithic |")
        w("|---|---|---|---|---|---|---|---|")
        any_row = False
        for m in [THERMAL] + GASES:
            eff = _effects(quad, m)
            if eff is None:
                continue
            any_row = True
            w(f"| {m.key} ({m.unit}) | "
              + " | ".join(_eff(eff[k], m.unit) for k in
                           ("main_cascade", "main_baseline", "interaction",
                            "simple_cascade_with_baseline",
                            "simple_cascade_no_baseline",
                            "simple_baseline_in_cascade",
                            "simple_baseline_monolithic"))
              + " |")
        if not any_row:
            w("| — | — | — | — | — | — | — | — |")
        w("")
        w("*A negative entry means the factor lowers error. `n/i` = not "
          "identified from the cells present.*")
        w("")
        if len(present) < 4:
            w(f"**The {arch} quadrant is incomplete ({len(present)}/4), so "
              "neither main effect nor the interaction is identified.** A main "
              "effect is an average over the other factor's levels and the "
              "interaction is a difference of differences; both need all four "
              "cells. What a partial quadrant does support is the **simple "
              "effects** in the right-hand columns — the effect of one factor "
              "at one fixed level of the other — and those are a narrower "
              "claim, not a smaller version of the same one.")
            w("")

    # ── Amendment 2 ──────────────────────────────────────────────────────────
    w("## 7. Amendment 2 — the bounded-correction COD variant")
    w("")
    cod = cells.get("cod")
    bnd = cells.get("cod_bounded_correction")
    if cod is None or bnd is None:
        w(f"*Not reportable: `cod` {'present' if cod else 'absent'}, "
          f"`cod_bounded_correction` "
          f"{'present' if bnd else 'absent'}. The comparison is against COD on "
          "the same seeds.*")
        w("")
    else:
        shared = sorted({r.seed for r in cod.runs} & {r.seed for r in bnd.runs})
        w(f"Seeds shared by both cells: {shared}.")
        w("")
        w("| metric | COD | COD bounded | delta | bar (a) | bar (b) | verdict |")
        w("|---|---|---|---|---|---|---|")
        for m in METRICS:
            cp = compare(bnd, cod, m)
            w(f"| {m.key} ({m.unit}) | {_cellstat(cod, m)} | "
              f"{_cellstat(bnd, m)} | "
              f"{'—' if cp.delta is None else _fmt(cp.delta, m.unit)} | "
              f"{_bar(cp.bar_a)} | {_bar(cp.bar_b)} | {cp.verdict} |")
        w("")
        w("Amendment 2: if the accuracy cost is negligible the paper gets a "
          "structural guarantee instead of a measurement; **if it costs real "
          "accuracy, that trade is itself the finding** and is reported as "
          "such.")
        w("")

    # ── what is not here ─────────────────────────────────────────────────────
    w("## 8. What this report does not contain, and why")
    w("")
    w("1. **Nothing is substituted for a metric a run has not been scored "
      "on.** A run without `rollout.json` contributes nothing to §3 rather "
      "than a zero, and §4 is never promoted in its place — Amendment 1 "
      "demoted the 12 h gas MAE for a stated reason, and reading it as the "
      "primary result would undo that decision silently.")
    w("2. **Gas NMAE appears nowhere** (C-9, tightened 2026-08-02). Median 12 h "
      "gas variation is 0.001%-0.046% of each gas's engineering threshold, so "
      "a gas percentage is a ratio to a physically empty quantity. `theta_TO` "
      "NMAE is legitimate but is secondary and is left to the per-run "
      "`run.json`.")
    w("3. **S3's swing and Jensen-gap tables are not here.** "
      "`18_swing_fidelity.py` writes markdown only; it has no `--json-out`, so "
      "there is no machine-readable per-cell swing ratio to aggregate. The "
      "C-11 honesty protocol requires those two tables to be read *with* "
      "thermal MAE, so a matrix report is incomplete until they are "
      "aggregable.")
    w("4. **S8, the post-hoc cascade, is not here.** It needs each monolithic "
      "cell's predicted `theta_TO` pushed through the quadrature offline — "
      "`predictions.npz` carries what it needs, but the computation is a "
      "separate script, not an aggregation.")
    w("")

    if problems:
        w("## Integrity problems")
        w("")
        for p in problems:
            w(f"- {p}")
        w("")

    return "\n".join(L), len(problems)


def _bar(x) -> str:
    return "—" if x is None else ("yes" if x else "no")


def _architectures(cells: dict) -> list:
    return sorted({c.factors.architecture for c in cells.values()})


def _find(cells: dict, arch: str, cascade: bool, baseline: bool,
          problems: list | None = None):
    """The one cell in a factorial slot, or None.

    Two cells in one slot is an integrity problem, not a tie to break: pooling
    them would average two different experiments and picking one would be a
    silent choice. It is recorded and the slot is left empty.
    """
    hits = [c for c in cells.values()
            if c.factors.architecture == arch and c.factors.cascade == cascade
            and c.factors.baseline == baseline and c.factors.in_factorial]
    if len(hits) > 1:
        msg = (f"{len(hits)} cells claim the same factorial slot for {arch} "
               f"(cascade={cascade}, baseline={baseline}): "
               f"{sorted(h.factors.variant for h in hits)}")
        if problems is not None and msg not in problems:
            problems.append(msg)
        return None
    return hits[0] if hits else None


def _find_cell_number(cells: dict, arch: str, n: int,
                      problems: list | None = None):
    hits = [c for c in cells.values()
            if c.factors.architecture == arch and c.factors.cell_number == n
            and c.factors.in_factorial]
    if len(hits) > 1:
        msg = (f"{len(hits)} cells claim {arch} factorial cell {n}: "
               f"{sorted(h.factors.variant for h in hits)}")
        if problems is not None and msg not in problems:
            problems.append(msg)
        return None
    return hits[0] if hits else None


def _effects(quad: dict, m) -> dict | None:
    """Factorial effects on the cell medians, or None if no cell has the metric.

    A **main effect** is an average over the other factor's levels and an
    **interaction** is a difference of differences; both need all four cells,
    so with a cell missing they are `None` — "not identified" — rather than a
    number computed from whatever happened to be there. What a partial quadrant
    does support is a **simple effect**: one factor at one fixed level of the
    other. That is a narrower claim and is reported under its own name, because
    calling a simple effect a main effect is how a two-cell comparison comes to
    be quoted as a factorial result.

    Sign convention throughout: `with - without`, so a negative entry means the
    factor lowers error.
    """
    med = {}
    for n, c in quad.items():
        s = None if c is None else c.stat(m.key)
        med[n] = None if s is None else s[0]
    if not any(v is not None for v in med.values()):
        return None

    def diff(a, b):
        return None if med[a] is None or med[b] is None else med[a] - med[b]

    complete = all(v is not None for v in med.values())
    return {
        # cell 1 mono+baseline, 2 mono, 3 cascade+baseline, 4 cascade
        "main_cascade": (((med[3] + med[4]) - (med[1] + med[2])) / 2
                         if complete else None),
        "main_baseline": (((med[1] + med[3]) - (med[2] + med[4])) / 2
                          if complete else None),
        "interaction": ((med[3] - med[4]) - (med[1] - med[2])
                        if complete else None),
        "simple_cascade_with_baseline": diff(3, 1),
        "simple_cascade_no_baseline": diff(4, 2),
        "simple_baseline_in_cascade": diff(3, 4),
        "simple_baseline_monolithic": diff(1, 2),
        "complete": complete,
        "n_cells": sum(1 for v in med.values() if v is not None),
    }


def _eff(x, unit: str) -> str:
    return "n/i" if x is None else _fmt(x, unit)


def _expected_cells() -> dict:
    """Every cell the matrix declares: a config on the frozen distribution.

    Membership is the distribution hash, not a list of names. A config on any
    other distribution — `configs/v57_faithful.yaml` reproduces the v57 sampler
    — is not a cell of this matrix, because a run on a different training
    distribution cannot be compared with one that is not.
    """
    out = {}
    for p in sorted(ROOT.glob("configs/**/*.yaml")):
        try:
            cfg = load_config(p)
            if cfg.distribution_hash != FROZEN_DISTRIBUTION_HASH:
                continue
            f = cell_factors(cfg.raw)
        except Exception:
            continue
        out[f.variant] = f
    return out


def _declared_quadrant_note(expected: dict) -> str:
    """Is any architecture's 2x2 incomplete in the CONFIG SET, not the results?

    Derived, not written down. This began as a hardcoded paragraph about
    PI-DeepONet having no cell 2 — true when it was written, false the moment
    the config was added, and a report that states a closed hole on every run is
    worse than one that states nothing. Deriving it means a future architecture
    added with three of its four cells is caught the same way, and means this
    note cannot outlive the condition it describes.

    A missing *config* is a different and worse thing than a missing *run*: no
    amount of compute fixes it, and it is what makes a main effect
    unidentifiable no matter how many seeds finish.
    """
    by_arch: dict = {}
    for f in expected.values():
        if f.in_factorial:
            by_arch.setdefault(f.architecture, set()).add(f.cell_number)
    holes = {a: sorted({1, 2, 3, 4} - ns) for a, ns in by_arch.items()
             if len(ns) < 4}
    if not holes:
        return ("Every architecture has all four factorial cells declared as a "
                "config, so no quadrant is incomplete for want of a cell to "
                "run. What is missing below is runs, which compute fixes.")
    parts = ["**A hole in the declared set, not in the results.** No amount of "
             "compute closes this one: an architecture missing a *config* "
             "cannot have that cell run at all, and a partial quadrant leaves "
             "both main effects and the interaction unidentifiable."]
    for a, missing in sorted(holes.items()):
        parts.append(f"**{a}** is missing factorial cell(s) "
                     + ", ".join(str(n) for n in missing)
                     + f" (has {sorted(by_arch[a])}).")
    return "  \n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="Directory to search recursively for run.json.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown output path. Required in a sweep; without "
                         "it the report goes to stdout only.")
    ap.add_argument("--include-smoke", action="store_true",
                    help="Aggregate smoke runs too. For testing this script "
                         "itself; a smoke run is not a result.")
    args = ap.parse_args()

    files = sorted(args.results.rglob("run.json"))
    if not files:
        raise SystemExit(f"[aggregate] no run.json under {args.results}")
    reg = _registry()
    # One unreadable run.json must not take the other 104 with it — a truncated
    # write on a Drive directory is a thing that happens, and losing the whole
    # aggregation to it would push someone toward reading the tables by hand.
    # It is still an error: the file is named, and the exit code is non-zero.
    runs, broken = [], []
    for p in files:
        try:
            runs.append(read_run(p, reg))
        except Exception as exc:
            broken.append(f"{p}: {type(exc).__name__}: {exc}")
    if args.include_smoke:
        for r in runs:
            r.status = "run"

    md, n_problems = build_report(runs, args.results)
    if broken:
        n_problems += len(broken)
        md += ("\n- " + str(len(broken)) + " `run.json` file(s) could not be "
               "read at all:\n"
               + "\n".join(f"  - `{b}`" for b in broken) + "\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"[aggregate] wrote {args.out} ({len(md.splitlines())} lines)")
    else:
        print(md)

    # C-9 is a rule about the output, so it is checked on the output. A gas
    # percentage must not be constructible from this document.
    for gas in (m.key for m in GASES):
        for line in md.splitlines():
            if gas in line and "%" in line and "0.001%-0.046%" not in line:
                print(f"[aggregate] C-9 VIOLATION: a gas percentage reached the "
                      f"report: {line.strip()[:120]}")
                n_problems += 1

    print(f"[aggregate] {len(runs)} run(s); "
          f"{sum(1 for r in runs if r.status == 'run')} production; "
          f"{n_problems} integrity problem(s)")
    return 1 if n_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
