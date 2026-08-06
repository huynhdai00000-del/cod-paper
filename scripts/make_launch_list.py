#!/usr/bin/env python3
"""The flat launch list: every (cell, seed) still to run, dealt across accounts.

    python scripts/make_launch_list.py --accounts 20 \
        --out audit_port/LAUNCH_LIST.md

Three inputs, none of them typed by hand:

  * **the cells** — every config under `configs/` on the frozen distribution
    hash, so a cell added to `configs/matrix/` appears here without this file
    being edited, and `configs/v57_faithful.yaml` stays out because it is on
    the v57 sampler;
  * **the seeds** — `1..SEED_TARGET` from ANALYSIS_PLAN §4;
  * **what is already done** — `audit_port/reusable_seeds.json`, written by
    `38_reusable_seed_audit.py`, which decides reusability against five stated
    conditions rather than by recollection.

**How the deal is arranged, and why it is not by architecture.** Each account
gets runs from as many *different* cells as possible at as few different seeds
as possible. Losing one account then costs one seed from several cells, which
leaves every cell reportable at six seeds; the alternative — one account owning
whole cells — loses a cell entirely, and a missing cell is what makes a main
effect unidentifiable (PORT_LOG J-95). The lines are otherwise interchangeable:
this is a flat list, not a priority order.

**Re-running a line is safe.** `run.py` refuses to overwrite an existing run
directory, so a repeated command errors instead of clobbering a finished result.
A line interrupted halfway can simply be run again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cod.cells import cell_factors  # noqa: E402
from cod.config import load_config  # noqa: E402

FROZEN = "fc4cb76c3b32ec17"
SEED_TARGET = 7                     # ANALYSIS_PLAN §4
WALL_SECONDS = 10800                # C-11, and it does not bind (runbook §0)
DEFAULT_OUT_DIR = "/content/drive/MyDrive/cod_matrix"

#: Measured T4 wall-clock per run, seconds, for balancing the deal. Only the
#: three that have actually run on a GPU are here; the rest fall back to
#: `DEFAULT_COST`. These are for *balancing lines*, not for scheduling — the
#: lesson of ANALYSIS_PLAN §9 is that a cost ranking that has not been measured
#: on the target device is wrong, sometimes in direction.
MEASURED_COST = {
    "cod": 4911,              # O-5
    "cod_no_baseline": 2835,  # O-12
    "fno_in_cascade": 1546,   # artifacts/fno2, post clamp fix
    "mionet_in_cascade": 187,      # ANALYSIS_PLAN §9
    "sdeeponet_in_cascade": 291,   # ANALYSIS_PLAN §9
}
#: For a cell nobody has run on a GPU. Deliberately conservative — roughly COD's
#: cost — so a line's estimate is an upper bound rather than a hope. The two
#: cheapest measured cells came in at 187 s and 291 s, an order below this, so
#: lines dominated by MIONet or S-DeepONet will finish far sooner than their
#: estimate says. Estimating them optimistically instead would risk a line that
#: does not fit in a session, which is the expensive direction to be wrong in.
DEFAULT_COST = 3000


def cells() -> dict:
    """variant -> (config path relative to repo root, CellFactors)."""
    out = {}
    for p in sorted(ROOT.glob("configs/**/*.yaml")):
        try:
            cfg = load_config(p)
            if cfg.distribution_hash != FROZEN:
                continue
            f = cell_factors(cfg.raw)
        except Exception:
            continue
        out[f.variant] = (p.relative_to(ROOT).as_posix(), f)
    return out


def done_pairs(path: Path) -> set:
    if not path.is_file():
        print(f"[launch] {path} not found — treating every seed as still to "
              "run. Run audit_port/scripts/38_reusable_seed_audit.py first if "
              "some are already done.")
        return set()
    j = json.loads(path.read_text(encoding="utf-8"))
    return {(r["variant"], int(r["seed"])) for r in j.get("reusable", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=SEED_TARGET)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Drive directory every account writes into.")
    ap.add_argument("--reusable",
                    type=Path, default=ROOT / "audit_port" / "reusable_seeds.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "audit_port" / "LAUNCH_LIST.md")
    args = ap.parse_args()

    cl = cells()
    done = done_pairs(args.reusable)

    # Seed-major order, so consecutive entries are different cells and the
    # round-robin below spreads each account across cells rather than seeds.
    todo = [(v, s) for s in range(1, args.seeds + 1) for v in sorted(cl)
            if (v, s) not in done]

    lanes = [[] for _ in range(args.accounts)]
    load = [0] * args.accounts
    for i, (v, s) in enumerate(todo):
        k = i % args.accounts
        lanes[k].append((v, s))
        load[k] += MEASURED_COST.get(v, DEFAULT_COST)

    md = ["# Launch list", "",
          f"{len(todo)} runs across {len(cl)} cells x seeds 1-{args.seeds}, "
          f"dealt into {args.accounts} lines. "
          f"{len(done)} already done and subtracted.", "",
          "Generated by `scripts/make_launch_list.py`; the cells come from "
          "`configs/` filtered to distribution hash "
          f"`{FROZEN}`, and the already-done set from "
          "`audit_port/reusable_seeds.json`.", "",
          "One line per account. The lines are interchangeable — this is a "
          "flat list, not a priority order. Re-running a line is safe: "
          "`run.py` refuses to overwrite an existing run directory, so a "
          "repeated command errors rather than clobbering a finished result, "
          "and an interrupted line can just be run again.", ""]
    if done:
        md += ["**Already done, do not repeat:**", ""]
        for v, s in sorted(done):
            md.append(f"- `{v}` seed {s}")
        md.append("")
    md += ["**The GPU-hour column is an upper bound.** Five cells have a "
           "measured T4 time; every other cell is costed at a conservative "
           f"{DEFAULT_COST} s, near COD's. The two cheapest measured cells "
           "took 187 s and 291 s, an order of magnitude below that, so lines "
           "carrying MIONet or S-DeepONet will finish well inside their "
           "estimate. If a line still looks too long for one session, "
           "regenerate with more lines: `--accounts 30`.", "",
           "| line | runs | est. GPU h | cells |", "|---|---|---|---|"]
    for i, lane in enumerate(lanes, 1):
        cs = sorted({v for v, _ in lane})
        md.append(f"| {i} | {len(lane)} | {load[i - 1] / 3600:.1f} | "
                  + ", ".join(cs) + " |")
    md += ["", "## The commands", ""]
    for i, lane in enumerate(lanes, 1):
        md += [f"### Line {i} — {len(lane)} runs, "
               f"~{load[i - 1] / 3600:.1f} GPU h", "", "```bash"]
        for v, s in lane:
            cfg_path, _f = cl[v]
            md.append(f"python scripts/run.py --config {cfg_path} \\")
            md.append(f"    --freeze-hash {FROZEN} "
                      f"--max-wall-seconds {WALL_SECONDS} \\")
            md.append(f"    --seed {s} --tag s{s} --out {args.out_dir}")
        md += ["```", ""]

    md += ["## After the runs", "",
           "Per run, before anything is aggregated — the primary metric is not "
           "in `run.json` and nothing else produces it:", "", "```bash",
           "python audit_port/scripts/24_rollout_thermal_error.py \\",
           "    --checkpoint <run>/model.pt --max-windows 730 \\",
           "    --k-scenarios 0.95 1.10 --json-out <run>/rollout.json \\",
           "    --out <run>/rollout.md",
           "python audit_port/scripts/18_swing_fidelity.py \\",
           "    --checkpoint <run>/model.pt --out <run>/swing_fidelity.md",
           "```", "",
           "Then, over the whole Drive directory:", "", "```bash",
           "python scripts/aggregate_results.py \\",
           "    --results /path/to/cod_matrix --out audit_port/MATRIX_RESULTS.md",
           "```", "",
           "Read its exit code: non-zero means an integrity problem that makes "
           "the tables unsafe to quote, and the problems are listed at the end "
           "of the report.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[launch] {len(cl)} cells, seeds 1-{args.seeds}, "
          f"{len(done)} done, {len(todo)} to run")
    print(f"[launch] dealt into {args.accounts} lines of "
          f"{min(len(x) for x in lanes)}-{max(len(x) for x in lanes)} runs, "
          f"{min(load) / 3600:.1f}-{max(load) / 3600:.1f} GPU h")
    print(f"[launch] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
