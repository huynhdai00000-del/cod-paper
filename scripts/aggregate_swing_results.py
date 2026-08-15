#!/usr/bin/env python3
"""Aggregate per-run swing/Jensen JSON without hiding seed instability.

Usage:
    python scripts/aggregate_swing_results.py \
        --results /path/to/cod_matrix --out audit_port/SWING_MATRIX_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


STATES = ("c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2", "DP")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells: dict[str, list[dict]] = defaultdict(list)
    missing: list[str] = []
    problems: list[str] = []
    n_converged = 0

    for run_path in sorted(args.results.glob("*/run.json")):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if not bool(run.get("outcome", {}).get("converged", False)):
            continue
        n_converged += 1
        swing_path = run_path.parent / "swing_fidelity.json"
        if not swing_path.is_file():
            missing.append(run_path.parent.name)
            continue
        try:
            swing = json.loads(swing_path.read_text(encoding="utf-8"))
            model_rows = swing["models"]
            if len(model_rows) != 1:
                raise ValueError(f"expected one model, found {len(model_rows)}")
            model = model_rows[0]
            evaluation = swing["evaluation"]
        except Exception as exc:
            problems.append(f"{run_path.parent.name}: {exc}")
            continue
        if evaluation.get("distribution_hash") != "fc4cb76c3b32ec17":
            problems.append(
                f"{run_path.parent.name}: swing distribution hash is "
                f"{evaluation.get('distribution_hash')!r}")
            continue
        cell = run.get("cell", {})
        variant = str(cell.get("variant", "<unknown>"))
        cells[variant].append({
            "seed": int(run.get("seed", -1)),
            "cell": cell,
            "model": model,
        })

    lines: list[str] = []
    A = lines.append
    A("# Swing fidelity and Jensen-gap matrix")
    A("")
    A(f"Source: `{args.results}`. Converged checkpoints: **{n_converged}**; "
      f"scored: **{sum(map(len, cells.values()))}**; missing: "
      f"**{len(missing)}**; malformed: **{len(problems)}**.")
    A("")
    A("All summaries are median and full min–max over seeds. Gate counts are "
      "reported explicitly; a failed seed is never dropped. `n/e` means the "
      "model tracked no swing band at median thermal MAE <= 5 degC, so the "
      "spectral-bias gate was not evaluable.")
    A("")
    A("## 1. Cell summary")
    A("")
    A("| cell | architecture | cascade | baseline | scored | gate "
      "(pass/fail/n/e) | median swing ratio [min–max] | median thermal MAE "
      "degC [min–max] |")
    A("|---|---|---|---|---|---|---|---|")
    for variant, rows in sorted(cells.items()):
        cell = rows[0]["cell"]
        ratios = [r["model"]["median_swing_ratio"] for r in rows]
        maes = [r["model"]["median_thermal_mae_degC"] for r in rows]
        gates = Counter(r["model"]["gate"] for r in rows)
        A(f"| `{variant}` | {cell.get('architecture', '?')} | "
          f"{cell.get('cascade', '?')} | {cell.get('baseline', '?')} | "
          f"{len(rows)} | {gates['pass']}/{gates['fail']}/"
          f"{gates['not_evaluated']} | {fmt(median(ratios))} "
          f"[{fmt(min(ratios))}–{fmt(max(ratios))}] | {fmt(median(maes))} "
          f"[{fmt(min(maes))}–{fmt(max(maes))}] |")

    A("")
    A("## 2. Jensen-gap preservation")
    A("")
    A("Median of each seed's median predicted/true Jensen-gap ratio over live "
      "cases. One is exact preservation; below one means gap lost.")
    A("")
    A("| cell | " + " | ".join(f"`{s}`" for s in STATES) + " |")
    A("|---|" + "---|" * len(STATES))
    for variant, rows in sorted(cells.items()):
        values = []
        for state in STATES:
            xs = [r["model"]["jensen_gap"][state]["median_ratio"]
                  for r in rows]
            values.append(f"{fmt(median(xs))} [{fmt(min(xs))}–{fmt(max(xs))}]")
        A(f"| `{variant}` | " + " | ".join(values) + " |")

    A("")
    A("## 3. Gate failures by tracked swing band")
    A("")
    A("| cell | band (degC) | scored seeds | tracked seeds | failed seeds | "
      "median ratio [min–max], tracked seeds |")
    A("|---|---|---|---|---|---|")
    for variant, rows in sorted(cells.items()):
        bands: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in rows:
            for band in row["model"]["bands"]:
                bands[(band["lo_degC"], band["hi_degC"])].append(band)
        for (lo, hi), vals in sorted(bands.items()):
            tracked = [v for v in vals if v["tracked"]]
            failed = [v for v in tracked if v["failed"]]
            ratio = (f"{fmt(median([v['median_ratio'] for v in tracked]))} "
                     f"[{fmt(min(v['median_ratio'] for v in tracked))}–"
                     f"{fmt(max(v['median_ratio'] for v in tracked))}]"
                     if tracked else "n/e")
            A(f"| `{variant}` | {lo}–{hi} | {len(vals)} | {len(tracked)} | "
              f"{len(failed)} | {ratio} |")

    A("")
    A("## 4. Integrity")
    A("")
    if not missing and not problems:
        A("**PASS** — every converged checkpoint has one readable swing JSON on "
          "the frozen distribution.")
    else:
        for name in missing:
            A(f"- missing: `{name}/swing_fidelity.json`")
        for problem in problems:
            A(f"- malformed: {problem}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate-swing] wrote {args.out} ({len(lines)} lines)")
    print(f"[aggregate-swing] {n_converged} converged; "
          f"{sum(map(len, cells.values()))} scored; "
          f"{len(missing) + len(problems)} integrity problem(s)")
    return 1 if missing or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
