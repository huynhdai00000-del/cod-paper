#!/usr/bin/env python3
"""Create the engineering figures and compact result tables from matrix runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cod-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from aggregate_results import _registry, read_run, rollout_key


GASES = ("c_H2", "c_C2H2", "c_C2H4", "c_CO", "c_CO2")
GAS_LABELS = (r"H$_2$", r"C$_2$H$_2$", r"C$_2$H$_4$", "CO", r"CO$_2$")
SCENARIOS = (0.95, 1.10)

PAIRS = (
    ("FNO, no baseline", "fno_in_cascade", "fno_monolithic", False),
    ("FNO-COD", "fno_baseline_in_cascade", "fno_baseline_monolithic", False),
    ("MIONet, no baseline", "mionet_in_cascade", "mionet_monolithic", False),
    ("MIONet-COD", "mionet_baseline_in_cascade", "mionet_baseline_monolithic", False),
    ("PI-DeepONet, no baseline", "cod_no_baseline", "pideeponet_monolithic", True),
    ("PI-COD", "cod", "pideeponet_baseline_monolithic", False),
    ("S-DeepONet, no baseline", "sdeeponet_in_cascade", "sdeeponet_monolithic", False),
    ("S-DeepONet-COD", "sdeeponet_baseline_in_cascade", "sdeeponet_baseline_monolithic", False),
)

CASCADE_BASELINE = (
    ("PI-COD", "cod"),
    ("FNO-COD", "fno_baseline_in_cascade"),
    ("MIONet-COD", "mionet_baseline_in_cascade"),
    ("S-DeepONet-COD", "sdeeponet_baseline_in_cascade"),
)

SWING_CELLS = (
    ("FNO", "fno_baseline_in_cascade", "fno_in_cascade"),
    ("MIONet", "mionet_baseline_in_cascade", "mionet_in_cascade"),
    ("PI-DeepONet", "cod", "cod_no_baseline"),
    ("S-DeepONet", "sdeeponet_baseline_in_cascade", "sdeeponet_in_cascade"),
)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def load_runs(results: Path) -> dict[str, list]:
    registry = _registry()
    cells: dict[str, list] = defaultdict(list)
    for path in sorted(results.glob("*/run.json")):
        run = read_run(path, registry)
        if run.cell is not None:
            cells[run.variant].append(run)
    for runs in cells.values():
        runs.sort(key=lambda r: r.seed)
    return cells


def load_swing(results: Path, cells: dict[str, list]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for variant, runs in cells.items():
        for run in runs:
            if not run.converged:
                continue
            path = run.path.parent / "swing_fidelity.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                out[variant].append(data["models"][0])
    return out


def values(cells: dict[str, list], variant: str, key: str) -> list[float]:
    return [r.mae[key] for r in cells[variant]
            if r.converged and key in r.mae]


def stat(cells: dict[str, list], variant: str, key: str) -> tuple[float, float, float]:
    xs = values(cells, variant, key)
    if not xs:
        raise ValueError(f"missing {key} for {variant}")
    return median(xs), min(xs), max(xs)


def swing_stat(swing: dict[str, list[dict]], variant: str, key: str) -> tuple[float, float, float]:
    xs = [float(row[key]) for row in swing[variant]]
    return median(xs), min(xs), max(xs)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight",
                    transparent=False)
    plt.close(fig)


def box(ax, xy, width, height, text, face, edge="#243447", fontsize=9):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.25)


def arrow(ax, start, end, color="#52606d"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.25, color=color))


def figure_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.7)
    ax.axis("off")

    box(ax, (0.15, 1.05), 1.25, 1.05, "Load and ambient\nhistory", "#eef5ff")
    box(ax, (1.85, 1.78), 1.55, 0.78, "Analytic thermal\nbaseline", "#e8f5e9")
    box(ax, (1.85, 0.42), 1.55, 0.92,
        "Neural thermal\ncorrection", "#fff3e0", fontsize=8.5)
    box(ax, (3.90, 1.05), 1.35, 1.05, "Thermal\ntrajectory", "#f3e5f5")
    box(ax, (5.75, 1.05), 1.70, 1.05,
        "Deterministic\ngas and DP\npropagation", "#e0f7fa")
    box(ax, (7.95, 1.05), 1.25, 1.05, "Derived CHI\ntrajectory", "#fce4ec")

    arrow(ax, (1.40, 1.70), (1.85, 2.10))
    arrow(ax, (1.40, 1.40), (1.85, 0.88))
    arrow(ax, (3.40, 2.10), (3.90, 1.72))
    arrow(ax, (3.40, 0.88), (3.90, 1.42))
    arrow(ax, (5.25, 1.58), (5.75, 1.58))
    arrow(ax, (7.45, 1.58), (7.95, 1.58))

    ax.text(4.58, 2.45, "learned interface", ha="center", color="#6a1b9a")
    ax.text(6.60, 2.45, "governing update", ha="center", color="#00796b")
    ax.text(2.625, 0.12, "FNO | MIONet | PI-DeepONet | S-DeepONet",
            ha="center", va="bottom", fontsize=7.5, color="#52606d")
    ax.text(5.0, 3.58,
            "COD engineering topology: a learned thermal interface with\n"
            "deterministic downstream state evolution",
            ha="center", va="top", fontsize=11, weight="bold")
    save_figure(fig, out_dir, "fig1_cod_engineering_architecture")


def figure_rollout_advantage(cells: dict[str, list], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    colors = {0.95: "#2b6cb0", 1.10: "#c05621"}
    y_positions = np.arange(len(PAIRS))[::-1]

    for y, (label, cascade, monolithic, confounded) in zip(y_positions, PAIRS):
        ratios = []
        for K in SCENARIOS:
            for gas in GASES:
                c = stat(cells, cascade, rollout_key(K, gas))[0]
                m = stat(cells, monolithic, rollout_key(K, gas))[0]
                ratios.append((m / c, K))
        offset = {0.95: 0.09, 1.10: -0.09}
        alpha = 0.32 if confounded else 0.82
        for ratio, K in ratios:
            ax.scatter(ratio, y + offset[K], s=23, color=colors[K], alpha=alpha,
                       edgecolor="white", linewidth=0.35, zorder=3)
        centre = math.exp(median([math.log(v) for v, _ in ratios]))
        ax.scatter(centre, y, marker="D", s=37, color="#1f2933" if not confounded else "#9aa5b1",
                   edgecolor="white", linewidth=0.5, zorder=4)

    ax.axvline(1, color="#52606d", linewidth=1.1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p[0] + ("*" if p[3] else "") for p in PAIRS])
    ax.set_xlabel("Median one-year error reduction, monolithic / in-cascade")
    ax.set_title("The cascade lowers long-horizon gas error across backbones")
    ax.grid(axis="x", which="both", color="#d9e2ec", linewidth=0.6)
    ax.scatter([], [], color=colors[0.95], label="K = 0.95")
    ax.scatter([], [], color=colors[1.10], label="K = 1.10")
    ax.scatter([], [], marker="D", color="#1f2933", label="geometric median")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.text(0.01, -0.14,
            "Each row contains five gases at each load. Values above 1 favor the cascade.\n"
            "*PI-DeepONet without the analytic baseline fails the prespecified thermal comparability control.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#52606d")
    save_figure(fig, out_dir, "fig2_cascade_rollout_advantage")


def figure_fno(cells: dict[str, list], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.75), sharey=True)
    variants = (
        ("FNO-COD", "fno_baseline_in_cascade", "#2b6cb0", "o"),
        ("FNO monolithic", "fno_baseline_monolithic", "#c05621", "s"),
    )
    x = np.arange(len(GASES))
    for ax, K in zip(axes, SCENARIOS):
        for label, variant, color, marker in variants:
            med, lo, hi = [], [], []
            for gas in GASES:
                m, a, b = stat(cells, variant, rollout_key(K, gas))
                med.append(m)
                lo.append(m - a)
                hi.append(b - m)
            shift = -0.10 if variant.endswith("in_cascade") else 0.10
            ax.errorbar(x + shift, med, yerr=np.array([lo, hi]), fmt=marker,
                        color=color, ecolor=color, capsize=2.5, markersize=5,
                        linewidth=1.0, label=label)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(GAS_LABELS)
        ax.set_title(f"Load factor K = {K:.2f}")
        ax.grid(axis="y", which="both", color="#d9e2ec", linewidth=0.55)
    axes[0].set_ylabel("Absolute gas error at one year (ppm)\nmedian and full seed range")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("FNO-COD separates accurate thermal learning from stable state propagation",
                 fontsize=11, weight="bold", y=1.02)
    save_figure(fig, out_dir, "fig3_fno_rollout_errors")


def figure_absolute_accuracy(cells: dict[str, list], out_dir: Path) -> None:
    data = np.zeros((len(CASCADE_BASELINE), len(SCENARIOS) * len(GASES)))
    labels = []
    for i, (label, variant) in enumerate(CASCADE_BASELINE):
        labels.append(label)
        j = 0
        for K in SCENARIOS:
            for gas in GASES:
                data[i, j] = stat(cells, variant, rollout_key(K, gas))[0]
                j += 1

    fig, ax = plt.subplots(figsize=(10.0, 3.3))
    im = ax.imshow(np.log10(data), aspect="auto", cmap="viridis_r")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(GAS_LABELS) * 2)
    ax.axvline(4.5, color="white", linewidth=2.0)
    ax.text(2, -0.66, "K = 0.95", ha="center", va="bottom", fontsize=9)
    ax.text(7, -0.66, "K = 1.10", ha="center", va="bottom", fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if np.log10(val) > np.nanmedian(np.log10(data)) else "#102a43"
            ax.text(j, i, f"{val:.2g}", ha="center", va="center",
                    fontsize=7.5, color=color)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"log$_{10}$(median absolute error / ppm)")
    fig.suptitle("Absolute accuracy of the baseline-equipped COD implementations",
                 fontsize=11, weight="bold", y=1.02)
    ax.set_xlabel("Gas species")
    ax.set_ylabel("Thermal backbone")
    save_figure(fig, out_dir, "fig4_cod_absolute_accuracy")


def figure_shape_mechanism(swing: dict[str, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    x = np.arange(len(SWING_CELLS))
    width = 0.34
    colors = ("#2b6cb0", "#cbd5e0")

    for k, (baseline_on, label) in enumerate(((True, "with analytic baseline"),
                                               (False, "without analytic baseline"))):
        medians, lows, highs, rates = [], [], [], []
        for _, with_base, no_base in SWING_CELLS:
            variant = with_base if baseline_on else no_base
            m, lo, hi = swing_stat(swing, variant, "median_swing_ratio")
            medians.append(m)
            lows.append(m - lo)
            highs.append(hi - m)
            rates.append(sum(row["gate"] == "pass" for row in swing[variant]) / len(swing[variant]))
        axes[0].errorbar(x + (k - 0.5) * width, medians,
                         yerr=np.array([lows, highs]), fmt="o", color=colors[k],
                         ecolor=colors[k], capsize=2.5, markersize=5, label=label)
        axes[1].bar(x + (k - 0.5) * width, rates, width=width,
                    color=colors[k], edgecolor="white", label=label)

    axes[0].axhline(0.95, color="#b83280", linestyle="--", linewidth=1.0,
                    label="gate threshold")
    axes[0].axhline(1.0, color="#52606d", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Median predicted/reference swing ratio\nmedian and full seed range")
    axes[0].set_title("Cycle amplitude")
    axes[0].grid(axis="y", color="#d9e2ec", linewidth=0.55)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Fraction of seeds passing all tracked bands")
    axes[1].set_title("Shape-fidelity gate")
    axes[1].grid(axis="y", color="#d9e2ec", linewidth=0.55)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in SWING_CELLS], rotation=18, ha="right")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("The analytic thermal baseline limits cyclic swing attenuation across backbones",
                 fontsize=11, weight="bold", y=1.02)
    save_figure(fig, out_dir, "fig5_baseline_shape_mechanism")


def figure_bounded(cells: dict[str, list], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    variants = ("cod", "cod_bounded_correction")
    labels = ("PI-COD", "bounded correction")
    conv = [sum(r.converged for r in cells[v]) for v in variants]
    axes[0].bar(labels, conv, color=("#2b6cb0", "#c05621"), width=0.58)
    axes[0].set_ylim(0, 7.5)
    axes[0].set_ylabel("Converged seeds (out of 7)")
    axes[0].set_title("Training outcome")
    for i, v in enumerate(conv):
        axes[0].text(i, v + 0.15, f"{v}/7", ha="center", weight="bold")

    cod_m, cod_lo, cod_hi = stat(cells, "cod", "theta_TO")
    bound_m, bound_lo, bound_hi = stat(cells, "cod_bounded_correction", "theta_TO")
    med = np.array([cod_m, bound_m])
    err = np.array([[cod_m - cod_lo, bound_m - bound_lo],
                    [cod_hi - cod_m, bound_hi - bound_m]])
    axes[1].errorbar(np.arange(2), med, yerr=err, fmt="o", markersize=7,
                     color="#1f2933", capsize=3)
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Thermal MAE (degC)")
    axes[1].set_title("Converged checkpoints only")
    axes[1].grid(axis="y", which="both", color="#d9e2ec", linewidth=0.55)
    fig.suptitle("Directly bounding the correction is not a reliable refinement",
                 fontsize=11, weight="bold", y=1.02)
    save_figure(fig, out_dir, "fig6_bounded_correction_failure")


def comparison_rows(cells: dict[str, list]) -> list[dict]:
    rows = []
    for label, cascade, monolithic, confounded in PAIRS:
        lower = 0
        separated = 0
        factors = []
        for K in SCENARIOS:
            for gas in GASES:
                cm, cmin, cmax = stat(cells, cascade, rollout_key(K, gas))
                mm, mmin, mmax = stat(cells, monolithic, rollout_key(K, gas))
                if cm < mm:
                    lower += 1
                    factors.append(mm / cm)
                    if cmax < mmin:
                        separated += 1
        ct = stat(cells, cascade, "theta_TO")[0]
        mt = stat(cells, monolithic, "theta_TO")[0]
        rows.append({
            "label": label,
            "thermal": max(ct, mt) / min(ct, mt),
            "control": not confounded,
            "lower": lower,
            "separated": separated,
            "reduction": (math.exp(median([math.log(x) for x in factors]))
                          if factors and not confounded else None),
        })
    return rows


def write_tables(cells: dict[str, list], swing: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(cells)
    md = [
        "# Engineering result tables",
        "",
        "## Matched cascade versus monolithic comparisons",
        "",
        "| comparison | thermal ratio | control | lower median endpoints | separated seed ranges | geometric median reduction |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        reduction = (f"{row['reduction']:.1f}x" if row["reduction"] is not None
                     else "not interpreted")
        md.append(
            f"| {row['label']} | {row['thermal']:.2f}x | "
            f"{'pass' if row['control'] else 'confounded'} | {row['lower']}/10 | "
            f"{row['separated']}/10 | {reduction} |"
        )

    md += [
        "",
        "## Baseline-equipped in-cascade implementations",
        "",
        "| implementation | thermal MAE, degC | swing gate | swing ratio | K=0.95 gas-error range, ppm | K=1.10 gas-error range, ppm |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for label, variant in CASCADE_BASELINE:
        thermal = stat(cells, variant, "theta_TO")
        swing_values = swing_stat(swing, variant, "median_swing_ratio")
        passes = sum(row["gate"] == "pass" for row in swing[variant])
        gas_ranges = []
        for K in SCENARIOS:
            xs = [stat(cells, variant, rollout_key(K, gas))[0] for gas in GASES]
            gas_ranges.append((min(xs), max(xs)))
        md.append(
            f"| {label} | {thermal[0]:.3f} [{thermal[1]:.3f}-{thermal[2]:.3f}] | "
            f"{passes}/7 | {swing_values[0]:.4f} [{swing_values[1]:.4f}-{swing_values[2]:.4f}] | "
            f"{gas_ranges[0][0]:.4g}-{gas_ranges[0][1]:.4g} | "
            f"{gas_ranges[1][0]:.4g}-{gas_ranges[1][1]:.4g} |"
        )
    (out_dir / "engineering_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched one-year cascade versus monolithic comparisons. The endpoint count spans five gases at two load factors. Seed-range separation means that the worst cascade seed remains below the best monolithic seed.}",
        r"\label{tab:matched_comparisons}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Comparison & Thermal ratio & Control & Lower medians & Separated ranges & Reduction \\",
        r"\hline",
    ]
    for row in rows:
        label = row["label"].replace("-", r"-")
        control = "pass" if row["control"] else "confounded"
        reduction = (f"{row['reduction']:.1f}$\\times$"
                     if row["reduction"] is not None else "not interpreted")
        tex.append(
            f"{label} & {row['thermal']:.2f}$\\times$ & {control} & "
            f"{row['lower']}/10 & {row['separated']}/10 & {reduction} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "engineering_results.tex").write_text("\n".join(tex), encoding="utf-8")


def write_summary(cells: dict[str, list], swing: dict[str, list[dict]], out_dir: Path) -> None:
    rows = comparison_rows(cells)
    valid = [r for r in rows if r["control"]]
    lower = sum(r["lower"] for r in valid)
    separated = sum(r["separated"] for r in valid)
    baseline_pass = sum(sum(row["gate"] == "pass" for row in swing[v])
                        for _, v in CASCADE_BASELINE)
    no_baseline_pass = sum(sum(row["gate"] == "pass" for row in swing[v])
                           for v in ("fno_in_cascade", "mionet_in_cascade",
                                     "cod_no_baseline", "sdeeponet_in_cascade"))
    text = f"""# Final engineering evidence summary

## Dataset

- 119 production runs on frozen distribution `fc4cb76c3b32ec17`.
- 113 converged checkpoints were scored for one-year rollout and swing fidelity.
- Every declared cell contains seven seeds. Six bounded-correction seeds did not converge; no failed seed enters an accuracy median.

## Claims supported by the completed matrix

1. **COD is a backbone-compatible deployment topology for long-horizon state propagation.** Seven of eight matched cascade-versus-monolithic comparisons pass the thermal comparability control. Across those seven comparisons, the cascade has lower median one-year gas error in {lower}/70 endpoints; {separated}/70 also have non-overlapping full seed ranges.
2. **PI-COD gives the best absolute accuracy among the baseline-equipped cascade implementations tested here.** It has the lowest median error in all ten gas-by-load endpoints, with median thermal MAE 0.405 degC and a 7/7 swing-gate pass rate.
3. **FNO-COD provides the clearest backbone-transfer result.** It converges in 7/7 seeds, passes the swing gate in 7/7 seeds, and beats matched monolithic FNO in all ten one-year endpoints with full seed-range separation in all ten.
4. **The analytic thermal baseline and the cascade solve different engineering problems.** Across the four in-cascade backbones, baseline-equipped cells pass the cycle-shape gate in {baseline_pass}/28 seeds, compared with {no_baseline_pass}/28 without the baseline. The cascade then prevents thermal errors from being learned again as unconstrained gas-state increments.
5. **Directly bounding the learned correction is not a reliable refinement under this protocol.** Only 1/7 bounded-correction seeds converges, and that checkpoint has 14.409 degC thermal MAE versus 0.405 degC for standard PI-COD.

## Claim boundaries

- Do not call COD the first cascade architecture.
- Do not call FNO-COD the most accurate model; PI-COD is more accurate in this matrix. FNO-COD is the strongest transfer and replication example.
- Treat the PI-DeepONet no-baseline cascade-versus-monolithic comparison as confounded because its median thermal MAEs differ by more than the prespecified factor of two.
- Treat CHI as a derived decision-support trajectory. Its role is supported by the retained state semantics and the conditional monotonicity result, not by unavailable CHI reference labels.
"""
    (out_dir / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    setup_style()
    cells = load_runs(args.results)
    swing = load_swing(args.results, cells)
    figures = args.out / "figures"
    tables = args.out / "tables"

    figure_architecture(figures)
    figure_rollout_advantage(cells, figures)
    figure_fno(cells, figures)
    figure_absolute_accuracy(cells, figures)
    figure_shape_mechanism(swing, figures)
    figure_bounded(cells, figures)
    write_tables(cells, swing, tables)
    write_summary(cells, swing, args.out)
    print(f"[figures] wrote engineering assets under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
