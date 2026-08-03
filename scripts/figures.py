"""Run figures. Plotting only — no science, per the repo rule.

`cod/eval/rollout.py` records that plotting was deliberately separated from the
package, so this lives in `scripts/` and takes already-computed arrays. Nothing
here computes a quantity that a figure then displays; if a number is worth
plotting it is worth being in `run.json`, `predictions.npz` or
`clamp_history.json` first.

WHAT CANNOT BE REBUILT FROM `predictions.npz`, AND IS THEREFORE SAVED SEPARATELY.
`predictions.npz` holds the evaluation: per-case `pred`, `gt`, `t_eval`, `x0`,
sensors, `kind` and `family`. Everything about the *fit* is reconstructible from
it offline. Nothing about *training* is:

  * the training loss curve            -> `loss_history.json`
  * the validation curve, which is what the plateau test actually decides on
    and which `run.json` used to hold only as final and best  -> `run.json`
    (`outcome.val_history`)
  * the clamp trajectories and the causal-weight trajectory, which answer "when
    did this fire", a question no summary statistic can (PORT_LOG J-85, J-89)
    -> `clamp_history.json`

Those three files exist because the figures below need them and no post-hoc
analysis can recover them. One training-time series is still **not** saved and is
named here rather than left to be discovered: the gradient norm per epoch —
`PathologyReport` keeps `grad_norm_final` only. It is not needed by any figure
here, so it is a known gap and not a silent one.

CONVENTIONS (also recorded in CLAUDE.md so figures outside this file match):
serif font; PDF and SVG side by side into `figures/` under the run directory; no
suptitle; subplot titles yes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Project figure conventions. Applied once, here, so every run figure matches.
RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}
FORMATS = ("pdf", "svg")


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    """Write one figure in every project format. No suptitle is ever added."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in FORMATS:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, format=ext)
        written.append(p)
    plt.close(fig)
    return written


def fig_learning_curves(loss_history, val_history, out_dir: Path) -> list[Path]:
    """Training loss and the validation series the convergence test runs on.

    Two panels rather than twin axes: the plateau criterion acts on the
    validation series alone, and overlaying them on one axis invites reading a
    smooth training curve as evidence of convergence when the criterion never
    looked at it.
    """
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        loss = np.asarray(loss_history, float)
        if loss.size:
            ax[0].semilogy(np.arange(1, loss.size + 1), loss, lw=0.8)
        ax[0].set_title("Training loss")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")

        if val_history:
            ep = np.array([e for e, _ in val_history], float)
            vl = np.array([v for _, v in val_history], float)
            ax[1].semilogy(ep, vl, marker="o", ms=2.5, lw=0.9)
            j = int(np.argmin(vl))
            ax[1].axvline(ep[j], ls="--", lw=0.8, color="0.4")
            ax[1].annotate(f"best {vl[j]:.3e}\nepoch {int(ep[j])}",
                           xy=(ep[j], vl[j]), xytext=(0.55, 0.8),
                           textcoords="axes fraction", fontsize=7,
                           arrowprops=dict(arrowstyle="->", lw=0.6))
        else:
            ax[1].text(0.5, 0.5, "no validation checks\n(check_every > epochs)",
                       ha="center", va="center", transform=ax[1].transAxes,
                       fontsize=8)
        ax[1].set_title("Validation loss (what the plateau test decides on)")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("validation loss")
        fig.tight_layout()
        return _save(fig, out_dir, "learning_curves")


def fig_clamp_trajectories(clamp_history: dict, causal_history,
                           out_dir: Path, threshold: float = 0.05
                           ) -> list[Path]:
    """When each clamp fired, which a peak and a final value cannot say.

    PORT_LOG J-85: the counters used to report the last epoch, so a clamp that
    fired hard early and stopped looked identical to one still firing at the end.
    J-89 names that failure mode. This is the figure that makes the difference
    visible: a spike near epoch 0 that decays is benign, a flat non-zero level is
    not.
    """
    with plt.rc_context(RC):
        names = sorted(clamp_history)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for name in names:
            v = np.asarray(clamp_history[name], float)
            if v.size:
                ax[0].plot(np.arange(1, v.size + 1), 100 * v, lw=0.9, label=name)
        ax[0].axhline(100 * threshold, ls="--", lw=0.8, color="0.4")
        ax[0].set_xscale("symlog", linthresh=100)
        ax[0].set_title("Clamp activation against epoch")
        ax[0].set_xlabel("epoch (symlog)")
        ax[0].set_ylabel("% of samples clamped")
        ax[0].legend(ncol=2, frameon=False)

        cw = np.asarray(causal_history, float)
        if cw.size:
            ax[1].semilogy(np.arange(1, cw.size + 1), np.maximum(cw, 1e-12),
                           lw=0.8)
            ax[1].axhline(1e-8, ls="--", lw=0.8, color="0.4")
            ax[1].annotate("weight floor", xy=(1, 1e-8), fontsize=7,
                           va="bottom")
        ax[1].set_title("Minimum causal weight against epoch")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("min causal weight")
        fig.tight_layout()
        return _save(fig, out_dir, "clamp_trajectories")


def fig_state_predictions(pred, gt, t_eval, state_names, out_dir: Path,
                          case: int | None = None, kinds=None) -> list[Path]:
    """One representative case: all six states, prediction against ground truth.

    The case is chosen as the **median** case by thermal MAE rather than the best
    or the worst, so the figure is representative by construction and cannot be
    read as cherry-picked. Its index is in the panel title so it is checkable
    against `predictions.npz`.
    """
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    if case is None:
        mae = np.abs(pred[:, :, 0] - gt[:, :, 0]).mean(axis=1)
        case = int(np.argsort(mae)[len(mae) // 2])
    kind = ""
    if kinds is not None:
        kind = f", {np.asarray(kinds)[case]}"

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
        units = ["degC", "ppm", "ppm", "ppm", "ppm", "ppm"]
        for i, axi in enumerate(axes.ravel()):
            axi.plot(t_eval, gt[case, :, i], lw=1.4, color="0.25",
                     label="RK45 reference")
            axi.plot(t_eval, pred[case, :, i], lw=1.1, ls="--", color="C3",
                     label="model")
            err = np.abs(pred[case, :, i] - gt[case, :, i]).mean()
            axi.set_title(f"{state_names[i]}  (MAE {err:.3g} {units[i]})")
            axi.set_ylabel(units[i])
            if i >= 3:
                axi.set_xlabel("t (min)")
        axes.ravel()[0].legend(frameon=False, loc="best")
        # Case identity goes in a corner annotation, not a suptitle.
        axes.ravel()[2].annotate(f"case {case}{kind}", xy=(0.98, 0.02),
                                 xycoords="axes fraction", ha="right",
                                 fontsize=7, color="0.4")
        fig.tight_layout()
        return _save(fig, out_dir, "state_predictions")


def write_run_figures(out_dir: Path) -> list[Path]:
    """Every figure for a completed run directory. Failures are reported, not fatal.

    Called at the end of `scripts/run.py`. A plotting error must never destroy a
    finished training run, so this catches and reports rather than raising — the
    weights and the metrics are already on disk by the time it runs.
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    written: list[Path] = []

    def _try(fn, *a, **kw):
        try:
            written.extend(fn(*a, **kw))
        except Exception as exc:                      # noqa: BLE001
            print(f"[fig] {fn.__name__} failed: {type(exc).__name__}: {exc}")

    loss_p = out_dir / "loss_history.json"
    run_p = out_dir / "run.json"
    clamp_p = out_dir / "clamp_history.json"
    pred_p = out_dir / "predictions.npz"

    loss = json.loads(loss_p.read_text(encoding="utf-8")) if loss_p.exists() else []
    val = []
    if run_p.exists():
        rec = json.loads(run_p.read_text(encoding="utf-8"))
        val = rec.get("outcome", {}).get("val_history", []) or []
    _try(fig_learning_curves, loss, val, fig_dir)

    if clamp_p.exists():
        ch = json.loads(clamp_p.read_text(encoding="utf-8"))
        _try(fig_clamp_trajectories, ch.get("clamp", {}),
             ch.get("causal_weight_min", []), fig_dir)

    if pred_p.exists():
        npz = np.load(pred_p, allow_pickle=False)
        _try(fig_state_predictions, npz["pred"], npz["gt"], npz["t_eval"],
             [str(s) for s in npz["state_names"]], fig_dir,
             None, npz["kind"] if "kind" in npz.files else None)

    return written


__all__ = ["write_run_figures", "fig_learning_curves", "fig_clamp_trajectories",
           "fig_state_predictions", "RC", "FORMATS"]
