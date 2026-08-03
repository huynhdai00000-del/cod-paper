#!/usr/bin/env python3
"""Does a checkpoint written by `run.py` reload into the same model?

`run.py` gained a `torch.save` so that a trained model survives the process that
trained it. A save path that has never been round-tripped is not verified: it can
write a file that loads without error and still be missing the normalisation
buffers, or carry a `theta_ss_mode` that silently differs from the one trained
under, and every downstream measurement would then be of a different model than
the one whose metrics are in run.json.

The test is end to end and deliberately crosses a process boundary:

  1. run `scripts/run.py` as a subprocess (a real training job, small budget);
  2. read the metrics it recorded in run.json;
  3. **in this process**, which never saw the trained weights, load model.pt,
     rebuild the same test set from the same config, and recompute the same
     metrics;
  4. require them to agree to within `TOL`.

Step 3 is the point. Recomputing inside the training process would prove nothing —
the model object is still in memory there. Here the only channel between training
and scoring is the file.

`TOL` is tight on purpose. Both sides run the identical deterministic forward pass
on the identical inputs, so the difference should be zero; anything above float
round-off means the file does not carry the whole model.

Run:  python audit_port/scripts/25_checkpoint_roundtrip.py
Exit: 0 if every reloaded metric matches, 1 otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import (  # noqa: E402
    build_realistic_test_set, build_test_set, rk45_ground_truth,
)
from cod.data.physics import STATE_DIM_FAST, TW  # noqa: E402
from cod.data.realistic import RealisticParams  # noqa: E402
from cod.data.steady_state import formula_A, true_fixed_point_np  # noqa: E402
from cod.eval.metrics import TRANSFORMER_STATES, evaluate_state  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402
from run import build_model  # noqa: E402

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
MATRIX_DIR = ROOT / "configs" / "matrix"
OUTROOT = ROOT / "results"
TAG = "roundtrip"
TOL = 1e-6          # relative; the two sides should be bit-identical

#: Every architecture that can be trained, not just COD. The original version of
#: this script checked `cod` alone, which would have left five C-11 cells free to
#: train for an hour on Colab and discard their weights — the O-5 failure repeated
#: five more times. A save path is verified per architecture or it is not verified.
ALL_CONFIGS = [CONFIG] + sorted(MATRIX_DIR.glob("*.yaml"))


def train_subprocess(cfg_path: Path, max_epochs: int, n_ic: int,
                     n_test: int) -> Path:
    """Run a real training job in its own process and return its output dir."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"),
           "--config", str(cfg_path), "--max-epochs", str(max_epochs),
           "--n-ic", str(n_ic), "--n-test", str(n_test),
           "--device", "cpu", "--tag", TAG, "--out", str(OUTROOT), "--overwrite"]
    print("[sub] " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("[FAIL] the training subprocess exited non-zero")
    out_line = [l for l in r.stdout.splitlines() if l.startswith("[run] out")]
    out_dir = Path(out_line[-1].split(None, 2)[2].strip())
    print(f"[sub] finished, out_dir = {out_dir}")
    return out_dir


def rebuild_cases(cfg, n_test):
    """The same test set run.py built, from the same config."""
    dist = cfg["distribution"]
    tier_cfg = cfg["evaluation"]["tiers"]
    tier_name = next(iter(tier_cfg))
    tier = tier_cfg[tier_name]
    T = float(dist["window_minutes"])
    if tier["source"] == "realistic_sampler":
        params = RealisticParams.from_config(dist["sampler"]["params"])
        return build_realistic_test_set(n_test=n_test, seed=int(tier["seed"]),
                                        params=params, T=T), T, tier_name
    ic = (formula_A if dist.get("steady_state_formula") == "A"
          else true_fixed_point_np)
    return (build_test_set(n_test=n_test, seed=int(tier["seed"]), T=T,
                           steady_state=ic), T, tier_name)


def score_from_checkpoint(ckpt_path: Path, cfg, n_test: int, guard: float):
    """Load the weights from disk and recompute run.py's physical metrics."""
    device = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "x_mean", "x_std", "theta_ss_mode",
                "distribution_hash", "config_hash"):
        if key not in ck:
            raise SystemExit(f"[FAIL] checkpoint is missing {key!r}")

    # Built through run.py's own builder so this checks the real construction
    # path per architecture rather than a COD-shaped reimplementation of it.
    # Placeholder normalisation: the buffers come back with the strict load,
    # and if they were NOT in the file strict=True fails here, which is the
    # check rather than an inconvenience.
    model, predict_fn = build_model(cfg, np.zeros(6), np.ones(6), device)
    model = model.to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    # The saved normalisation must be what the model actually carries.
    if not np.allclose(model.x_mean_TO.numpy(), np.asarray(ck["x_mean"])[:1],
                       rtol=0, atol=0):
        raise SystemExit("[FAIL] x_mean in the checkpoint metadata does not "
                         "match the x_mean_TO buffer restored from the weights")

    cases, T, tier_name = rebuild_cases(cfg, n_test)
    t_eval = np.linspace(0, T, 50)
    gt = np.stack([rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors, t_eval,
                                     T=T, t_clip_frac=guard) for c in cases])
    with torch.no_grad():
        t_q = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(-1)
        preds = []
        for c in cases:
            s_k = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                               dtype=torch.float32).unsqueeze(0)
            x0_t = torch.tensor(c.x0, dtype=torch.float32).unsqueeze(0)
            preds.append(predict_fn(model, x0_t.expand(50, -1).contiguous(),
                                    s_k.expand(50, -1).contiguous(),
                                    t_q).numpy())
    pred = np.stack(preds)
    return {spec.name: evaluate_state(pred[:, :, i], gt[:, :, i], spec).to_dict()
            for i, spec in enumerate(TRANSFORMER_STATES[:STATE_DIM_FAST])}, ck


def check_one(cfg_path: Path, max_epochs: int, n_ic: int, n_test: int) -> int:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    print(f"\n{'=' * 72}\n### {cfg_path.name}  (kind = {cfg['model']['kind']})\n"
          f"{'=' * 72}")
    out_dir = train_subprocess(cfg_path, max_epochs=max_epochs, n_ic=n_ic,
                               n_test=n_test)

    ckpt = out_dir / "model.pt"
    print(f"\n=== 1. does the file exist ===")
    for f in ("run.json", "model.pt", "loss_history.json", "predictions.npz"):
        exists = (out_dir / f).exists()
        size = (out_dir / f).stat().st_size if exists else 0
        print(f"  [{'ok ' if exists else 'MISSING'}] {f:20s} {size:>10,} bytes")
        if not exists:
            print(f"[FAIL] {f} was not written")
            return 1

    record = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    recorded = record["evaluation"]["mae_physical_units"]
    guard = float(record["evaluation"]["right_edge_guard"])

    print("\n=== 2. reload in this process and rescore ===")
    reloaded, ck = score_from_checkpoint(ckpt, cfg, n_test, guard)
    print(f"  checkpoint config_hash       {ck['config_hash']}")
    print(f"  checkpoint distribution_hash {ck['distribution_hash']}")
    print(f"  run.json  distribution_hash  {record['config']['distribution_hash']}")
    if ck["distribution_hash"] != record["config"]["distribution_hash"]:
        print("[FAIL] the checkpoint and run.json disagree about the distribution")
        return 1

    print("\n=== 3. do the metrics match ===")
    fields = ("mae", "rmse", "max_abs_error", "nmae_median", "denominator_median")
    print(f"{'state':>10} {'field':>20} {'run.json':>16} {'reloaded':>16} {'rel':>10}")
    bad = []
    for name, rec in recorded.items():
        for f in fields:
            a, b = float(rec[f]), float(reloaded[name][f])
            rel = abs(a - b) / max(abs(a), 1e-30)
            flag = "" if rel <= TOL else "  <-- MISMATCH"
            if rel > TOL:
                bad.append(f"{name}.{f}: {a!r} vs {b!r} (rel {rel:.3e})")
            print(f"{name:>10} {f:>20} {a:16.9g} {b:16.9g} {rel:10.2e}{flag}")

    # The stored predictions must also be the ones the metrics were computed from,
    # or the offline rebuild path is decorative.
    print("\n=== 4. do the stored predictions reproduce the stored metrics ===")
    npz = np.load(out_dir / "predictions.npz", allow_pickle=False)
    p, g = npz["pred"], npz["gt"]
    for i, spec in enumerate(TRANSFORMER_STATES[:STATE_DIM_FAST]):
        m = evaluate_state(p[:, :, i].astype(float), g[:, :, i].astype(float),
                           spec)
        a = float(recorded[spec.name]["mae"])
        rel = abs(a - m.mae) / max(abs(a), 1e-30)
        # Same TOL as the weight check: the arrays are stored float64, so this
        # is a lossless round-trip and there is no reason to allow slack. An
        # earlier float32 version failed here at 3.5e-4 on c_H2, which is how the
        # storage dtype got fixed rather than the tolerance widened.
        if rel > TOL:
            bad.append(f"predictions.npz {spec.name}.mae: {a!r} vs {m.mae!r} "
                       f"(rel {rel:.3e})")
        print(f"  {spec.name:>10} mae {a:14.8g} vs {m.mae:14.8g}  rel {rel:.2e}")

    if bad:
        print(f"\nFAIL ({cfg_path.name}) — the reloaded model is not the model "
              "that was scored:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"\nPASS ({cfg_path.name}) — every metric reproduced from the reloaded "
          f"weights within {TOL:g} relative, and predictions.npz reproduces them.")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", type=Path, default=None,
                    help="Configs to check. Default: COD plus every C-11 matrix "
                         "cell, because a save path verified for one "
                         "architecture is not verified for the others.")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--n-ic", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=8)
    args = ap.parse_args()

    configs = args.configs if args.configs else ALL_CONFIGS
    print(f"[plan] {len(configs)} architecture(s) to round-trip: "
          + ", ".join(c.stem for c in configs))

    results = {}
    for cfg_path in configs:
        # One architecture failing must not abort the others: the point is to
        # learn which save paths work, and stopping at the first failure hides
        # every cell after it.
        try:
            results[cfg_path.name] = check_one(cfg_path, args.max_epochs,
                                               args.n_ic, args.n_test)
        except SystemExit as exc:
            print(f"[FAIL] {cfg_path.name}: {exc}")
            results[cfg_path.name] = 1
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {cfg_path.name}: {type(exc).__name__}: {exc}")
            results[cfg_path.name] = 1

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for name, rc in results.items():
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name}")
    n_fail = sum(1 for rc in results.values() if rc)
    print(f"\n{len(results) - n_fail}/{len(results)} architectures round-trip "
          "their checkpoints correctly.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
