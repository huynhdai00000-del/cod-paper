#!/usr/bin/env python3
"""Does `run.py --seed` actually change what is computed?

PORT_LOG J-93 claims it does. This script re-establishes the claim from scratch
rather than trusting it, because the failure it guards against is the one that
looks correct in every artifact it leaves behind (J-89 instance 5): a loop
variable that changes where the answer is written without changing the answer.

Three checks, and the second is the one the first cannot substitute for.

  A. **Two production runs differ.** Two run directories from `run.py --seed`,
     identical config hash and identical distribution hash — which is what makes
     them a seed sweep rather than two unrelated jobs. Their weights must differ
     and their loss curves must differ from step 0. Also asserts
     `status == "run"`: `--seed` is a production override, and a seed sweep that
     recorded itself as a smoke test would be unusable as evidence.

  B. **Batch order alone differs.** Identical model weights are loaded into both
     sides via `load_state_dict`, so initialisation is held constant *by
     construction* and anything that differs is batch order. Check A cannot
     prove this: two runs with different initial weights diverge at step 0
     whether or not the seed reaches the trainer's own generator, so step-0
     divergence is consistent with the bug being present.

  C. **The control: same seed, same draws.** Check B's comparison is only
     evidence if it can also return "identical". Two trainers built with the
     *same* seed must draw bit-identical batches. Without this, a comparison
     that always reports a difference would pass B while measuring nothing.
     (J-89 step 5: can the gate fail? Here, can it *pass* when it should?)

Run:
    python audit_port/scripts/36_seed_flag_independent_check.py \
        --run-a results/<dir_seed_a> --run-b results/<dir_seed_b>

    # checks B and C only, no run directories needed:
    python audit_port/scripts/36_seed_flag_independent_check.py --skip-runs

Exit: 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.config import load_config  # noqa: E402
from cod.data.generate import generate_realistic_training_set  # noqa: E402
from cod.data.realistic import RealisticParams  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402
from run import build_model, build_trainer  # noqa: E402

#: Cheap cell, and the same one J-93 used, so the two results are comparable.
CONFIG = ROOT / "configs" / "matrix" / "mionet_in_cascade.yaml"
N_IC_INSTRUMENT = 96      # checks B/C need a dataset, not a good one
N_BATCHES = 3


def _fail(msg: str, failures: list) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


# ── A. two production runs ────────────────────────────────────────────────────
def check_runs(run_a: Path, run_b: Path, failures: list) -> None:
    print("\n=== A. two production runs of the same cell, different --seed ===")
    ja = json.loads((run_a / "run.json").read_text(encoding="utf-8"))
    jb = json.loads((run_b / "run.json").read_text(encoding="utf-8"))

    print(f"  A: {run_a.name}")
    print(f"  B: {run_b.name}")

    # The pair is only a seed sweep if everything except the seed is the same.
    for key, path in (("config_hash", ("config", "config_hash")),
                      ("distribution_hash", ("config", "distribution_hash"))):
        va, vb = ja[path[0]][path[1]], jb[path[0]][path[1]]
        if va != vb:
            _fail(f"{key} differs ({va} vs {vb}): these are two different "
                  "experiments, not two seeds of one", failures)
        else:
            print(f"  ok   {key} identical: {va}")

    if ja["seed"] == jb["seed"]:
        _fail(f"both runs record seed {ja['seed']}", failures)
    else:
        print(f"  ok   seeds differ: {ja['seed']} vs {jb['seed']}")

    # `--seed` is a production override. `--n-ic`/`--max-epochs` shrink the
    # experiment and set smoke status; a seed sweep IS the experiment.
    for name, j in (("A", ja), ("B", jb)):
        st = j.get("status")
        if st != "run":
            _fail(f"run {name} recorded status {st!r}, expected 'run'. A seed "
                  "sweep marked as a smoke test cannot be reported.", failures)
        else:
            print(f"  ok   run {name} status = 'run'")
        if int(j["n_ic"]) != int(load_config(CONFIG)["distribution"]["n_ic"]):
            print(f"  note run {name} n_ic = {j['n_ic']} (config default is "
                  f"{load_config(CONFIG)['distribution']['n_ic']})")

    # Weights. Loaded on CPU, compared tensor by tensor.
    sa = torch.load(run_a / "model.pt", map_location="cpu",
                    weights_only=False)["model_state_dict"]
    sb = torch.load(run_b / "model.pt", map_location="cpu",
                    weights_only=False)["model_state_dict"]
    if sa.keys() != sb.keys():
        _fail("state_dict keys differ; not the same architecture", failures)
        return
    diffs = {k: float((sa[k].float() - sb[k].float()).abs().max())
             for k in sa if sa[k].dtype.is_floating_point}
    n_diff = sum(1 for v in diffs.values() if v > 0)
    worst = max(diffs, key=diffs.get)
    print(f"  {n_diff}/{len(diffs)} weight tensors differ; "
          f"max |diff| {diffs[worst]:.4g} on {worst}")
    if n_diff == 0:
        _fail("every weight tensor is bit-identical: the seed did not reach "
              "weight initialisation", failures)

    # Loss curves, from step 0. Equal-length prefix so a different epoch count
    # (the wall-clock budget does not stop two runs at the same epoch) is not
    # itself read as a difference.
    la = json.loads((run_a / "loss_history.json").read_text(encoding="utf-8"))
    lb = json.loads((run_b / "loss_history.json").read_text(encoding="utf-8"))
    n = min(len(la), len(lb))
    if n == 0:
        _fail("a loss history is empty", failures)
        return
    d = np.abs(np.asarray(la[:n]) - np.asarray(lb[:n]))
    print(f"  loss history: {len(la)} vs {len(lb)} epochs; over the common "
          f"{n}, step-0 |diff| {d[0]:.4g}, max |diff| {d.max():.4g}")
    if d[0] == 0.0:
        _fail("the two runs have an identical loss at step 0", failures)
    if d.max() == 0.0:
        _fail("the two loss curves are bit-identical", failures)


# ── B and C. batch order, with initialisation held constant ──────────────────
def _trainer_with_seed(cfg, ts, seed, ref_state, device):
    """A trainer whose model carries exactly `ref_state`, seeded with `seed`.

    The weights are overwritten after construction, so whatever the global RNG
    did during `build_model` is erased. What remains different between two calls
    with different `seed` is the trainer's own batch-order generator.
    """
    model, predict_fn = build_model(cfg, ts.x_mean, ts.x_std, device)
    model.load_state_dict(ref_state)
    model = model.to(device)
    trainer, _ = build_trainer(cfg, model, predict_fn, ts, device,
                               max_epochs=10, seed=seed)
    return trainer


def _draw(trainer, k):
    """The first `k` batches, as arrays. `_gather` returns (x0, sensors, ss)."""
    out = []
    for _ in range(k):
        x0, sens, _ss = trainer.data.train_batch()
        out.append((x0.detach().cpu().numpy().copy(),
                    sens.detach().cpu().numpy().copy()))
    return out


def _compare_draws(da, db, label, expect_differ, failures):
    max_x0 = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(da, db))
    max_se = max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(da, db))
    n_same = sum(1 for a, b in zip(da, db) if np.array_equal(a[0], b[0]))
    print(f"  {label}: {n_same}/{len(da)} of the first batches identical; "
          f"max |diff| {max_x0:.4g} on drawn x0, {max_se:.4g} on sensors")
    differ = max_x0 > 0 or max_se > 0
    if expect_differ and not differ:
        _fail(f"{label}: two seeds drew identical batches — the seed does not "
              "reach the trainer's batch-order generator, so a seed sweep "
              "reports initialisation variance alone", failures)
    if (not expect_differ) and differ:
        _fail(f"{label}: the SAME seed drew different batches. The comparison "
              "above cannot distinguish a real difference from an unrelated "
              "source of randomness, so check B proves nothing.", failures)


def check_batch_order(failures: list) -> None:
    print("\n=== B/C. batch order, with weights held constant by construction ===")
    cfg = load_config(CONFIG)
    device = torch.device("cpu")
    params = RealisticParams.from_config(
        cfg["distribution"]["sampler"].get("params", {}))
    ts = generate_realistic_training_set(n_ic=N_IC_INSTRUMENT, seed=42,
                                         params=params)
    print(f"  instrument dataset: {len(ts)} ICs (this is not a run)")

    ref_model, _ = build_model(cfg, ts.x_mean, ts.x_std, device)
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    da = _draw(_trainer_with_seed(cfg, ts, 11, ref_state, device), N_BATCHES)
    db = _draw(_trainer_with_seed(cfg, ts, 12, ref_state, device), N_BATCHES)
    dc = _draw(_trainer_with_seed(cfg, ts, 11, ref_state, device), N_BATCHES)

    _compare_draws(da, db, "B  seed 11 vs seed 12", True, failures)
    _compare_draws(da, dc, "C  seed 11 vs seed 11 (control)", False, failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, default=None)
    ap.add_argument("--run-b", type=Path, default=None)
    ap.add_argument("--skip-runs", action="store_true",
                    help="Checks B and C only; no run directories needed.")
    args = ap.parse_args()

    failures: list = []
    if not args.skip_runs:
        if not (args.run_a and args.run_b):
            raise SystemExit("--run-a and --run-b are required (or --skip-runs)")
        check_runs(args.run_a, args.run_b, failures)
    check_batch_order(failures)

    print("\n" + "=" * 72)
    if failures:
        print(f"[FAIL] {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[PASS] --seed reaches weight initialisation AND batch order, and the "
          "runs record status 'run'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
