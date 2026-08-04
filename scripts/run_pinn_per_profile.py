#!/usr/bin/env python3
"""Train one PINN per profile — C-11 tier 2, the amortisation baseline.

The audit lists the §7.2 PINN baseline as not existing. This is it.

WHAT THIS PRODUCES. Two numbers per profile and one comparison:

  * the PINN's own accuracy on its own profile, against RK45;
  * the wall-clock cost of getting it, measured rather than asserted — the audit
    found every timing claim in the manuscript untraceable to an artifact;
  * the amortisation ledger: `N x PINN training` against
    `operator training once + N x forward pass`.

READ THE COMPARISON THE RIGHT WAY ROUND. A converged per-profile PINN is the
**specialised gold standard** for its profile and should be *more* accurate than
an operator that must be right about a whole distribution with similar capacity.
If COD beats a converged PINN on that PINN's own profile, the finding to
investigate is the PINN, not a win for COD. See `cod/models/pinn.py`.

FAIRNESS. The PINN gets causal weighting (Wang et al. 2024 Eq. 3.2, the same
`causal_weights` the operator uses), the same hard IC ansatz, the same per-state
output scaling, and the same convergence criterion. Its wall-clock budget is
deliberately **not** capped to the operator's: it trains to its own plateau,
because capping it would rig the one number this baseline exists to produce.

WHICH PROFILES. Drawn from the same frozen sampler as the T1 tier at the same
held-out seed, so the profiles are the ones the operator is scored on. Note that
"per-profile" means this model legitimately sees the profile it is evaluated on —
that is what makes it the gold standard — but it never sees the RK45 solution,
only the ODE residual, so no label leaks.

Run:
    python scripts/run_pinn_per_profile.py --n-profiles 10 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cod.data.generate import build_realistic_test_set, rk45_ground_truth
from cod.data.physics import STATE_DIM_FAST, STATE_NAMES_FAST, TW
from cod.data.realistic import RealisticParams
from cod.eval.metrics import TRANSFORMER_STATES, evaluate_state
from cod.models.pinn import PINN_DEPTH, PINN_WIDTH, PerProfilePINN
from cod.provenance import warn_if_dirty, write_run_record
from cod.training.harness import ConvergenceCriterion, train
from cod.training.pinn_train import PerProfilePINNTrainer

CONFIG = ROOT / "configs" / "example_cod_seed1.yaml"
N_EVAL = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--n-profiles", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tier-seed", type=int, default=None,
                    help="Defaults to the T1 tier seed, so the profiles are the "
                         "ones the operator is scored on.")
    ap.add_argument("--max-epochs", type=int, default=20000)
    ap.add_argument("--max-wall-seconds", type=float, default=1800.0,
                    help="Per PINN. Not the operator's budget: this baseline "
                         "trains to its own plateau, and capping it would rig "
                         "the amortisation comparison.")
    ap.add_argument("--width", type=int, default=PINN_WIDTH)
    ap.add_argument("--depth", type=int, default=PINN_DEPTH)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-collocation", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    warn_if_dirty()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dist = cfg["distribution"]
    params = RealisticParams.from_config(dist["sampler"]["params"])
    T = float(dist["window_minutes"])
    tier = cfg["evaluation"]["tiers"]["T1_in_distribution"]
    tier_seed = args.tier_seed if args.tier_seed is not None else int(tier["seed"])
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.out) / (f"pinn_per_profile_n{args.n_profiles}"
                                f"_s{args.seeds}"
                                f"{('_' + args.tag) if args.tag else ''}")
    if (out_dir / "run.json").exists():
        raise SystemExit(f"[pinn] {out_dir} already holds a run.json; use --tag.")

    print(f"[pinn] {args.n_profiles} profiles x {args.seeds} seeds  device={device}")
    print(f"[pinn] profiles from the frozen sampler at tier seed {tier_seed}")
    cases = build_realistic_test_set(n_test=args.n_profiles, seed=tier_seed,
                                     params=params, T=T)

    # x_std for the output scaling, from the training distribution rather than
    # from these profiles — using the evaluation profiles' own spread would leak
    # information the operator does not get.
    from cod.data.generate import generate_realistic_training_set
    ts = generate_realistic_training_set(n_ic=512, seed=int(dist.get("seed", 42)),
                                         params=params)
    x_std = np.asarray(ts.x_std)

    t_eval = np.linspace(0, T, N_EVAL)
    rows = []
    for pi, case in enumerate(cases):
        gt = rk45_ground_truth(case.x0, case.K_sensors, case.Ta_sensors, t_eval,
                               T=T)
        sensors = np.concatenate([case.K_sensors, case.Ta_sensors])
        for seed in range(1, args.seeds + 1):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PerProfilePINN(case.x0, sensors, T=T, width=args.width,
                                   depth=args.depth, x_std=x_std).to(device)
            trainer = PerProfilePINNTrainer(
                model, case.x0, sensors, device=device, lr=args.lr,
                n_col=args.n_collocation, seed=seed, T=T)
            crit = ConvergenceCriterion(
                max_epochs=args.max_epochs,
                max_wall_seconds=args.max_wall_seconds,
                patience=int(cfg["training"]["convergence"]["patience"]),
                min_delta_rel=float(cfg["training"]["convergence"]["min_delta_rel"]),
                check_every=int(cfg["training"]["convergence"]["check_every"]))
            t0 = time.monotonic()
            outcome = train(trainer, crit, log_every=10 ** 9)
            wall = time.monotonic() - t0

            with torch.no_grad():
                tq = torch.tensor(t_eval, dtype=torch.float32,
                                  device=device).unsqueeze(-1)
                pred = model.predict(tq).cpu().numpy()
            # Inference cost, which is the other half of the amortisation ledger.
            with torch.no_grad():
                t1 = time.monotonic()
                for _ in range(20):
                    model.predict(tq)
                infer = (time.monotonic() - t1) / 20

            per_state = {}
            for i, spec in enumerate(TRANSFORMER_STATES[:STATE_DIM_FAST]):
                m = evaluate_state(pred[None, :, i], gt[None, :, i], spec)
                per_state[spec.name] = {"mae": m.mae, "max_abs": m.max_abs_error}
            rows.append({
                "profile": pi, "seed": seed, "kind": case.kind,
                "family": case.family,
                "converged": bool(outcome.converged),
                "stop_reason": outcome.stop_reason,
                "epochs": int(outcome.epochs_reached),
                "wall_seconds": float(wall),
                "infer_seconds": float(infer),
                "n_parameters": int(model.n_parameters()),
                "mae": {k: float(v["mae"]) for k, v in per_state.items()},
            })
            r = rows[-1]
            print(f"  profile {pi:2d} seed {seed}  {r['stop_reason']:18s} "
                  f"{r['epochs']:6d} ep  {wall:7.1f} s  "
                  f"theta_TO MAE {r['mae']['theta_TO']:.4f} degC")

    # ── Amortisation ledger ────────────────────────────────────────────────
    conv = [r for r in rows if r["converged"]]
    tot_pinn = sum(r["wall_seconds"] for r in rows)
    med_pinn = float(np.median([r["wall_seconds"] for r in rows])) if rows else 0.0
    med_infer = float(np.median([r["infer_seconds"] for r in rows])) if rows else 0.0
    print(f"\n=== cost ===")
    print(f"  PINN training, median per profile : {med_pinn:.1f} s")
    print(f"  PINN training, total this run     : {tot_pinn:.1f} s "
          f"({len(rows)} fits)")
    print(f"  PINN inference, median            : {1000 * med_infer:.2f} ms")
    print(f"  converged                         : {len(conv)}/{len(rows)}")
    if conv:
        th = [r["mae"]["theta_TO"] for r in conv]
        print(f"  theta_TO MAE over converged fits  : median {np.median(th):.4f}, "
              f"min {min(th):.4f}, max {max(th):.4f} degC")

    extra = {
        "status": "run",
        "baseline": "pinn_per_profile",
        "tier": "T1_in_distribution",
        "tier_seed": tier_seed,
        "n_profiles": args.n_profiles,
        "seeds_per_profile": args.seeds,
        "hyperparameters": {"width": args.width, "depth": args.depth,
                            "lr": args.lr, "n_collocation": args.n_collocation,
                            "max_epochs": args.max_epochs,
                            "max_wall_seconds": args.max_wall_seconds},
        "distribution_hash": None,
        "cost": {"pinn_train_median_s": med_pinn,
                 "pinn_train_total_s": tot_pinn,
                 "pinn_infer_median_s": med_infer,
                 "n_fits": len(rows),
                 "n_converged": len(conv)},
        "per_fit": rows,
        "note": "A per-profile PINN is the specialised gold standard for its own "
                "profile and should be MORE accurate than an operator. The claim "
                "this baseline supports is amortised cost, not accuracy "
                "superiority. If the operator wins here, investigate this PINN.",
    }
    from cod.config import load_config
    extra["distribution_hash"] = load_config(Path(args.config)).distribution_hash
    path = write_run_record(out_dir, {"config_path": str(args.config)}, 0,
                            extra=extra)
    print(f"\n[pinn] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
