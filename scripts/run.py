#!/usr/bin/env python3
"""Single entry point for every experiment.

Usage:
    python scripts/run.py --config configs/example_cod_seed1.yaml --out results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cod.config import load_config
from cod.provenance import warn_if_dirty, write_run_record
from cod.training.harness import ConvergenceCriterion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--freeze-hash", default=None,
                    help="Expected distribution hash; fails if it has changed.")
    args = ap.parse_args()

    warn_if_dirty()
    cfg = load_config(args.config)

    if args.freeze_hash:
        from cod.config import assert_distribution_unchanged
        assert_distribution_unchanged(cfg, args.freeze_hash)

    seed = int(cfg["training"]["seed"])
    out_dir = Path(args.out) / f"{cfg['experiment']['name']}_{cfg['experiment']['variant']}_s{seed}_{cfg.hash}"

    print(f"[run] config          {args.config}")
    print(f"[run] config hash     {cfg.hash}")
    print(f"[run] distribution    {cfg.distribution_hash}")
    print(f"[run] seed            {seed}")
    print(f"[run] out             {out_dir}")

    crit = ConvergenceCriterion(**cfg["training"]["convergence"])

    # TODO(Claude Code): build model from cfg['model'], build data from
    # cfg['distribution'], then:
    #     outcome = train(model, crit)
    #     metrics = evaluate_system(pred, true)
    # and pass outcome.to_dict() / metrics into write_run_record(extra=...).

    write_run_record(out_dir, cfg.summary(), seed,
                     extra={"convergence_criterion": crit.__dict__,
                            "status": "scaffold_only"})
    print(f"[run] wrote {out_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
