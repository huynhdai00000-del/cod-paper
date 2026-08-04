#!/usr/bin/env python3
"""Emit one config per C-11 matrix cell, sharing the frozen distribution verbatim.

WHY GENERATE RATHER THAN HAND-WRITE. C-11's comparison is only meaningful if every
architecture trains on the same distribution and is scored on the same tier. Four
hand-copied YAML files drift: one gets a parameter edited during debugging and the
matrix silently stops being a comparison. Here the `distribution` and `evaluation`
blocks are copied verbatim from `configs/example_cod_seed1.yaml`, so every emitted
config carries distribution hash `fc4cb76c3b32ec17` by construction, and
`scripts/run.py --freeze-hash` will reject any that does not.

What differs between cells is exactly two things: the model block, and the
training loop. Everything else is held fixed on purpose.

The wall-clock budget is deliberately left as the base config's value and is NOT
the matrix budget — see C-11, which records that the tier-1 budget is a separate
decision taken by finding where COD converges comfortably and then applying that
same figure to every architecture. Copying 7200 s out of the O-5 config would
import a number chosen for a different purpose. `run.py --max-wall-seconds` is how
the matrix budget gets applied once it exists.

Run:  python scripts/make_matrix_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "example_cod_seed1.yaml"
OUT_DIR = ROOT / "configs" / "matrix"

#: (config stem, model block). The model block is the ONLY architectural
#: difference between cells; defaults inside each class are the paper's values,
#: so an entry here that sets nothing is running the reference configuration.
CELLS = {
    "fno_monolithic": {
        "kind": "fno_monolithic",
        "notes": "FNO (Li et al. 2021), all six states direct. "
                 "1-d paper config: modes 16, width 64, 4 layers.",
        "fno": {"modes": 16, "width": 64, "layers": 4, "domain_padding": 0},
    },
    "fno_in_cascade": {
        "kind": "fno_in_cascade",
        "notes": "FNO predicting theta_TO only; gases by Arrhenius quadrature.",
        "fno": {"modes": 16, "width": 64, "layers": 4, "domain_padding": 0},
    },
    "mionet_monolithic": {
        "kind": "mionet_monolithic",
        "notes": "MIONet (Jin et al. 2022) low-rank, all six states direct. "
                 "ODE-experiment config: depth 2, width 200.",
        "mionet": {"depth": 2, "width": 200, "basis_dim": 200},
    },
    "mionet_in_cascade": {
        "kind": "mionet_in_cascade",
        "notes": "MIONet predicting theta_TO only; gases by Arrhenius quadrature.",
        "mionet": {"depth": 2, "width": 200, "basis_dim": 200},
    },
    "sdeeponet_monolithic": {
        "kind": "sdeeponet_monolithic",
        "notes": "S-DeepONet (He et al. 2024), GRU branch, all six states. "
                 "TRUNK TAKES t, not spatial coordinates -- the published design "
                 "has the trunk take (x,y) with time entering only via the "
                 "branch. See PORT_LOG J-90; this is a departure, not a "
                 "parameter change, and is a candidate explanation if this cell "
                 "underperforms.",
        "sdeeponet": {"cell": "gru", "trunk_layers": 6},
    },
    "sdeeponet_in_cascade": {
        "kind": "sdeeponet_in_cascade",
        "notes": "S-DeepONet predicting theta_TO only; gases by quadrature. "
                 "Same trunk departure as above.",
        "sdeeponet": {"cell": "gru", "trunk_layers": 6},
    },
    "cod_bounded_correction": {
        "kind": "cod",
        "notes": "ANALYSIS_PLAN Amendment 2. COD with tanh on the neural "
                 "correction, so |correction| <= sigma holds BY CONSTRUCTION "
                 "rather than empirically. NOT a factorial cell -- reported "
                 "separately against COD on the same seeds. Adding it to the "
                 "factorial would double the matrix to test a design "
                 "refinement rather than the hypothesis.",
        "branch": {"layers": 4, "width": 128, "arch": "modified_mlp"},
        "trunk": {"layers": 3, "width": 128, "arch": "modified_mlp"},
        "basis_dim": 64,
        "steady_state": "true_fixed_point",
        "bounded_correction": True,
    },
    "cod_no_baseline": {
        "kind": "cod_no_baseline",
        "notes": "DECISIONS O-12 / N-8, Ablation A. COD with the analytic "
                 "baseline H replaced by the constant x0 -- same network, trunk, "
                 "cascade, trainer and budget. ONE variable. The monolithic "
                 "checkpoints are not a substitute: they drop the baseline AND "
                 "the cascade, and carry J-8.",
        "branch": {"layers": 4, "width": 128, "arch": "modified_mlp"},
        "trunk": {"layers": 3, "width": 128, "arch": "modified_mlp"},
        "basis_dim": 64,
        "steady_state": "true_fixed_point",
    },
}

#: Cells whose training block must differ from the shared default. Ablation A
#: uses COD's own loop, because a different trainer would be a second variable
#: and O-12 is a one-variable test.
TRAINING_OVERRIDES = {
    "cod_no_baseline": {"loop": "train_v34"},
    "cod_bounded_correction": {"loop": "train_v34"},
}


def main() -> int:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, model in CELLS.items():
        cfg = {
            "experiment": {
                "name": "cod_matrix",
                "variant": stem,
                "notes": "C-11 tier-1 matrix cell. Generated by "
                         "scripts/make_matrix_configs.py -- edit that, not this.",
            },
            # Verbatim from the base config, so the frozen hash is shared by
            # construction rather than by careful copying.
            "distribution": base["distribution"],
            "model": model,
            "training": {
                **base["training"],
                # Every baseline uses the shared physics trainer, which is what
                # makes the budget and the convergence criterion comparable
                # (C-11: equal wall clock, not equal epochs). Ablation A is the
                # exception and takes COD's loop; see TRAINING_OVERRIDES.
                "loop": "train_physics",
                **TRAINING_OVERRIDES.get(stem, {}),
            },
            "evaluation": base["evaluation"],
        }
        path = OUT_DIR / f"{stem}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False,
                                       default_flow_style=False),
                        encoding="utf-8")
        written.append(path)

    # Assert what the generator exists to guarantee.
    sys.path.insert(0, str(ROOT))
    from cod.config import load_config
    base_hash = load_config(BASE).distribution_hash
    print(f"base distribution hash: {base_hash}")
    bad = []
    for p in written:
        h = load_config(p).distribution_hash
        ok = h == base_hash
        print(f"  [{'ok ' if ok else 'DIFF'}] {p.relative_to(ROOT)}  {h}")
        if not ok:
            bad.append(p.name)
    if bad:
        print(f"\nFAIL: {bad} do not share the frozen distribution")
        return 1
    print(f"\nPASS: {len(written)} configs, all on distribution {base_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
