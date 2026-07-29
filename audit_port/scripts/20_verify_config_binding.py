#!/usr/bin/env python3
"""Verify the sampler is genuinely on the config path.

The freeze is only worth something if the hashed text is what produces the data.
`DISTRIBUTION_FREEZE.md` §2.1 recorded the opposite state of affairs: `K_base`
sat in the hashed block, moved the hash when edited, and reached no sampler.

Four checks:

1. Every `RealisticParams` field appears in the config, and nothing else does.
   Enforced by `from_config`; this asserts it fires on both a missing key and an
   unknown one, so the enforcement itself is tested rather than assumed.
2. Editing `cycle_period` in the YAML **changes the generated data**. This is the
   direct refutation of the old failure mode. A knob that moves the hash must
   move the dataset.
3. Editing a knob changes the distribution hash, so the change cannot be made
   quietly.
4. The v57 path still reproduces its own data and rejects `params`.

Run:  python audit_port/scripts/20_verify_config_binding.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.config import canonical_hash, load_config  # noqa: E402
from cod.data.generate import (generate_realistic_training_set,  # noqa: E402
                               generate_training_set)
from cod.data.realistic import RealisticParams  # noqa: E402

CFG = ROOT / "configs" / "example_cod_seed1.yaml"
V57 = ROOT / "configs" / "v57_faithful.yaml"
N = 24          # enough to see a distribution move; small enough to be quick


def hs_swing(ts):
    """Median half peak-to-peak of the load profile — cheap proxy for the shape."""
    K = ts.sensors[:, :100]
    return float(np.median(0.5 * (K.max(axis=1) - K.min(axis=1))))


def main() -> int:
    fails = []
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    block = raw["distribution"]["sampler"]["params"]

    # ── 1. completeness, both directions ───────────────────────────────────
    print("=== 1. config completeness ===")
    p = RealisticParams.from_config(block)
    print(f"accepted: {len(block)} params, cycle_period={p.cycle_period:g}")

    missing = copy.deepcopy(block)
    del missing["K_amp"]
    try:
        RealisticParams.from_config(missing)
        fails.append("a missing K_amp was accepted")
        print("MISSING-KEY CHECK DID NOT FIRE")
    except ValueError as e:
        print(f"missing key rejected: {str(e).splitlines()[1].strip()}")

    unknown = copy.deepcopy(block)
    unknown["K_base"] = [0.5, 1.2]
    try:
        RealisticParams.from_config(unknown)
        fails.append("an unknown K_base was accepted")
        print("UNKNOWN-KEY CHECK DID NOT FIRE")
    except ValueError as e:
        print(f"unknown key rejected: {str(e).splitlines()[1].strip()}")

    # ── 2. the knob actually moves the data ────────────────────────────────
    print("\n=== 2. cycle_period changes the generated data ===")
    base = generate_realistic_training_set(N, 42, p)
    edited = copy.deepcopy(block)
    edited["cycle_period"] = 720.0          # the N-6 defect, reinstated
    p720 = RealisticParams.from_config(edited)
    other = generate_realistic_training_set(N, 42, p720)

    same_sensors = np.array_equal(base.sensors, other.sensors)
    same_x0 = np.array_equal(base.x0s, other.x0s)
    d_sensors = float(np.abs(base.sensors - other.sensors).max())
    d_x0 = float(np.abs(base.x0s - other.x0s).max())
    print(f"cycle_period 1440 -> load swing (median half p-p) {hs_swing(base):.4f}")
    print(f"cycle_period  720 -> load swing (median half p-p) {hs_swing(other):.4f}")
    print(f"sensors identical: {same_sensors}   max abs delta {d_sensors:.4f}")
    print(f"x0s     identical: {same_x0}   max abs delta {d_x0:.4f}")
    if same_sensors or same_x0:
        fails.append("editing cycle_period did not change the data")

    # and the same params give the same data, i.e. it is the knob not the noise
    again = generate_realistic_training_set(N, 42, p)
    if not (np.array_equal(base.sensors, again.sensors)
            and np.array_equal(base.x0s, again.x0s)):
        fails.append("same params gave different data; generation is not seeded")
    print(f"same params reproduce byte-identically: "
          f"{np.array_equal(base.sensors, again.sensors)}")

    # ── 3. and it moves the hash ───────────────────────────────────────────
    print("\n=== 3. the same edit moves the distribution hash ===")
    h_base = canonical_hash(raw["distribution"])
    raw_edit = copy.deepcopy(raw)
    raw_edit["distribution"]["sampler"]["params"]["cycle_period"] = 720.0
    h_edit = canonical_hash(raw_edit["distribution"])
    print(f"cycle_period 1440 -> {h_base}")
    print(f"cycle_period  720 -> {h_edit}")
    if h_base == h_edit:
        fails.append("editing cycle_period did not move the distribution hash")

    # ── 4. v57 path intact ─────────────────────────────────────────────────
    print("\n=== 4. v57 reproduction path ===")
    v57_raw = yaml.safe_load(V57.read_text(encoding="utf-8"))
    v57_sampler = v57_raw["distribution"]["sampler"]
    print(f"v57 sampler kind={v57_sampler['kind']!r}  "
          f"params present={'params' in v57_sampler}")
    if v57_sampler.get("kind") != "v57" or "params" in v57_sampler:
        fails.append("v57 config no longer declares a bare kind: v57")
    a = generate_training_set(n_ic=N, seed=42)
    b = generate_training_set(n_ic=N, seed=42)
    print(f"v57 sampler still deterministic: "
          f"{np.array_equal(a.sensors, b.sensors)}")
    if not np.array_equal(a.sensors, b.sensors):
        fails.append("v57 sampler is not reproducible")
    if np.array_equal(a.sensors[:, :100], base.sensors[:, :100]):
        fails.append("v57 and realistic samplers produced the same profiles")
    print(f"v57 differs from realistic: "
          f"{not np.array_equal(a.sensors, base.sensors)}")

    # ── report ─────────────────────────────────────────────────────────────
    print("\n=== hashes now ===")
    for path in (CFG, V57):
        c = load_config(path)
        print(f"{path.name:26s} config {c.hash}  distribution {c.distribution_hash}")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
