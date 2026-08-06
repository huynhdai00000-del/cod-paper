#!/usr/bin/env python3
"""Is the cached rollout path bit-identical to the uncached one?

DECISIONS N-13 caches the model-independent half of a rollout — the reference
trajectory and the cyclic-endpoint burn-ins — so that scoring 117 checkpoints
computes them once per scenario instead of 117 times. That is a large speedup
sitting directly underneath the primary metric, which makes it exactly the kind
of optimisation that has to be proved rather than believed.

**The bar is `==`, not `allclose`.** Both integrators are deterministic: the
`--exact` self-test reproduced its residuals to the last digit across two
separate runs an hour apart. So "cached equals uncached" is an exact question,
and answering it approximately would throw away the only property that makes the
check conclusive. A tolerance here would hide precisely the failure mode this
cache can have.

Four checks:

  1. **The reference rollout.** Every array — years, theta_TO_end, theta_pts,
     dp, gas_end — and the `reached_eol` flag, cached against freshly computed.
  2. **The burn-in cache.** Same key set, and every value identical. This is the
     one with a real trap in it: `cyclic_endpoint_theta` iterates to a fixed
     point with `tol = 1e-6` and stops when the endpoint stops moving, so *where
     the iteration starts changes the last digits of where it stops*. The
     rollout seeds each window from the previous window's answer.
     `cyclic_endpoint_series` reproduces that sequence deliberately; seeding
     each entry from its own equilibrium instead would give values that are
     equally valid, differ in the last digits, and never be noticed.
  3. **The whole scenario, end to end.** `run_scenario` with the cache and
     without it, on the same model, compared on every reported quantity
     including Amendment 1's gas errors. This is what actually matters: checks 1
     and 2 could both pass and the wiring still hand the cache to the wrong
     argument.
  4. **The key discriminates.** A different scenario must not collide, and a
     change to the physics source must invalidate — checked by perturbing the
     fingerprint input rather than by reading the code that computes it.

Run:  python audit_port/scripts/39_rollout_cache_identical.py
      python audit_port/scripts/39_rollout_cache_identical.py --max-windows 60
Exit: 0 if every comparison is exact, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.steady_state import true_fixed_point_np  # noqa: E402
from cod.eval import rollout_cache as rc  # noqa: E402
from cod.eval.rollout import reference_rollout  # noqa: E402

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else '[FAIL]'} {name}" + (f" — {detail}" if detail
                                                      else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def _load24():
    spec = importlib.util.spec_from_file_location(
        "r24", Path(__file__).with_name("24_rollout_thermal_error.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exact_eq(a, b) -> tuple:
    """(identical, worst absolute difference). No tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, float("inf")
    if a.size == 0:
        return True, 0.0
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return bool(np.array_equal(a, b)), float(d.max())


def check_reference(K: float, n: int, cache_dir: Path) -> None:
    print("\n1. the reference rollout, cached against freshly computed")
    fresh = reference_rollout(K, max_windows=n,
                              steady_state=true_fixed_point_np)
    # First call populates; second reads back from disk.
    rc.load_or_compute(K, n, steady_state=true_fixed_point_np,
                       cache_dir=cache_dir, verbose=False)
    cached, _cyc = rc.load_or_compute(K, n, steady_state=true_fixed_point_np,
                                      cache_dir=cache_dir, verbose=False)
    for field in ("years", "theta_TO_end", "theta_pts", "dp", "gas_end"):
        a, b = getattr(fresh, field), getattr(cached, field)
        if a is None or b is None:
            check(f"{field} present on both sides", a is None and b is None,
                  f"fresh={type(a).__name__} cached={type(b).__name__}")
            continue
        same, worst = _exact_eq(a, b)
        check(f"{field} identical", same, f"worst abs diff {worst:.3e}")
    check("reached_eol identical", fresh.reached_eol == cached.reached_eol,
          f"{fresh.reached_eol} vs {cached.reached_eol}")
    check("K_base survives the round trip", fresh.K_base == cached.K_base,
          f"{fresh.K_base} vs {cached.K_base}")


def check_burnins(K: float, n: int, cache_dir: Path) -> None:
    print("\n2. the burn-in cache, and the seeding sequence it depends on")
    fresh = rc.cyclic_endpoint_series(K, n, steady_state=true_fixed_point_np)
    _ref, cached = rc.load_or_compute(K, n, steady_state=true_fixed_point_np,
                                      cache_dir=cache_dir, verbose=False)
    check("same key set", set(fresh) == set(cached),
          f"{len(fresh)} fresh vs {len(cached)} cached")
    if set(fresh) == set(cached):
        worst = max((abs(fresh[k] - cached[k]) for k in fresh), default=0.0)
        check("every burn-in value identical",
              all(fresh[k] == cached[k] for k in fresh),
              f"worst abs diff {worst:.3e} over {len(fresh)} entries")

    # The trap, made visible: seed every entry from its own equilibrium instead
    # of from the previous window, and the values move. If this comes out
    # identical the tolerance is loose enough that the sequence does not matter
    # and the warning in `cyclic_endpoint_series` is overcautious; if it moves,
    # reproducing the sequence is load-bearing and the code that does it is
    # doing real work.
    from cod.data.physics import N_SENSORS
    from cod.eval.rollout import cyclic_endpoint_theta, window_forcing
    naive = {}
    for w in range(n):
        K_w, Ta_w, K_s, Ta_s = window_forcing(w, K, rc.TW, N_SENSORS)
        ck = (round(K_w, 9), round(Ta_w, 9))
        if ck not in naive:
            naive[ck] = cyclic_endpoint_theta(
                K_s.astype(float), Ta_s.astype(float),
                float(true_fixed_point_np(K_w, Ta_w)), T=rc.TW)
    diffs = [abs(naive[k] - fresh[k]) for k in fresh if k in naive]
    worst = max(diffs, default=0.0)
    n_moved = sum(1 for d in diffs if d != 0.0)
    print(f"  note  seeding each burn-in from its own equilibrium instead moves "
          f"{n_moved}/{len(diffs)} values, worst {worst:.3e} degC — which is "
          "why the precompute mirrors the rollout's sequence rather than "
          "recomputing them independently")


def check_end_to_end(K: float, n: int, cache_dir: Path) -> None:
    print("\n3. the whole scenario, with the cache and without it")
    r24 = _load24()
    model = r24.load_exact()
    device = torch.device("cpu")

    ref_u, free_u, teach_u = r24.run_scenario(model, K, n, device,
                                              cache_dir=None)
    ref_c, free_c, teach_c = r24.run_scenario(model, K, n, device,
                                              cache_dir=cache_dir)

    for label, a, b in (("reference", ref_u, ref_c),
                        ("free-running", free_u, free_c),
                        ("teacher-forced", teach_u, teach_c)):
        for field in ("years", "theta_TO_end", "dp", "gas_end"):
            x, y = getattr(a, field, None), getattr(b, field, None)
            if x is None and y is None:
                continue
            same, worst = _exact_eq(x, y)
            check(f"{label}.{field} identical", same,
                  f"worst abs diff {worst:.3e}")

    # And the reported quantities, which is what a reader would ever see.
    j = min(len(ref_u.gas_end), len(free_u.gas_end)) - 1
    gu = np.asarray(free_u.gas_end[j]) - np.asarray(ref_u.gas_end[j])
    gc = np.asarray(free_c.gas_end[j]) - np.asarray(ref_c.gas_end[j])
    same, worst = _exact_eq(gu, gc)
    check("Amendment 1 gas errors identical", same,
          f"worst abs diff {worst:.3e} ppm; uncached "
          + ", ".join(f"{g} {v:+.4g}" for g, v in zip(r24.GAS_NAMES, gu)))

    m = min(len(ref_u.theta_TO_end), len(free_u.theta_TO_end))
    bu = float((free_u.theta_TO_end[:m] - ref_u.theta_TO_end[:m]).mean())
    bc = float((free_c.theta_TO_end[:m] - ref_c.theta_TO_end[:m]).mean())
    check("thermal bias identical", bu == bc, f"{bu:+.6e} vs {bc:+.6e}")


def check_key(K: float, n: int) -> None:
    print("\n4. does the key discriminate")
    k = rc.scenario_key(K, n, steady_state=true_fixed_point_np)
    check("a different K gives a different key",
          rc.scenario_key(K + 0.05, n,
                          steady_state=true_fixed_point_np) != k)
    check("a different horizon gives a different key",
          rc.scenario_key(K, n + 1, steady_state=true_fixed_point_np) != k)
    check("a different steady_state gives a different key",
          rc.scenario_key(K, n,
                          steady_state=_other_steady_state()) != k)

    # The source fingerprint is the guard against a stale cache surviving a
    # change to the physics. Perturbed by pointing it at a file that differs,
    # rather than by trusting that it reads what it claims to.
    real = rc.source_fingerprint()
    saved_files, saved_cache = rc._SOURCE_FILES, dict(rc._fingerprint_cache)
    try:
        rc._fingerprint_cache.clear()
        rc._SOURCE_FILES = tuple(list(saved_files) + ["cod/eval/metrics.py"])
        moved = rc.source_fingerprint()
    finally:
        rc._SOURCE_FILES = saved_files
        rc._fingerprint_cache.clear()
        rc._fingerprint_cache.update(saved_cache)
    check("the fingerprint moves when the hashed source set changes",
          moved != real, f"{real} vs {moved}")
    check("and is stable when it does not", rc.source_fingerprint() == real)

    missing = [f for f in rc._SOURCE_FILES if not (ROOT / f).is_file()]
    check("every file the fingerprint hashes exists", not missing,
          f"missing: {missing} — a fingerprint over a missing file hashes the "
          "literal b'<missing>' and would not move when that file is restored")


def _other_steady_state():
    from cod.data.steady_state import formula_A
    return formula_A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=0.95)
    ap.add_argument("--max-windows", type=int, default=40,
                    help="Short by default: the comparison is exact, so it "
                         "does not need a long horizon to be conclusive.")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rollout_cache_check_"))
    print(f"[tmp] {tmp}")
    print(f"[cfg] K = {args.K}, {args.max_windows} windows, "
          f"source fingerprint {rc.source_fingerprint()}")
    try:
        check_reference(args.K, args.max_windows, tmp)
        check_burnins(args.K, args.max_windows, tmp)
        check_end_to_end(args.K, args.max_windows, tmp)
        check_key(args.K, args.max_windows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} comparison(s) not exact:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[PASS] the cached path is bit-identical to the uncached one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
