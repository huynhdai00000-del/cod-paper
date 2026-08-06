#!/usr/bin/env python3
"""What does the N-13 cache actually save, measured on a real checkpoint?

DECISIONS N-13 said the scoring pass cost ~107 CPU-hours against ANALYSIS_PLAN
Amendment 1's budgeted 4.6, because the reference rollout and the cyclic
burn-ins — both model-independent — were recomputed for every checkpoint. This
runs the same command three times through the real CLI and times it: once with
the cache empty, twice with it warm.

Timed end to end as a **subprocess**, deliberately. An in-process timing would
share the burn-in dict, the imported modules and the warm allocator, and would
measure something no user experiences. The scoring pass is 117 separate
invocations, so that is what is measured.

A real neural checkpoint, not `ExactModel`. `ExactModel` is itself RK45, so its
model rollout costs about what the reference costs and the saving would look
smaller than it is for every cell in the matrix.

The printed gas errors are also the point: they must be **identical** across all
three runs. `39_rollout_cache_identical.py` proves bit-identity at the library
level; this shows the same thing survives the CLI, the JSON round trip and a
process boundary.

Run:  python audit_port/scripts/40_rollout_cache_speedup.py
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = (ROOT / "results" /
        "cod_matrix_mionet_in_cascade_s11_b23cd5d8fa43ff88_j94a" / "model.pt")
CACHE = ROOT / "audit_port" / "_rollout_cache"
N = 120

if not CKPT.is_file():
    raise SystemExit(
        f"no checkpoint at {CKPT}.\n"
        "  Any trained model.pt works — point CKPT at one. It must be a real\n"
        "  network rather than ExactModel, or the saving is understated.")

shutil.rmtree(CACHE, ignore_errors=True)
cmd = [sys.executable, str(ROOT / "audit_port" / "scripts" /
                           "24_rollout_thermal_error.py"),
       "--checkpoint", str(CKPT), "--max-windows", str(N),
       "--k-scenarios", "0.95",
       "--out", str(ROOT / "audit_port" / "scripts" / "_time_cache.md")]

times, gas_lines = [], []
for label in ("cold (cache empty)", "warm (cache hit)", "warm again"):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    times.append(dt)
    hit = "[cache] hit" in r.stdout
    gas = [ln for ln in r.stdout.splitlines() if "gas ppm at window" in ln]
    print(f"{label:22s} {dt:7.1f} s   cache_hit={hit}")
    if gas:
        print(f"                       {gas[0].strip()}")
        gas_lines.append(gas[0].strip())
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("subprocess failed")

cold, warm = times[0], min(times[1:])
print(f"\ncold {cold:.1f} s, warm {warm:.1f} s  ->  saved {cold - warm:.1f} s "
      f"per additional checkpoint ({100 * (cold - warm) / cold:.0f}% of the "
      f"cold run), speedup {cold / warm:.1f}x at {N} windows")

if len(set(gas_lines)) != 1:
    print("\n[FAIL] the reported gas errors are NOT identical across the three "
          "runs, so the cache changed the answer:")
    for g in gas_lines:
        print(f"  {g}")
    raise SystemExit(1)
print("and the reported gas errors are identical across all three runs.")
