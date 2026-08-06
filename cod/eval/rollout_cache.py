"""The model-independent half of a rollout, computed once per scenario.

DECISIONS N-13. Scoring the C-11 matrix on ANALYSIS_PLAN Amendment 1's primary
metric means rolling 117 checkpoints, and each rollout needs two things that do
not depend on the model at all:

  * the **reference rollout** — RK45 on `fast_rhs_np`, integrated continuously
    across the horizon — which is the truth every model is scored against;
  * the **cyclic-endpoint burn-ins**, several RK45 cycles per distinct window
    forcing, which are a property of `(K_w, Ta_w)` and nothing else.

Both were being recomputed for every checkpoint. Measured: 57 minutes for two
scenarios at 730 windows, almost all of it these two, so ~107 CPU-hours over the
matrix. Amendment 1 budgeted **4.6 GPU-hours** for the whole scoring pass on
exactly the right reasoning — *"the reference rollout and the cyclic burn-ins
are model-independent, so they are computed once per scenario and reused across
all 112 cell-seeds; only the model forward passes scale with the matrix"* — and
this module is what makes the code match the plan.

**The danger this cache creates, and how it is closed.** A stale cache is a
silent sentinel of the purest kind (PORT_LOG J-89): it returns a well-formed
reference for a scenario whose physics has since changed, and every model is
then scored against a truth that is quietly wrong. Nothing downstream can tell.

So the key includes a **fingerprint of the source that defines the reference** —
`cod/eval/rollout.py`, `cod/data/physics.py`, `cod/data/steady_state.py` — and
not merely the scenario parameters. Edit any of them and the key changes, the
old entry is never read again, and the reference is recomputed. This is
deliberately blunt: it will invalidate on a docstring edit, which costs an hour
of recompute, and the alternative is a cache that survives a change to
`fast_rhs_np`. That trade is not close.

**Bit-identity is the acceptance criterion, not closeness.** Both integrators
are deterministic and the residuals are reproducible to the last digit, so
"cached equals uncached" is checkable exactly. `audit_port/scripts/
39_rollout_cache_identical.py` asserts `==` on every array, not `allclose`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cod.data.physics import N_SENSORS, TW
from cod.data.steady_state import true_fixed_point_np
from cod.eval.rollout import (ReferenceRollout, cyclic_endpoint_theta,
                              reference_rollout, window_forcing)

#: Bumped by hand if the stored layout changes in a way the fingerprint cannot
#: see — a new array, a renamed field. The source fingerprint covers the
#: *values*; this covers the *shape of the file*.
CACHE_VERSION = 1

#: The files that define what the reference is. A change to any of them changes
#: every cached answer, so they are hashed into the key.
_SOURCE_FILES = (
    "cod/eval/rollout.py",
    "cod/data/physics.py",
    "cod/data/steady_state.py",
)

_ROOT = Path(__file__).resolve().parents[2]
_fingerprint_cache: dict = {}


def source_fingerprint() -> str:
    """A hash of the code that decides what the reference rollout is."""
    if "v" in _fingerprint_cache:
        return _fingerprint_cache["v"]
    h = hashlib.sha256()
    for rel in _SOURCE_FILES:
        p = _ROOT / rel
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    _fingerprint_cache["v"] = h.hexdigest()[:16]
    return _fingerprint_cache["v"]


def scenario_key(K_base: float, max_windows: int, *, T: float = TW,
                 n_eval: int = 20, rtol: float = 1e-9, atol: float = 1e-11,
                 steady_state=None) -> str:
    """Everything the cached values depend on, as one string.

    `steady_state` enters by name: passing `formula_B` to reproduce v57 gives a
    different reference and must not collide with the default.
    """
    ss = (steady_state or true_fixed_point_np).__name__
    payload = json.dumps({
        "version": CACHE_VERSION, "source": source_fingerprint(),
        "K_base": float(K_base), "max_windows": int(max_windows),
        "T": float(T), "n_eval": int(n_eval), "n_sensors": int(N_SENSORS),
        "rtol": float(rtol), "atol": float(atol), "steady_state": ss,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cyclic_endpoint_series(K_base: float, max_windows: int, T: float = TW,
                           steady_state=None) -> dict:
    """The burn-in cache `chi_lifetime_rollout` would have built, standalone.

    **The seeding sequence is reproduced exactly, and it has to be.**
    `cyclic_endpoint_theta` iterates to a fixed point with `tol = 1e-6` and
    returns as soon as the endpoint stops moving by that much, so *where the
    iteration starts changes the last digits of where it stops*. The rollout
    seeds each new window from the previous window's answer — the seasonal
    forcing moves ~0.01 degC per window, so it starts inside tolerance — and
    falls back to the window-mean equilibrium for the first one.

    Precomputing in a different order, or seeding every entry from its own
    equilibrium, would give values that are *equally valid* and *not identical*.
    That is the whole trap: the cached rollout would differ from the uncached
    one in the last digits, the difference would be far below any threshold, and
    nothing would ever flag it — while the acceptance criterion for this cache
    is bit-identity. So this loop mirrors `chi_lifetime_rollout`'s, including
    that `prev` advances on a cache hit as well as on a miss.
    """
    steady_state = steady_state or true_fixed_point_np
    cache: dict = {}
    prev = None
    for w in range(int(max_windows)):
        K_w, Ta_w, K_s, Ta_s = window_forcing(w, K_base, T, N_SENSORS)
        ck = (round(K_w, 9), round(Ta_w, 9))
        if ck not in cache:
            seed = prev if prev is not None else float(steady_state(K_w, Ta_w))
            cache[ck] = cyclic_endpoint_theta(K_s.astype(float),
                                              Ta_s.astype(float), seed, T=T)
        prev = cache[ck]
    return cache


def _store(path: Path, ref: ReferenceRollout, cyc: dict, meta: dict) -> None:
    keys = np.array([[a, b] for a, b in cyc], dtype=np.float64).reshape(-1, 2)
    vals = np.array(list(cyc.values()), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    # Written to a temporary name and moved into place, so an interrupted write
    # leaves no half-file that a later run would load as a complete reference.
    np.savez(tmp, years=ref.years, theta_TO_end=ref.theta_TO_end,
             theta_pts=ref.theta_pts, dp=ref.dp,
             reached_eol=np.array(ref.reached_eol),
             gas_end=(np.empty((0, 5)) if ref.gas_end is None else ref.gas_end),
             has_gas=np.array(ref.gas_end is not None),
             cyc_keys=keys, cyc_vals=vals,
             meta=np.array(json.dumps(meta)))
    tmp.replace(path)


def _load(path: Path):
    z = np.load(path, allow_pickle=False)
    ref = ReferenceRollout(
        K_base=float(json.loads(str(z["meta"]))["K_base"]),
        years=z["years"], theta_TO_end=z["theta_TO_end"],
        theta_pts=z["theta_pts"], dp=z["dp"],
        reached_eol=bool(z["reached_eol"]),
        gas_end=(z["gas_end"] if bool(z["has_gas"]) else None))
    cyc = {(float(a), float(b)): float(v)
           for (a, b), v in zip(z["cyc_keys"], z["cyc_vals"])}
    return ref, cyc


def load_or_compute(K_base: float, max_windows: int, *, T: float = TW,
                    n_eval: int = 20, rtol: float = 1e-9, atol: float = 1e-11,
                    steady_state=None, cache_dir: Path | None = None,
                    verbose: bool = True):
    """`(ReferenceRollout, cyc_cache)` for one scenario, from disk if possible.

    `cache_dir=None` disables the cache entirely and computes both, which is the
    path `39_rollout_cache_identical.py` compares against.

    The returned `cyc_cache` is a **fresh dict** every call. `chi_lifetime_rollout`
    writes into the cache it is handed, and one scenario's rollout must not be
    able to reach into another's — nor into the copy this function will hand out
    next time in the same process.
    """
    steady_state = steady_state or true_fixed_point_np

    if cache_dir is not None:
        key = scenario_key(K_base, max_windows, T=T, n_eval=n_eval, rtol=rtol,
                           atol=atol, steady_state=steady_state)
        path = Path(cache_dir) / f"scenario_{key}.npz"
        if path.is_file():
            ref, cyc = _load(path)
            if verbose:
                print(f"[cache] hit  K={K_base:.2f} n={max_windows} "
                      f"({path.name}) — reference and {len(cyc)} burn-ins "
                      "reused, no RK45")
            return ref, dict(cyc)
        if verbose:
            print(f"[cache] miss K={K_base:.2f} n={max_windows} — computing "
                  "the reference rollout and the burn-ins once")

    ref = reference_rollout(K_base, n_eval=n_eval, T=T,
                            steady_state=steady_state, max_windows=max_windows,
                            rtol=rtol, atol=atol)
    cyc = cyclic_endpoint_series(K_base, max_windows, T=T,
                                 steady_state=steady_state)

    if cache_dir is not None:
        _store(path, ref, cyc, {
            "K_base": float(K_base), "max_windows": int(max_windows),
            "T": float(T), "n_eval": int(n_eval), "rtol": rtol, "atol": atol,
            "steady_state": steady_state.__name__,
            "source_fingerprint": source_fingerprint(),
        })
        if verbose:
            print(f"[cache] wrote {path.name} — every later checkpoint on this "
                  "scenario reuses it")
    return ref, dict(cyc)


__all__ = ["CACHE_VERSION", "source_fingerprint", "scenario_key",
           "cyclic_endpoint_series", "load_or_compute"]
