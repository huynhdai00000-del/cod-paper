# Distribution freeze

Re-frozen 2026-07-29 after the realistic sampler was put on the config path.
The previous freeze, recorded earlier the same day, is superseded — it hashed a
block that the training path largely did not read. See
`CHANGELOG_DISTRIBUTION.md` for the supersession record and the old hashes.

The gate came first, both times: a hash recorded against a package that cannot
run is worthless. See §4.

## 1. The frozen hashes

`distribution_hash` is `canonical_hash(raw["distribution"])`, reported separately
from the whole-config hash so that "the training distribution was frozen before
any model was trained" is checkable rather than asserted (`cod/config.py`).

| config | sampler | config hash | distribution hash |
|---|---|---|---|
| `configs/example_cod_seed1.yaml` | `realistic` | `c5715352d1ddf87e` | **`fc4cb76c3b32ec17`** |
| `configs/v57_faithful.yaml` | `v57` | `4cc034c6b60de703` | **`3ad5f68876934c75`** |

Enforce on the command line:

```
python scripts/run.py --config configs/example_cod_seed1.yaml \
                      --freeze-hash fc4cb76c3b32ec17
```

Any intentional change goes in `CHANGELOG_DISTRIBUTION.md` with date and reason,
the hash here is updated, and the change is disclosed in the paper.

The two hashes move independently, and that is the point of reporting them
separately. Raising `training.convergence.max_wall_seconds` from 3600 to 7200 on
2026-07-30 moved the config hash from `95b56b1d79ac7c40` to `c5715352d1ddf87e`
and left the distribution hash untouched — a budget is not a distribution. Only
a change to the latter is a change to what the model is being trained on, and
only that needs a `CHANGELOG_DISTRIBUTION.md` entry.

## 2. What the hash now covers, and what it still does not

**Covered, and enforced.** Every field of `RealisticParams` — all 22 — must
appear in `distribution.sampler.params`. `RealisticParams.from_config` rejects
the config if one is missing, so no sampler parameter can take a Python default
that the hashed text does not state. It equally rejects a key that is *not* a
field, which is the same failure seen from the other side: the previous config
carried `K_base`, `ambient_base`, `ambient_amplitude` and nine `profile_families`
that reached no sampler, so editing them moved the hash and changed no data.
Adding a field to `RealisticParams` now breaks every config until the config
declares it. That is intended.

It also rejects a `families`/`weights` length mismatch, weights that do not sum
to 1 (which `rng.choice` would silently renormalise), and a family name
`make_realistic_day` has no branch for (which would fall through to `multi_step`
without saying so).

**Not covered: the sampler code itself.** A config hash cannot see a change to
`cod/data/realistic.py` that alters behaviour without altering a parameter. Fix 7
is the worked example — it moved the median realised hot-spot swing from 11.20 to
13.18 degC and no hash moved, because at that time no config named the sampler at
all. That specific hole is closed (`cycle_period` is now a hashed parameter), but
the general one is structural: git history pins the code, and this hash is not a
substitute for it. Quote both when the paper describes the protocol.

**Not covered: the test set's own ranges.** `distribution` pins what the model
trains on. The evaluation tiers name `n_cases` and `seed`, but
`build_test_set`'s CK and TV ranges — `U(0.4, 1.4)`, `U(0.5, 1.2)`, amplitude
0.20, phase pi/3 — are hardcoded in `cod/data/generate.py`. That is the same
class of gap as the one just closed, one layer over, and it is why audit B-5 and
M-8 are recorded against the test set rather than against the sampler. Not
changed here: the seed-999 benchmark is what every stored gate number was scored
on, so it is frozen by being immutable rather than by being hashed.

**Not covered: physics constants.** `tau_oil`, `DTheta_oil_R`, `n_exp`, `k_gen`,
`k_dis`, `E_act` live in `cod/data/physics.py`. O-3 and O-11 close as declared
limitations rather than calibrations, so these are fixed by the benchmark
definition and pinned by git, not by this hash.

## 3. Two samplers, one default

`distribution.sampler.kind` selects, and it sits inside the hashed block because
*which sampler drew the data* is part of the distribution.

`realistic` (default, `cod/data/realistic.py`) is the path for anything whose
numbers go in the paper. The day is drawn first and the initial condition is the
periodic state of that day's own load pattern read at the window's offset, which
is what audit M-9 is about.

`v57` (`cod/data/profiles.py` via `generate_training_set`) exists to reproduce
`transformer_training_v57.npz` byte for byte and to keep the Phase 1 gates
reproducing. **Its ranges are deliberately hardcoded and deliberately not
configurable**, and `run.py` raises if a `params` block is supplied for it. A
reproduction gate that a YAML edit can move is not a gate; those constants are a
frozen historical artifact, not a second source of truth competing with the
config. `generate_training_set` is docstring-deprecated for new work.

## 4. The gates this was written behind

**The knob moves the data** — `audit_port/scripts/20_verify_config_binding.py`,
which is the direct refutation of the failure the last freeze documented:

| check | result |
|---|---|
| all 22 params present, extras rejected | both enforcement paths fire |
| `cycle_period` 1440 -> 720 changes `sensors` | max abs delta **15.17** |
| `cycle_period` 1440 -> 720 changes `x0s` | max abs delta **8.44** |
| median load swing (half p-p) at 1440 / 720 | 0.1045 / **0.1261** |
| same params reproduce byte-identically | yes |
| the same edit moves the distribution hash | `fc4cb76c…` -> `cefe1e0e…` |
| v57 path deterministic and distinct | yes |

The swing figures are the physics of N-6 read back through the config: a 720 min
period puts a whole cycle inside the 720 min window, so the window sees a larger
load excursion. Setting `cycle_period: 720.0` reinstates the N-6 defect, and it
is now a one-line, hash-visible, reviewable edit rather than a property of the
code.

**Phase 1 still reproduces.** `scripts/verify_phase1.py --gate 1` passes on the
v57 path: theta_TO 1.5, c_H2 1.3, c_C2H2 2.3, c_C2H4 1.6, c_CO 1.1, c_CO2 1.1,
overall 1.5, 99/100 under 10%. The gates read stored checkpoints and the stored
npz directly and never load a config, so changing `v57_faithful.yaml`'s hash
cannot affect them.

**The pipeline runs end to end on both samplers.**
`run.py --config configs/example_cod_seed1.yaml --freeze-hash fc4cb76c3b32ec17
--max-epochs 40 --n-ic 48 --n-test 6` completes, and `run.json` records all 22
resolved sampler parameters under `data_provenance.realistic_params`. The v57
config completes on its own path and records its own flags instead.

Both smoke runs report `NOT CONVERGED (stop_reason=epoch_budget)`, which is
correct at 40 epochs on 48 ICs. **None of their metrics are results.** The
package refusing to dress a 40-epoch run as a performance number is the
convergence rule working.

## 5. What has to happen next, in order

1. Retrain (O-5) on `fc4cb76c3b32ec17`. Freezing after training is not a
   protocol, it is a record; this freeze is only meaningful because nothing has
   been trained on it yet.
2. Re-run the swing-fidelity check (`18_swing_fidelity.py`) on the retrained
   checkpoint. Its current numbers are v57's, scored in the v57 distribution, and
   `SWING_FIDELITY.md` §5.3 says they must be rerun here.
3. Re-run `19_verify_bias_fix.py`'s rollout against the retrained model to get
   the real thermal rollout error, which O-9 left open and which decides whether
   an end-of-life number is publishable.
